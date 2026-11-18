"""
main.py
-------
F.R.I.D.A.Y. Smart Home Voice Assistant (NodeMCU ESP8266 + Gemini 3.8 Flash +
Cloud Female TTS + Double-Clap Standby/Wake).

Lifecycle (Zero startup menus):
  1. Connects to NodeMCU on shared Wi-Fi, shows live relay states, and calibrates
     the microphone once.
  2. Immediately enters silent STANDBY mode waiting for a double clap.
  3. Double clap wakes F.R.I.D.A.Y. -> speaks "All systems online." and enters
     hands-free Voice Command Mode automatically.
  4. Saying "bye", "quit", "goodbye", "standby", or "stop listening" speaks
     "Bye, it was great talking to you." and returns silently to STANDBY until
     your next double clap. (Press Ctrl+C to exit the program.)
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any, Dict

import clap_detector
import config
import nlp_processor
import nodemcu_client
import speech_engine

_STANDBY_PATTERN = re.compile(
    r"\b(?:bye|goodbye|good\s+bye|quit|exit|stop\s+listening|standby|stand\s+by|"
    r"go\s+to\s+sleep|sleep\s+mode|that(?:'s|\s+is)\s+all)\b",
    flags=re.IGNORECASE,
)


def is_standby_phrase(text: str) -> bool:
    """Return True if the user asked to return F.R.I.D.A.Y. to standby."""
    return bool(_STANDBY_PATTERN.search(text.strip()))


def display_status(base_url: str, status_data: Dict[str, Any]) -> None:
    """Print the live status of all 4 NodeMCU relay channels."""
    divider = "=" * 66
    voice_label = (
        f"Edge Neural ({config.EDGE_FRIDAY_VOICE})"
        if config.PREFER_FAST_EDGE_VOICE
        else f"Gemini Cloud TTS ({config.TTS_MODEL}, voice={config.GEMINI_VOICE_NAME}) + Edge fallback"
    )
    print(f"\n{divider}")
    print("   F.R.I.D.A.Y.  |  GEMINI SMART HOME CONTROLLER (NodeMCU ESP8266)")
    print(divider)
    print(f"  NodeMCU Link : {base_url}")
    print(f"  LLM Brain    : {config.LLM_MODEL} (thinking_level={config.LLM_THINKING_LEVEL})")
    print(f"  Voice Engine : {voice_label}")
    print(f"  Wake Trigger : Double Clap ({config.CLAP_COUNT_REQUIRED}x)")
    print("-" * 66)
    for key in config.get_all_device_keys():
        name = config.get_device_name(key)
        state = str(status_data.get(key, "UNKNOWN")).upper()
        print(f"    - {key:<7} ({name:<15}) : {state}")
    print(f"{divider}")
    print("  System is on STANDBY. Clap twice to activate. Press Ctrl+C to exit.\n")


def process_user_input(
    user_text: str,
    base_url: str,
    current_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send user input to F.R.I.D.A.Y.'s Gemini brain, execute any returned relay
    commands on the NodeMCU, refresh `current_status`, and speak her reply.
    """
    cleaned = user_text.strip()
    if not cleaned:
        return current_status

    brain_output = nlp_processor.understand_command(cleaned, current_status)
    hardware_errors = []
    status_requested = False

    for cmd in brain_output.commands:
        print(f"[Action]: device='{cmd.device}', action='{cmd.action}'")
        if cmd.action == "status":
            status_requested = True
            ok, refreshed = nodemcu_client.get_status(base_url)
            if ok:
                current_status = refreshed
            else:
                hardware_errors.append(refreshed.get("error", "Could not read status."))
        else:
            ok, msg = nodemcu_client.send_command(base_url, cmd.device, cmd.action)
            print(f"[NodeMCU]: {msg}")
            if not ok:
                hardware_errors.append(msg)

    if brain_output.commands and not status_requested:
        ok, refreshed = nodemcu_client.get_status(base_url)
        if ok:
            current_status = refreshed

    if hardware_errors:
        speech_engine.speak(
            f"Couldn't reach the NodeMCU controller. {hardware_errors[0]}"
        )
    else:
        speech_engine.speak(brain_output.reply)

    return current_status


def run_active_voice_mode(
    base_url: str,
    current_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Hands-free continuous voice loop activated automatically after clapping.
    Returns silently to STANDBY when the user says 'bye', 'quit', or 'standby'.
    """
    print("\n--- ACTIVE VOICE MODE (say 'bye' or 'quit' to return to standby) ---")

    while True:
        utterance = speech_engine.listen()
        if not utterance:
            continue

        if is_standby_phrase(utterance):
            speech_engine.speak(config.STANDBY_GOODBYE_RESPONSE)
            print("--- RETURNING TO STANDBY ---\n")
            time.sleep(config.CLAP_COOLDOWN_SEC)
            return current_status

        current_status = process_user_input(utterance, base_url, current_status)


def main() -> None:
    """Direct STANDBY <-> ACTIVE clap-wake loop (no startup menu)."""
    print("\nInitializing F.R.I.D.A.Y. Smart Home System...")

    # 1. Connect to NodeMCU on shared Wi-Fi (or Simulation Mode)
    base_url = nodemcu_client.connect_to_nodemcu()

    # 2. Fetch and display initial status of all 4 relays
    _, current_status = nodemcu_client.get_status(base_url)
    display_status(base_url, current_status)

    # 3. Calibrate microphone ONCE at startup
    speech_engine.init_microphone()

    # 4. Stay silently on STANDBY until a double clap activates Voice Mode
    try:
        while True:
            woke = clap_detector.wait_for_clap()
            if not woke:
                time.sleep(0.5)
                continue

            speech_engine.speak(config.CLAP_WAKE_RESPONSE)
            current_status = run_active_voice_mode(base_url, current_status)

    except KeyboardInterrupt:
        print("\n[F.R.I.D.A.Y.]: Shutting down cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
