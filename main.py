"""
main.py
-------
F.R.I.D.A.Y. Master Lab Automation Voice Assistant:
- Controls Master ESP8266 controller running at `http://192.168.4.1` (from `index.html`)
  with full Master Controls, 3 Zones (28 relays total), and 4 Motorized Curtains.
- Authentic MCU F.R.I.D.A.Y. persona (calm, crisp, intelligent Irish AI assistant;
  speaks directly and never spams 'Boss').
- Hands-Free Double-Clap Activation (Iron Man style, zero startup menus).
- Cloud Studio Female Voices (Gemini TTS + Edge Neural Irish Voice, zero pyttsx3/robotic voice).
- High speed with local failover.

Lifecycle:
  1. Connects to Master ESP at `http://192.168.4.1` (or offline Simulation Mode).
  2. Displays the laboratory status dashboard and calibrates microphone once.
  3. Enters silent STANDBY waiting for a double clap.
  4. Double clap activates F.R.I.D.A.Y. -> speaks "All lab systems online. What do you need?"
     and enters hands-free continuous voice mode.
  5. Saying "bye", "goodbye", "sleep", or "standby" speaks "Standing by. Call me when you need me."
     and returns silently to STANDBY until your next double clap.
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
    r"go\s+to\s+sleep|sleep\s+mode|that(?:'s|\s+is)\s+all|"
    r"so\s*jao|sojao|alvida|chalo\s*bye|bas\s*karo|khatam|standby\s*par\s*jao)\b|"
    r"[\u0900-\u097f]*(?:अलविदा|सो\s*जाओ)",
    flags=re.IGNORECASE,
)


def is_standby_phrase(text: str) -> bool:
    """Return True if the user asks to return F.R.I.D.A.Y. to standby."""
    return bool(_STANDBY_PATTERN.search(text.strip()))


def display_dashboard(base_url: str, status_data: Dict[str, Any]) -> None:
    """Print the complete Lab Automation dashboard (from `index.html`)."""
    divider = "=" * 70
    subdivider = "-" * 70
    voice_label = (
        f"Edge Neural ({config.EDGE_FRIDAY_VOICE})"
        if config.PREFER_FAST_EDGE_VOICE
        else f"Gemini Cloud TTS ({config.TTS_MODEL}, voice={config.GEMINI_VOICE_NAME}) + Edge fallback"
    )

    print(f"\n{divider}")
    print("      F.R.I.D.A.Y.  |  LABORATORY AUTOMATION SYSTEM (ESP8266)")
    print(divider)
    print(f"  Master Controller : {base_url}")
    print(f"  AI Brain Engine   : {config.LLM_MODEL} (thinking={config.LLM_THINKING_LEVEL})")
    print(f"  Voice Profile     : {voice_label}")
    print(f"  Wake Activation   : Double Clap ({config.CLAP_COUNT_REQUIRED}x)")
    print(subdivider)

    # Zone 1 (Zone A)
    z1 = status_data.get("zone_1", {})
    z1_summary = " ".join(f"R{r}:{z1.get(r, 'OFF')}" for r in range(1, 13))
    print(f"  ZONE A [Relays 1-12] : {z1_summary}")

    # Zone 2 (Zone B)
    z2 = status_data.get("zone_2", {})
    z2_fans = f"Fan1:{z2.get(1, 'OFF')}  Fan2:{z2.get(2, 'OFF')}  Fan3:{z2.get(3, 'OFF')}"
    z2_lights = f"L1:{z2.get(4, 'OFF')} L2:{z2.get(5, 'OFF')} L3:{z2.get(6, 'OFF')} L4:{z2.get(7, 'OFF')} L5:{z2.get(8, 'OFF')}"
    print(f"  ZONE B [Fans 1-3]    : {z2_fans}")
    print(f"  ZONE B [Lights 1-5]  : {z2_lights}")

    # Zone 3 (Zone C)
    z3 = status_data.get("zone_3", {})
    z3_summary = " ".join(f"R{r}:{z3.get(r, 'OFF')}" for r in range(1, 9))
    print(f"  ZONE C [Relays 1-8]  : {z3_summary}")

    # Curtains
    curtains = status_data.get("curtains", {})
    c_summary = " ".join(f"C{c}:{curtains.get(c, 'READY')}" for c in range(1, 5))
    print(f"  CURTAINS [1-4]       : {c_summary}")

    print(divider)
    print("  System is on STANDBY. Clap twice to activate. Press Ctrl+C to exit.\n")


def execute_lab_actions(
    base_url: str,
    commands: list[nlp_processor.LabAction],
) -> list[str]:
    """Execute parsed lab directives on the Master ESP8266 controller."""
    feedback = []

    for cmd in commands:
        print(f"[Action]: type={cmd.target_type}, target={cmd.target}, action={cmd.action}")

        if cmd.target_type == "master":
            state_val = 1 if cmd.action in {"on", "up"} else 0
            ok, msg = nodemcu_client.send_master(base_url, cmd.target, state_val)
            print(f"[Master ESP]: {msg}")
            if not ok:
                feedback.append(msg)

        elif cmd.target_type == "zone":
            zone_num = cmd.zone if cmd.zone in {1, 2, 3} else 1
            state_val = 1 if cmd.action in {"on", "up"} else 0
            ok, msg = nodemcu_client.send_zone(base_url, zone_num, state_val)
            print(f"[Master ESP]: {msg}")
            if not ok:
                feedback.append(msg)

        elif cmd.target_type == "relay":
            zone_num = cmd.zone if cmd.zone in {1, 2, 3} else 1
            relay_num = cmd.relay if cmd.relay else 1
            ok, msg = nodemcu_client.send_relay(base_url, zone_num, relay_num)
            print(f"[Master ESP]: {msg}")
            if not ok:
                feedback.append(msg)

        elif cmd.target_type == "curtain":
            curtain_num = cmd.curtain if cmd.curtain is not None else 0
            if cmd.action in {"up", "open"}:
                act_code = 1
            elif cmd.action in {"down", "close"}:
                act_code = 2
            else:
                act_code = 0  # stop
            ok, msg = nodemcu_client.send_curtain(base_url, curtain_num, act_code)
            print(f"[Master ESP]: {msg}")
            if not ok:
                feedback.append(msg)

    return feedback


def _optimistically_update_status(
    status_data: Dict[str, Any],
    commands: list[nlp_processor.LabAction],
) -> Dict[str, Any]:
    """Instantly update in-memory lab state without blocking on micro-controller HTTP scrapes."""
    for cmd in commands:
        if cmd.target_type == "master":
            state_str = "ON" if cmd.action in {"on", "up"} else "OFF"
            if cmd.target in {"lights", "all"}:
                for r in range(4, 9):
                    status_data.setdefault("zone_2", {})[r] = state_str
            if cmd.target in {"fans", "all"}:
                for r in range(1, 4):
                    status_data.setdefault("zone_2", {})[r] = state_str
            if cmd.target == "all":
                for r in range(1, 13):
                    status_data.setdefault("zone_1", {})[r] = state_str
                for r in range(1, 9):
                    status_data.setdefault("zone_3", {})[r] = state_str
        elif cmd.target_type == "zone":
            z_key = f"zone_{cmd.zone or 1}"
            state_str = "ON" if cmd.action in {"on", "up"} else "OFF"
            for r in status_data.get(z_key, {}):
                status_data[z_key][r] = state_str
        elif cmd.target_type == "relay":
            z_key = f"zone_{cmd.zone or 1}"
            r_idx = cmd.relay or 1
            curr = status_data.get(z_key, {}).get(r_idx, "OFF")
            new_st = "OFF" if curr == "ON" else "ON" if cmd.action == "toggle" else cmd.action.upper()
            status_data.setdefault(z_key, {})[r_idx] = new_st
        elif cmd.target_type == "curtain":
            c_idx = cmd.curtain if cmd.curtain is not None else 0
            act_str = "UP" if cmd.action == "up" else "DOWN" if cmd.action == "down" else "STOPPED"
            if c_idx == 0:
                for c in range(1, 5):
                    status_data.setdefault("curtains", {})[c] = act_str
            else:
                status_data.setdefault("curtains", {})[c_idx] = act_str
    return status_data


def process_user_input(
    user_text: str,
    base_url: str,
    current_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process user utterance through F.R.I.D.A.Y.'s brain:
    1. Instant hardware command execution (< 30ms).
    2. Optimistic in-memory status update (zero lag).
    3. Simultaneous authentic MCU voice response.
    """
    cleaned = user_text.strip()
    if not cleaned:
        return current_status

    brain_output = nlp_processor.understand_command(cleaned, current_status)
    hardware_errors = []

    if brain_output.commands:
        # Check if user specifically requested a status query
        is_status_query = any(c.target_type == "status" for c in brain_output.commands)

        hardware_errors = execute_lab_actions(base_url, brain_output.commands)

        if is_status_query:
            # Re-fetch live states from micro-controller
            ok, refreshed = nodemcu_client.get_status(base_url)
            if ok:
                current_status = refreshed
        else:
            # Instant zero-lag optimistic in-memory update
            current_status = _optimistically_update_status(current_status, brain_output.commands)

    if hardware_errors:
        speech_engine.speak(
            f"Encountered an issue communicating with the master controller. {hardware_errors[0]}"
        )
    else:
        speech_engine.speak(brain_output.reply)

    return current_status


def run_active_voice_mode(
    base_url: str,
    current_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Hands-free continuous voice loop activated automatically by clapping.
    Returns silently to STANDBY when the user says 'bye', 'goodbye', or 'sleep'.
    """
    print("\n--- ACTIVE VOICE MODE (Say 'bye', 'sleep', or 'standby' to return) ---")

    while True:
        utterance = speech_engine.listen()
        if not utterance:
            continue

        if is_standby_phrase(utterance):
            goodbye_reply = (
                "Theek hai, main standby par ja rahi hoon. Jab zaroorat ho, bula lijiyega."
                if nlp_processor.is_hindi_or_hinglish(utterance)
                else config.STANDBY_GOODBYE_RESPONSE
            )
            speech_engine.speak(goodbye_reply)
            print("--- RETURNING TO STANDBY ---\n")
            time.sleep(config.CLAP_COOLDOWN_SEC)
            return current_status

        current_status = process_user_input(utterance, base_url, current_status)


def main() -> None:
    """Entrypoint: Direct STANDBY <-> ACTIVE clap-wake loop."""
    print("\nInitializing F.R.I.D.A.Y. Lab Automation System...")

    # 1. Connect to Master ESP at http://192.168.4.1 (or Simulation Mode)
    base_url = nodemcu_client.connect_to_nodemcu()

    # 2. Fetch initial lab status and display the dashboard
    _, current_status = nodemcu_client.get_status(base_url)
    display_dashboard(base_url, current_status)

    # 3. Calibrate microphone ONCE at startup
    speech_engine.init_microphone()

    # 4. Pre-warm Irish female voice cache in background for ~0ms speech latency
    speech_engine.warmup_tts_cache()

    # 5. Silent STANDBY waiting for double clap
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
