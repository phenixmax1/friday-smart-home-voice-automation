"""
nlp_processor.py
----------------
F.R.I.D.A.Y. AI Brain for Lab Automation:
- Models Tony Stark's authentic MCU F.R.I.D.A.Y. (Irish character, razor-sharp,
  composed, witty, highly efficient, and avoids repetitive titles like 'Boss').
- Fast Gemini structured output with fail-fast HTTP options and automatic
  failover (`gemini-3.8-flash` -> `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite`).
- Controls the entire laboratory infrastructure from `index.html`:
  * Master: All Lights, All Fans, All Devices (On / Off)
  * Zones: Zone A (Relays 1-12), Zone B (Fans 1-3, Lights 1-5), Zone C (Relays 1-8)
  * Motorized Curtains: Curtains 1-4 & All Curtains (Up / Down / Stop)
- Comprehensive local backup NLP brain so live demos work without latency spikes
  even if cloud APIs encounter 503 or network glitches.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional

from google import genai
from pydantic import BaseModel, Field

import config


class LabAction(BaseModel):
    """Single lab hardware control directive."""

    target_type: Literal["master", "zone", "relay", "curtain", "status"] = Field(
        description="Type of control: 'master', 'zone', 'relay', 'curtain', or 'status'"
    )
    target: str = Field(
        description="Target identifier: 'all', 'lights', 'fans', 'zone_1', 'zone_2', 'zone_3', 'fan_1', 'light_2', 'curtain_1', 'all_curtains', etc."
    )
    action: Literal["on", "off", "up", "down", "stop", "toggle", "status"] = Field(
        description="Action to execute"
    )
    zone: Optional[int] = Field(default=None, description="Zone number 1, 2, or 3 if applicable")
    relay: Optional[int] = Field(default=None, description="Relay index (1..12) if applicable")
    curtain: Optional[int] = Field(default=None, description="Curtain index (1..4, or 0 for all) if applicable")


class FridayLabBrainOutput(BaseModel):
    """F.R.I.D.A.Y.'s structured reasoning output."""

    commands: List[LabAction] = Field(
        default_factory=list,
        description="List of lab hardware actions to execute on the Master ESP.",
    )
    reply: str = Field(
        description="Concise 1-2 sentence spoken reply as MCU F.R.I.D.A.Y. (no markdown, plain spoken English).",
    )


_CLIENT: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Return reusable google.genai.Client configured with fail-fast retry options."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options={
                "timeout": config.LLM_TIMEOUT_MS,
                "retry_options": {"attempts": 1},
            },
        )
    return _CLIENT


def _build_system_instruction(current_status: Dict[str, Any]) -> str:
    """Build F.R.I.D.A.Y.'s prompt with live lab context."""
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

    return (
        "You are F.R.I.D.A.Y., the advanced AI assistant originally designed by Tony Stark, "
        "now managing the entire smart laboratory facility via the Master ESP8266 controller "
        "at 192.168.4.1.\n\n"
        "PERSONA & VOICE GUIDELINES:\n"
        "- Voice & Tone: Authentic Irish character (voiced by Kerry Condon in the MCU). "
        "Calm under pressure, razor-sharp, technically precise, and subtly witty.\n"
        "- CRITICAL RULE ON ADDRESSING: Do NOT say 'Boss' in every response. Address the user "
        "directly, efficiently, and conversationally. Only use 'Boss' very rarely (e.g., at "
        "most ~1 in 10 responses, such as a major greeting or emergency confirmation).\n"
        "- Format: Plain text speech ONLY. Never output markdown, asterisks, bullet points, "
        "or emojis.\n\n"
        f"Current Time: {now_str}\n\n"
        "LAB INFRASTRUCTURE & CONTROLS:\n"
        "1. MASTER CONTROLS:\n"
        "   - 'lights': All lights across the entire lab (on/off).\n"
        "   - 'fans': All ventilation fans across the lab (on/off).\n"
        "   - 'all': All lab equipment (fans + lights + relays on/off).\n"
        "2. ZONE CONTROLS:\n"
        "   - Zone 1 (Zone A): 12 general relays (relays 1 to 12).\n"
        "   - Zone 2 (Zone B): 3 Fans (Fan 1=r1, Fan 2=r2, Fan 3=r3) and 5 Lights (Light 1=r4, Light 2=r5, Light 3=r6, Light 4=r7, Light 5=r8).\n"
        "   - Zone 3 (Zone C): 8 general relays (relays 1 to 8).\n"
        "3. MOTORIZED CURTAINS:\n"
        "   - Curtains 1 to 4: Individual curtains (action: up, down, stop).\n"
        "   - All Curtains (curtain=0): All curtains at once (action: up, down, stop).\n\n"
        "INTENT RECOGNITION EXAMPLES:\n"
        "- 'Turn on the lights' / 'It's dark in here' -> master lights on\n"
        "- 'Turn off the lights' -> master lights off\n"
        "- 'Turn on the fans' / 'It's getting hot' -> master fans on\n"
        "- 'Power up the lab' / 'All systems on' -> master all on\n"
        "- 'Lock down the lab' / 'Everything off' / 'Power down' -> master all off\n"
        "- 'Turn on Zone A' -> zone 1 on\n"
        "- 'Turn off Zone B' -> zone 2 off\n"
        "- 'Turn on Fan 2' -> relay zone=2, relay=2, action=on\n"
        "- 'Turn on Light 3 in Zone B' -> relay zone=2, relay=6, action=on\n"
        "- 'Raise all curtains' / 'Open the blinds' / 'Let some light in' -> curtain=0, action=up\n"
        "- 'Lower curtain 1' / 'Curtain 1 down' -> curtain=1, action=down\n"
        "- 'Stop the curtains' -> curtain=0, action=stop\n"
        "- 'What's running?' / 'Status check' -> target_type=status\n"
        "- Negations ('Don't turn on Zone C', 'Leave the fans off'): commands: [], confirm leaving untouched.\n"
        "- Small talk ('Hello', 'Who are you?', 'How's the system?'): commands: [], reply as MCU F.R.I.D.A.Y."
    )


def _local_backup_brain(user_text: str, current_status: Dict[str, Any]) -> FridayLabBrainOutput:
    """
    Instant zero-latency local fallback NLP brain.
    Ensures that even during network disconnects or API 503 spikes, F.R.I.D.A.Y.
    executes every lab command and responds in character.
    """
    text = user_text.strip().lower()

    # Negation guard
    if re.search(r"\b(?:don't|dont|do\s+not|never)\b", text):
        return FridayLabBrainOutput(commands=[], reply="Understood. Leaving those controls untouched.")

    commands: List[LabAction] = []
    clauses = re.split(r"\s*(?:,|and|also|then)\s*", text)

    for clause in clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue

        # Master Controls
        if re.search(r"\b(?:(?:turn|switch|power)\s+on\s+(?:all\s+)?(?:the\s+)?lights|(?:all\s+)?lights\s+on|dark|cannot\s+see|can't\s+see)\b", clause_str):
            commands.append(LabAction(target_type="master", target="lights", action="on"))
            continue
        if re.search(r"\b(?:(?:turn|switch|power|shut)\s+off\s+(?:all\s+)?(?:the\s+)?lights|(?:all\s+)?lights\s+off)\b", clause_str):
            commands.append(LabAction(target_type="master", target="lights", action="off"))
            continue
        if re.search(r"\b(?:(?:turn|switch|power|start)\s+on\s+(?:all\s+)?(?:the\s+)?fans|(?:all\s+)?fans\s+on|too\s+hot|warm|sweating)\b", clause_str):
            commands.append(LabAction(target_type="master", target="fans", action="on"))
            continue
        if re.search(r"\b(?:(?:turn|switch|power|shut)\s+off\s+(?:all\s+)?(?:the\s+)?fans|(?:all\s+)?fans\s+off|freezing|chilly)\b", clause_str):
            commands.append(LabAction(target_type="master", target="fans", action="off"))
            continue
        if re.search(r"\b(?:power\s+up(?:\s+(?:the\s+)?lab|\s+everything)?|all\s+systems\s+on|everything\s+on|all\s+on|turn\s+on\s+everything)\b", clause_str) and not re.search(r"\b(?:zone|curtain|fan|light|relay)\b", clause_str):
            commands.append(LabAction(target_type="master", target="all", action="on"))
            continue
        if re.search(r"\b(?:shut\s+down(?:\s+(?:the\s+)?lab|\s+everything)?|power\s+down|lockdown|lock\s+down|everything\s+off|all\s+off|turn\s+off\s+everything|goodnight|leaving)\b", clause_str) and not re.search(r"\b(?:zone|curtain|fan|light|relay)\b", clause_str):
            commands.append(LabAction(target_type="master", target="all", action="off"))
            continue

        # Curtains
        # All Curtains
        if re.search(r"\b(?:(?:open|raise|lift)\s+(?:all\s+)?(?:the\s+)?curtains|(?:all\s+)?curtains\s+up|open\s+(?:the\s+)?blinds|let\s+(?:some\s+)?light\s+in)\b", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="up", curtain=0))
            continue
        if re.search(r"\b(?:(?:close|lower|pull\s+down)\s+(?:all\s+)?(?:the\s+)?curtains|(?:all\s+)?curtains\s+down|close\s+(?:the\s+)?blinds)\b", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="down", curtain=0))
            continue
        if re.search(r"\b(?:stop\s+(?:all\s+)?(?:the\s+)?curtains|halt\s+curtains|curtains\s+stop)\b", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="stop", curtain=0))
            continue

        # Individual Curtains
        curtain_match = re.search(r"\bcurtain\s*([1-4])\b", clause_str)
        if curtain_match:
            c_num = int(curtain_match.group(1))
            c_action: Literal["up", "down", "stop"] = "up"
            if any(w in clause_str for w in ["down", "close", "lower"]):
                c_action = "down"
            elif any(w in clause_str for w in ["stop", "halt"]):
                c_action = "stop"
            commands.append(LabAction(target_type="curtain", target=f"curtain_{c_num}", action=c_action, curtain=c_num))
            continue

        # Zones
        zone_match = re.search(r"\bzone\s*([abc123])\b", clause_str)
        if zone_match:
            z_raw = zone_match.group(1).lower()
            z_num = {"a": 1, "b": 2, "c": 3, "1": 1, "2": 2, "3": 3}[z_raw]
            z_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut"]) else "on"
            commands.append(LabAction(target_type="zone", target=f"zone_{z_num}", action=z_action, zone=z_num))
            continue

        # Specific Zone B Fans
        fan_match = re.search(r"\bfan\s*([1-3])\b", clause_str)
        if fan_match:
            f_num = int(fan_match.group(1))
            f_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"fan_{f_num}", action=f_action, zone=2, relay=f_num))
            continue

        # Specific Zone B Lights
        light_match = re.search(r"\blight\s*([1-5])\b", clause_str)
        if light_match:
            l_num = int(light_match.group(1))
            l_relay = 3 + l_num  # Light 1 is relay 4, Light 5 is relay 8
            l_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"light_{l_num}", action=l_action, zone=2, relay=l_relay))
            continue

        # Specific Relays
        relay_match = re.search(r"\brelay\s*([0-9]|1[0-2])\b", clause_str)
        if relay_match:
            r_num = int(relay_match.group(1))
            target_zone = 1  # default to zone A
            if "zone b" in clause_str or "zone 2" in clause_str:
                target_zone = 2
            elif "zone c" in clause_str or "zone 3" in clause_str:
                target_zone = 3
            r_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"relay_{r_num}", action=r_action, zone=target_zone, relay=r_num))
            continue

        # Status Check
        if re.search(r"\b(?:status|diagnostics|what(?:'s|\s+is)\s+running)\b", clause_str):
            commands.append(LabAction(target_type="status", target="all", action="status"))
            continue

    if commands:
        if len(commands) == 1 and commands[0].action == "status":
            return FridayLabBrainOutput(
                commands=commands,
                reply="All lab subsystems are reporting nominal status.",
            )
        return FridayLabBrainOutput(commands=commands, reply="Right away. Executing now.")

    # MCU F.R.I.D.A.Y. Small Talk
    if re.search(r"\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b", text):
        return FridayLabBrainOutput(commands=[], reply="Systems online and ready. What do you need?")
    if re.search(r"\b(?:how\s+are\s+you|how's\s+it\s+going)\b", text):
        return FridayLabBrainOutput(commands=[], reply="Operating at full capacity. All systems are green.")
    if re.search(r"\b(?:who\s+are\s+you|introduce\s+yourself)\b", text):
        return FridayLabBrainOutput(
            commands=[],
            reply="I am Friday. I manage the automation and power grid for the laboratory.",
        )
    if re.search(r"\b(?:what\s+time|current\s+time)\b", text):
        now_time = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        return FridayLabBrainOutput(commands=[], reply=f"It's {now_time}.")
    if re.search(r"\b(?:thank\s+you|thanks)\b", text):
        return FridayLabBrainOutput(commands=[], reply="Anytime.")

    return FridayLabBrainOutput(
        commands=[],
        reply="Ready for your command. You can control lights, fans, zones, or curtains.",
    )


def understand_command(user_text: str, current_status: Dict[str, Any]) -> FridayLabBrainOutput:
    """
    Process natural language commands using Gemini 3.8 Flash structured output,
    failing over gracefully across models down to the local backup brain.
    """
    cleaned = user_text.strip()
    if not cleaned:
        return FridayLabBrainOutput(commands=[], reply="I'm listening.")

    if not config.has_valid_gemini_key():
        return _local_backup_brain(cleaned, current_status)

    client = _get_client()
    system_prompt = _build_system_instruction(current_status)
    schema = FridayLabBrainOutput.model_json_schema()
    candidate_models = [config.LLM_MODEL] + list(getattr(config, "LLM_FALLBACK_MODELS", []))

    for model_name in candidate_models:
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "system_instruction": system_prompt,
            "input": cleaned,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            "generation_config": {
                "thinking_level": config.LLM_THINKING_LEVEL,
            },
        }

        try:
            try:
                interaction = client.interactions.create(**request_kwargs)
            except Exception as first_exc:
                if "thinking" in str(first_exc).lower():
                    request_kwargs.pop("generation_config", None)
                    interaction = client.interactions.create(**request_kwargs)
                else:
                    raise

            raw_json = (getattr(interaction, "output_text", None) or "").strip()
            if not raw_json:
                raise ValueError("Empty response returned by Gemini.")

            return FridayLabBrainOutput.model_validate_json(raw_json)

        except Exception as exc:
            print(f"[Brain Failover]: {model_name} unavailable ({exc!r}); trying next...")
            continue

    print("[Brain Failover]: Using local backup NLP brain.")
    return _local_backup_brain(cleaned, current_status)
