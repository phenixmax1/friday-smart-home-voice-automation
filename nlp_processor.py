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


def is_hindi_or_hinglish(text: str) -> bool:
    """Return True if text contains Devanagari characters or common Hindi/Hinglish keywords."""
    if re.search(r"[\u0900-\u097f]", text):
        return True
    hinglish_tokens = {
        "karo", "kardo", "kar", "kijiye", "chalu", "band", "khol", "kholo", "roko",
        "pankha", "pankhe", "parda", "parde", "upar", "niche", "saare", "saari", "sab",
        "kaise", "kaisi", "kaisa", "theek", "hai", "hain", "ho", "hoon", "kya", "haal",
        "roshni", "andhera", "garmi", "dhup", "dhoop", "namaste", "namaskar", "alvida",
        "sojao", "batao", "suno", "haan", "nahi", "jaldi", "chahiye", "karoge", "shuru",
        "chal", "kare", "karna", "bhi", "hawa", "jala", "jalao", "bujha", "bujhao",
        "sabko", "kripya", "raha", "rahi", "gaya", "kaun", "aap", "tum", "tera", "mera",
        "parichay", "shukriya", "dhanyawad", "madad"
    }
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return bool(words & hinglish_tokens)


def _local_backup_brain(user_text: str, current_status: Dict[str, Any]) -> FridayLabBrainOutput:
    """
    Instant zero-latency local fallback NLP brain.
    Fully supports English, Hindi (Devanagari), and Hinglish (Romanized Hindi)
    for all master controls, zones, relays, curtains, and conversational small talk.
    """
    text = user_text.strip().lower()
    is_hindi = is_hindi_or_hinglish(text)

    # Negation guard (English, Hindi & Hinglish)
    if re.search(r"\b(?:don't|dont|do\s+not|never|mat\s+karo|nahi\s+karna|mat\s+chalu|mat\s+band)\b|[\u0900-\u097f]*(?:मत\s*करो|नहीं)", text):
        reply = "Theek hai, in controls ko waise hi chhod rahi hoon." if is_hindi else "Understood. Leaving those controls untouched."
        return FridayLabBrainOutput(commands=[], reply=reply)

    commands: List[LabAction] = []
    clauses = re.split(r"\s*(?:,|\band\b|\balso\b|\bthen\b|\baur\b|\btatha\b|\bya\b)\s*", text)

    for clause in clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue

        # Master Lights ON
        if re.search(r"\b(?:(?:turn|switch|power)\s+on\s+(?:all\s+)?(?:the\s+)?lights|(?:all\s+)?lights\s+on|dark|cannot\s+see|can't\s+see)\b", clause_str) or \
           re.search(r"\b(?:light\s*(?:on|chalu|jala)|lights\s*(?:on|chalu|jala)|roshni\s*(?:karo|chalu)|andhera|bohot\s*andhera)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:लाइट.*(?:चालू|ऑन|जला)|रोशनी|अंधेरा)", clause_str):
            commands.append(LabAction(target_type="master", target="lights", action="on"))
            continue

        # Master Lights OFF
        if re.search(r"\b(?:(?:turn|switch|power|shut)\s+off\s+(?:all\s+)?(?:the\s+)?lights|(?:all\s+)?lights\s+off)\b", clause_str) or \
           re.search(r"\b(?:light\s*(?:off|band|bujha)|lights\s*(?:off|band|bujha)|roshni\s*band)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:लाइट.*(?:बंद|ऑफ|बुझा))", clause_str):
            commands.append(LabAction(target_type="master", target="lights", action="off"))
            continue

        # Master Fans ON
        if re.search(r"\b(?:(?:turn|switch|power|start)\s+on\s+(?:all\s+)?(?:the\s+)?fans|(?:all\s+)?fans\s+on|too\s+hot|warm|sweating)\b", clause_str) or \
           re.search(r"\b(?:fan\s*(?:on|chalu)|fans\s*(?:on|chalu)|pankh[ae]\s*(?:on|chalu|chala)|garmi|bohot\s*garmi|hawa\s*aane\s*do)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:पंख[ाे].*(?:चालू|ऑन|चला)|गर्मी)", clause_str):
            commands.append(LabAction(target_type="master", target="fans", action="on"))
            continue

        # Master Fans OFF
        if re.search(r"\b(?:(?:turn|switch|power|shut)\s+off\s+(?:all\s+)?(?:the\s+)?fans|(?:all\s+)?fans\s+off|freezing|chilly)\b", clause_str) or \
           re.search(r"\b(?:fan\s*(?:off|band)|fans\s*(?:off|band)|pankh[ae]\s*(?:off|band)|thand)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:पंख[ाे].*(?:बंद|ऑफ)|ठंड)", clause_str):
            commands.append(LabAction(target_type="master", target="fans", action="off"))
            continue

        # Master ALL ON
        if (re.search(r"\b(?:power\s+up(?:\s+(?:the\s+)?lab|\s+everything)?|all\s+systems\s+on|everything\s+on|all\s+on|turn\s+on\s+everything)\b", clause_str) or \
            re.search(r"\b(?:sab\s*(?:kuch\s*)?(?:on|chalu)|pura\s*lab\s*(?:on|chalu)|poora\s*lab\s*(?:on|chalu)|saare\s*systems\s*on)\b", clause_str) or \
            re.search(r"[\u0900-\u097f]*(?:सब.*(?:चालू|ऑन)|पूरा.*(?:चालू|ऑन))", clause_str)) and not re.search(r"\b(?:zone|curtain|fan|light|relay|parda|parde|pankha|pankhe)\b|[\u0900-\u097f]*(?:ज़ोन|कर्टेन|पर्दा|पंखा)", clause_str):
            commands.append(LabAction(target_type="master", target="all", action="on"))
            continue

        # Master ALL OFF
        if (re.search(r"\b(?:shut\s+down(?:\s+(?:the\s+)?lab|\s+everything)?|power\s+down|lockdown|lock\s+down|everything\s+off|all\s+off|turn\s+off\s+everything|goodnight|leaving)\b", clause_str) or \
            re.search(r"\b(?:sab\s*(?:kuch\s*)?band|pura\s*lab\s*band|poora\s*lab\s*band|saare\s*systems\s*band|lockdown\s*karo)\b", clause_str) or \
            re.search(r"[\u0900-\u097f]*(?:सब.*(?:बंद|ऑफ)|पूरा.*(?:बंद|ऑफ)|लॉकडाउन)", clause_str)) and not re.search(r"\b(?:zone|curtain|fan|light|relay|parda|parde|pankha|pankhe)\b|[\u0900-\u097f]*(?:ज़ोन|कर्टेन|पर्दा|पंखा)", clause_str):
            commands.append(LabAction(target_type="master", target="all", action="off"))
            continue

        # Curtains ALL UP
        if re.search(r"\b(?:(?:open|raise|lift)\s+(?:all\s+)?(?:the\s+)?curtains|(?:all\s+)?curtains\s+up|open\s+(?:the\s+)?blinds|let\s+(?:some\s+)?light\s+in)\b", clause_str) or \
           re.search(r"\b(?:pard[ae]\s*(?:upar|khol)|curtains?\s*(?:upar|khol)|saare\s*parde\s*(?:upar|khol)|dhoop\s*aane\s*do)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:पर्द[ाे].*(?:ऊपर|खोल)|कर्टेन.*ऊपर|धूप)", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="up", curtain=0))
            continue

        # Curtains ALL DOWN
        if re.search(r"\b(?:(?:close|lower|pull\s+down)\s+(?:all\s+)?(?:the\s+)?curtains|(?:all\s+)?curtains\s+down|close\s+(?:the\s+)?blinds)\b", clause_str) or \
           re.search(r"\b(?:pard[ae]\s*(?:niche|band|gira)|curtains?\s*(?:niche|band|gira)|saare\s*parde\s*(?:niche|band))\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:पर्द[ाे].*(?:नीचे|बंद)|कर्टेन.*नीचे)", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="down", curtain=0))
            continue

        # Curtains ALL STOP
        if re.search(r"\b(?:stop\s+(?:all\s+)?(?:the\s+)?curtains|halt\s+curtains|curtains\s+stop)\b", clause_str) or \
           re.search(r"\b(?:curtains?\s*roko|pard[ae]\s*roko)\b", clause_str) or \
           re.search(r"[\u0900-\u097f]*(?:पर्द[ाे].*रोक|कर्टेन.*रोक)", clause_str):
            commands.append(LabAction(target_type="curtain", target="all_curtains", action="stop", curtain=0))
            continue

        # Individual Curtains (1..4)
        curtain_match = re.search(r"\b(?:curtain|parda|कर्टेन|पर्दा)\s*([1-4])\b", clause_str)
        if curtain_match:
            c_num = int(curtain_match.group(1))
            c_action: Literal["up", "down", "stop"] = "up"
            if any(w in clause_str for w in ["down", "close", "lower", "niche", "band", "gira", "नीचे"]):
                c_action = "down"
            elif any(w in clause_str for w in ["stop", "halt", "roko", "रोक"]):
                c_action = "stop"
            commands.append(LabAction(target_type="curtain", target=f"curtain_{c_num}", action=c_action, curtain=c_num))
            continue

        # Zones (A, B, C / 1, 2, 3)
        zone_match = re.search(r"\b(?:zone|ज़ोन)\s*([abc123एबीसी])\b", clause_str)
        if zone_match:
            z_raw = zone_match.group(1).lower()
            z_map = {"a": 1, "b": 2, "c": 3, "1": 1, "2": 2, "3": 3, "ए": 1, "बी": 2, "सी": 3}
            z_num = z_map.get(z_raw, 1)
            z_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut", "band", "बंद"]) else "on"
            commands.append(LabAction(target_type="zone", target=f"zone_{z_num}", action=z_action, zone=z_num))
            continue

        # Specific Zone B Fans (1..3)
        fan_match = re.search(r"\b(?:fan|pankha|फैन|पंखा)\s*([1-3])\b", clause_str)
        if fan_match:
            f_num = int(fan_match.group(1))
            f_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut", "band", "बंद"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"fan_{f_num}", action=f_action, zone=2, relay=f_num))
            continue

        # Specific Zone B Lights (1..5)
        light_match = re.search(r"\b(?:light|roshni|लाइट)\s*([1-5])\b", clause_str)
        if light_match:
            l_num = int(light_match.group(1))
            l_relay = 3 + l_num  # Light 1 is relay 4, Light 5 is relay 8
            l_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut", "band", "बंद"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"light_{l_num}", action=l_action, zone=2, relay=l_relay))
            continue

        # Specific Relays
        relay_match = re.search(r"\b(?:relay|रिले)\s*([0-9]|1[0-2])\b", clause_str)
        if relay_match:
            r_num = int(relay_match.group(1))
            target_zone = 1
            if "zone b" in clause_str or "zone 2" in clause_str or "बी" in clause_str:
                target_zone = 2
            elif "zone c" in clause_str or "zone 3" in clause_str or "सी" in clause_str:
                target_zone = 3
            r_action = "off" if any(w in clause_str for w in ["off", "stop", "disable", "kill", "shut", "band", "बंद"]) else "on"
            commands.append(LabAction(target_type="relay", target=f"relay_{r_num}", action=r_action, zone=target_zone, relay=r_num))
            continue

        # Status Check
        if re.search(r"\b(?:status|diagnostics|what(?:'s|\s+is)\s+running|status\s*batao|kya\s*chal\s*raha\s*hai|kya\s*haal|sab\s*theek\s*hai)\b|[\u0900-\u097f]*(?:स्टेटस|हाल)", clause_str):
            commands.append(LabAction(target_type="status", target="all", action="status"))
            continue

    if commands:
        if len(commands) == 1:
            cmd = commands[0]
            if cmd.action == "status":
                reply = "Lab ke sabhi subsystems normal kaam kar rahe hain." if is_hindi else "All lab subsystems are reporting nominal status."
            elif cmd.target_type == "master":
                if cmd.target == "lights":
                    if is_hindi:
                        reply = "Lab ki saari lights on kar di hain." if cmd.action == "on" else "Lab ki saari lights band kar di hain."
                    else:
                        reply = "All lab lights are on." if cmd.action == "on" else "All lab lights are off."
                elif cmd.target == "fans":
                    if is_hindi:
                        reply = "Lab ke fans on kar diye hain." if cmd.action == "on" else "Lab ke fans band kar diye hain."
                    else:
                        reply = "All lab fans are on." if cmd.action == "on" else "All lab fans are off."
                elif cmd.target == "all":
                    if is_hindi:
                        reply = "Poora lab system power up kar diya hai." if cmd.action == "on" else "Poora lab system power down kar diya hai."
                    else:
                        reply = "Powering up all lab systems." if cmd.action == "on" else "All lab systems powered down."
                else:
                    reply = f"Master {cmd.target} {cmd.action}."
            elif cmd.target_type == "curtain":
                if cmd.curtain == 0 or "all" in cmd.target:
                    if is_hindi:
                        reply = "Saare parde upar kiye ja rahe hain." if cmd.action == "up" else "Saare parde niche kiye ja rahe hain." if cmd.action == "down" else "Parde rok diye gaye hain."
                    else:
                        reply = "Raising all curtains." if cmd.action == "up" else "Lowering all curtains." if cmd.action == "down" else "Curtains stopped."
                else:
                    if is_hindi:
                        act_hi = "upar ja raha hai" if cmd.action == "up" else "niche aa raha hai" if cmd.action == "down" else "rok diya gaya hai"
                        reply = f"Curtain {cmd.curtain} {act_hi}."
                    else:
                        act_desc = "going up" if cmd.action == "up" else "coming down" if cmd.action == "down" else "stopped"
                        reply = f"Curtain {cmd.curtain} is {act_desc}."
            elif cmd.target_type == "zone":
                z_letter = chr(64 + (cmd.zone or 1))
                if is_hindi:
                    reply = f"Zone {z_letter} {'on' if cmd.action == 'on' else 'band'} kar diya hai."
                else:
                    reply = f"Zone {z_letter} is {'online' if cmd.action == 'on' else 'offline'}."
            elif cmd.target_type == "relay":
                if "fan" in cmd.target:
                    f_num = cmd.relay or 1
                    reply = f"Fan {f_num} {'on' if cmd.action == 'on' else 'band'} kar diya hai." if is_hindi else f"Fan {f_num} is {'on' if cmd.action == 'on' else 'off'}."
                elif "light" in cmd.target:
                    l_num = (cmd.relay or 4) - 3
                    reply = f"Light {l_num} {'on' if cmd.action == 'on' else 'band'} kar di hai." if is_hindi else f"Light {l_num} is {'on' if cmd.action == 'on' else 'off'}."
                else:
                    reply = f"Relay {cmd.relay or 1} {'on' if cmd.action == 'on' else 'band'} kar diya hai." if is_hindi else f"Relay {cmd.relay or 1} is {'on' if cmd.action == 'on' else 'off'}."
            else:
                reply = "Theek hai." if is_hindi else "Right away."
        else:
            reply = "Theek hai. Diye gaye sabhi lab commands execute kar diye hain." if is_hindi else "Right away. Executing requested lab commands."
        return FridayLabBrainOutput(commands=commands, reply=reply)

    # Conversational Small Talk (English & Hindi/Hinglish)
    if re.search(r"\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|namaste|namaskar)\b|[\u0900-\u097f]*(?:नमस्ते|नमस्कार)", text):
        reply = "Namaste! Main online hoon aur ready hoon, bataiye kya karna hai?" if is_hindi else "Systems online and ready. What do you need?"
        return FridayLabBrainOutput(commands=[], reply=reply)
    if re.search(r"\b(?:how\s+are\s+you|how's\s+it\s+going|kaise\s+ho|kaisi\s+ho|kya\s+haal)\b|[\u0900-\u097f]*(?:कैसी\s*हो|कैसे\s*हो)", text):
        reply = "Main bilkul theek hoon. Sabhi lab systems green aur operational hain." if is_hindi else "Operating at full capacity. All systems are green."
        return FridayLabBrainOutput(commands=[], reply=reply)
    if re.search(r"\b(?:who\s+are\s+you|introduce\s+yourself|tum\s+kaun\s+ho|aap\s+kaun\s+ho)\b|[\u0900-\u097f]*(?:तुम\s*कौन\s*हो|आप\s*कौन)", text):
        reply = "Main Friday hoon, aapki AI lab assistant. Main pure laboratory automation ko control karti hoon." if is_hindi else "I am Friday. I manage the automation and power grid for the laboratory."
        return FridayLabBrainOutput(commands=[], reply=reply)
    if re.search(r"\b(?:what\s+time|current\s+time|samay|kya\s+baje)\b|[\u0900-\u097f]*(?:समय|टाइम)", text):
        now_time = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        reply = f"Abhi samay {now_time} ho raha hai." if is_hindi else f"It's {now_time}."
        return FridayLabBrainOutput(commands=[], reply=reply)
    if re.search(r"\b(?:thank\s+you|thanks|shukriya|dhanyawad)\b|[\u0900-\u097f]*(?:धन्यवाद|शुक्रिया)", text):
        reply = "Aapka swagat hai." if is_hindi else "Anytime."
        return FridayLabBrainOutput(commands=[], reply=reply)

    fallback_reply = (
        "Main aapke command ke liye ready hoon. Aap lights, fans, zones, ya curtains control kar sakte hain."
        if is_hindi
        else "Ready for your command. You can control lights, fans, zones, or curtains."
    )
    return FridayLabBrainOutput(commands=[], reply=fallback_reply)


def understand_command(user_text: str, current_status: Dict[str, Any]) -> FridayLabBrainOutput:
    """
    Process natural language commands with sub-millisecond instant hardware
    matching for standard controls, and Gemini 3.8 Flash for open-ended queries.
    """
    cleaned = user_text.strip()
    if not cleaned:
        return FridayLabBrainOutput(commands=[], reply="I'm listening.")

    # 1. FAST ZERO-LATENCY MATCHER: Sub-millisecond hardware resolution
    fast_result = _local_backup_brain(cleaned, current_status)
    if fast_result.commands or (
        fast_result.reply != "Ready for your command. You can control lights, fans, zones, or curtains."
    ):
        return fast_result

    # 2. GENERATIVE CLOUD BRAIN: Used only for non-hardware open-ended queries
    if not config.has_valid_gemini_key():
        return fast_result

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
    return fast_result
