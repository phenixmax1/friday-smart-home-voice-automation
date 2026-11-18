"""
ai_brain.py
-----------
Intelligent conversational, small-talk, and smart-home reasoning engine.

- Online + `GEMINI_API_KEY`: Uses the Gemini Interactions API (`gemini-3.8-flash`)
  with structured JSON output, live local time awareness, and stateful
  multi-turn conversation (`previous_interaction_id`).
- Offline / No API Key: Uses local `nlp_processor` for hardware commands AND a
  sharp, natural, understated F.R.I.D.A.Y./J.A.R.V.I.S. small-talk engine so
  casual conversation works smoothly out of the box with zero latency.
"""

from __future__ import annotations

import datetime
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import config
import nlp_processor

try:
    from google import genai
except ImportError:  # pragma: no cover - optional until installed
    genai = None  # type: ignore[assignment]


_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": (
                "Zero or more hardware relay actions to execute on the NodeMCU. "
                "Leave empty if the user is making small talk, asking a general "
                "question, or negating a command (e.g. 'don't turn on the fan')."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "enum": ["light", "fan", "relay3", "relay4", "all"],
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "toggle", "status"],
                    },
                },
                "required": ["device", "action"],
            },
        },
        "reply": {
            "type": "string",
            "description": (
                "A concise, natural, composed 1-2 sentence spoken response in "
                "a calm JARVIS/FRIDAY style. No markdown, asterisks, or emojis."
            ),
        },
    },
    "required": ["actions", "reply"],
}


def _time_of_day_greeting() -> str:
    """Return 'Good morning', 'Good afternoon', or 'Good evening'."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _match_local_small_talk(text: str) -> Optional[str]:
    """
    Fast, zero-latency local small-talk handler in a calm, non-cringe
    F.R.I.D.A.Y. / J.A.R.V.I.S. voice when running without Gemini API.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    # Time & Date
    if re.search(
        r"\b(?:what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+time|current\s+time|tell\s+me\s+the\s+time|time\s+now)\b",
        cleaned,
    ):
        now_str = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        return f"It's {now_str}, sir."

    if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+date|today(?:'s)?\s+date)|what\s+day\s+is\s+it)\b", cleaned):
        today = datetime.datetime.now().strftime("%A, %B %d")
        return f"Today is {today}, sir."

    # Check-ins / Presence ("Friday?", "Are you there?", "You up?")
    if re.fullmatch(r"(?:hey\s+)?(?:friday|jarvis)[?.!]?", cleaned) or re.search(
        r"\b(?:are\s+you\s+there|you\s+there|you\s+up|can\s+you\s+hear\s+me)\b", cleaned
    ):
        return random.choice(
            [
                "Right here, sir.",
                "Listening, sir. What do you need?",
                "Loud and clear, sir.",
            ]
        )

    # Greetings
    if re.search(
        r"\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|whats\s+up|what's\s+up)\b",
        cleaned,
    ):
        greeting = _time_of_day_greeting()
        return random.choice(
            [
                f"{greeting}, sir. Everything's ready when you are.",
                f"Hey sir. How's the day going?",
                f"{greeting}, sir. What are we tackling today?",
            ]
        )

    # How are you / status of Friday
    if re.search(r"\b(?:how\s+are\s+you|how(?:'s|\s+is)\s+it\s+going|how\s+do\s+you\s+feel)\b", cleaned):
        return random.choice(
            [
                "Running smooth and responsive, sir. How about you?",
                "All clear on my end, sir. What's on your mind?",
                "Sharp as ever, sir. Need anything adjusted in the room?",
            ]
        )

    # Identity
    if re.search(r"\b(?:who\s+are\s+you|what(?:'s|\s+is)\s+your\s+name|introduce\s+yourself)\b", cleaned):
        return (
            "I'm Friday, your smart home assistant. I handle the room controls "
            "and keep things running smoothly."
        )

    # Capabilities
    if re.search(r"\b(?:what\s+can\s+you\s+do|help\s+me|your\s+features)\b", cleaned):
        return (
            "I can control your lights, fan, and relays, check what's running, "
            "or just talk while you work, sir."
        )

    # Gratitude / Compliments
    if re.search(r"\b(?:thank\s+you|thanks|appreciate\s+it|good\s+job|nice\s+work|well\s+done|awesome|cool)\b", cleaned):
        return random.choice(
            [
                "Anytime, sir.",
                "Happy to help.",
                "Always, sir.",
                "Glad that worked out.",
            ]
        )

    # User sharing mood / casual chat
    if re.search(
        r"\b(?:i(?:'m|\s+am)\s+(?:doing\s+)?(?:good|fine|great|okay|well|alright))\b",
        cleaned,
    ):
        return random.choice(
            [
                "Good to hear, sir. Let me know if you need anything.",
                "Glad to hear it. Just say the word if you want the room adjusted.",
            ]
        )

    if re.search(r"\b(?:i(?:'m|\s+am)\s+(?:tired|exhausted|sleepy))\b", cleaned):
        return "Sounds like a long day, sir. Say goodnight whenever you want me to shut everything down."

    if re.search(r"\b(?:i(?:'m|\s+am)\s+bored|tell\s+me\s+a\s+joke|make\s+me\s+laugh)\b", cleaned):
        return random.choice(
            [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "I'd tell you a UDP joke, sir, but you might not get it.",
                "Smart homes are great until you realize your fan listens better than most people.",
            ]
        )

    return None


def _build_system_instruction() -> str:
    """Build dynamic system prompt with live device map and current local time."""
    device_lines = []
    for key in config.get_all_device_keys():
        name = config.get_device_name(key)
        synonyms = ", ".join(config.get_device_synonyms(key))
        device_lines.append(f'  - "{key}": {name} (aliases: {synonyms})')

    devices_block = "\n".join(device_lines)
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

    return (
        f"{config.ASSISTANT_PERSONA}\n\n"
        f"Current local date and time: {now_str}\n\n"
        "You control a 4-channel NodeMCU smart home controller with these endpoints:\n"
        f"{devices_block}\n"
        '  - "all": targets all 4 relays simultaneously.\n\n'
        "Rules for `actions`:\n"
        "1. Extract direct commands, multi-command sentences, and implicit contextual "
        "needs (e.g., 'it is too dark' -> light on; 'it is hot' -> fan on; "
        "'I am going to watch a movie' -> light off + relay4 on; 'goodnight' -> all off).\n"
        "2. Strictly respect negations ('don't turn on the fan' -> actions: []).\n"
        "3. If the user is making small talk or asking a general question, leave "
        "`actions` empty and respond naturally and concisely in `reply`.\n"
        "4. Return valid JSON matching the schema."
    )


class AssistantBrain:
    """
    Hybrid AI Brain:
      - Uses Gemini 3.8 Flash (Interactions API) with stateful `previous_interaction_id`
        when `GEMINI_API_KEY` is configured and internet is reachable.
      - Uses local hybrid NLP + natural small-talk engine when offline or without
        a Gemini API key (with zero network latency).
    """

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._previous_interaction_id: Optional[str] = None

        if genai is not None and config.GEMINI_API_KEY:
            try:
                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as exc:
                print(f"[AI Warning]: Could not initialize Gemini client ({exc}).")
                self._client = None

    @property
    def gemini_enabled(self) -> bool:
        """Return True when the Gemini 3.8 Flash client is initialized."""
        return self._client is not None

    def reset_session(self) -> None:
        """Clear multi-turn conversation state when returning to standby."""
        self._previous_interaction_id = None

    def process(self, user_text: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Interpret `user_text` into `(actions, suggested_reply)`.

        Fast path: if Gemini is NOT enabled, runs local NLP + local small talk
        in < 2 ms. If Gemini IS enabled, uses Gemini 3.8 Flash for advanced
        reasoning & natural conversation, falling back locally if unreachable.
        """
        if self._client is not None:
            gemini_result = self._ask_gemini(user_text)
            if gemini_result is not None:
                return gemini_result

        # Fast local path (offline or no Gemini API key)
        actions = nlp_processor.parse_command(user_text)
        if actions:
            return actions, None

        small_talk = _match_local_small_talk(user_text)
        if small_talk:
            return [], small_talk

        return [], None

    def _ask_gemini(self, user_text: str) -> Optional[Tuple[List[Dict[str, str]], str]]:
        """Call Gemini 3.8 Flash via the Interactions API with structured output."""
        if self._client is None:
            return None

        kwargs: Dict[str, Any] = {
            "model": config.GEMINI_CHAT_MODEL,
            "system_instruction": _build_system_instruction(),
            "input": user_text,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": _RESPONSE_SCHEMA,
            },
            "generation_config": {
                "thinking_level": "minimal",
            },
        }
        if self._previous_interaction_id:
            kwargs["previous_interaction_id"] = self._previous_interaction_id

        try:
            interaction = self._client.interactions.create(**kwargs)
            self._previous_interaction_id = getattr(interaction, "id", None)
            raw_text = (getattr(interaction, "output_text", None) or "").strip()
            if not raw_text:
                return None

            parsed = json.loads(raw_text)
            raw_actions = parsed.get("actions", [])
            valid_devices = set(config.get_all_device_keys()) | {"all"}
            valid_actions = {"on", "off", "toggle", "status"}

            clean_actions: List[Dict[str, str]] = []
            for item in raw_actions:
                if not isinstance(item, dict):
                    continue
                dev = str(item.get("device", "")).strip().lower()
                act = str(item.get("action", "")).strip().lower()
                if dev in valid_devices and act in valid_actions:
                    clean_actions.append({"device": dev, "action": act})

            reply = str(parsed.get("reply", "")).strip()
            return clean_actions, reply
        except Exception as exc:
            print(f"[AI Fallback]: Gemini unreachable ({exc}); using local engine.")
            return None


def format_local_reply(
    actions: List[Dict[str, str]],
    outcomes: List[Tuple[bool, str, str, str]],
) -> str:
    """Generate a crisp, understated J.A.R.V.I.S./F.R.I.D.A.Y. confirmation."""
    if not outcomes:
        return "Nothing to change right now, sir."

    success_phrases: List[str] = []
    error_phrases: List[str] = []

    for ok, device, action, detail in outcomes:
        if not ok:
            error_phrases.append(detail)
            continue

        if action == "status":
            success_phrases.append(detail)
        elif device == "all":
            if action == "off":
                success_phrases.append("Everything is powered down")
            elif action == "on":
                success_phrases.append("All relays are powered up")
            else:
                success_phrases.append("All relays toggled")
        else:
            name = config.get_device_name(device)
            if action == "on":
                success_phrases.append(f"{name} is on")
            elif action == "off":
                success_phrases.append(f"{name} is off")
            else:
                success_phrases.append(f"{name} toggled")

    parts: List[str] = []
    if success_phrases:
        joined = (
            success_phrases[0]
            if len(success_phrases) == 1
            else ", ".join(success_phrases[:-1]) + f" and {success_phrases[-1]}"
        )
        if not joined.endswith("."):
            joined += ", sir."
        parts.append(joined)

    if error_phrases:
        parts.append(error_phrases[0])

    return " ".join(parts)
