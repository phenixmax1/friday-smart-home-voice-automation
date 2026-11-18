"""
nlp_processor.py
----------------
Fast Gemini 3.8 Flash NLP + F.R.I.D.A.Y. conversational brain using the
official `google-genai` Interactions API (`client.interactions.create`) with:
  - Fail-fast HTTP options (`attempts=1`, 8s timeout) so a temporary `503` never
    hangs for 30+ seconds in SDK backoff loops.
  - Automatic multi-model failover (`gemini-3.8-flash` -> `gemini-3.5-flash-lite`
    -> `gemini-3.1-flash-lite`).
  - Zero-latency local F.R.I.D.A.Y. NLP + small-talk backup if all cloud models
    are temporarily unreachable during a live demo.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional

from google import genai
from pydantic import BaseModel, Field

import config


class RelayCommand(BaseModel):
    """Single NodeMCU hardware relay instruction."""

    device: Literal["light", "fan", "relay3", "relay4", "all"]
    action: Literal["on", "off", "toggle", "status"]


class FridayBrainOutput(BaseModel):
    """Structured output returned by F.R.I.D.A.Y. in a single fast API call."""

    commands: list[RelayCommand] = Field(
        default_factory=list,
        description="Hardware commands to execute on NodeMCU (empty for pure chat or negations).",
    )
    reply: str = Field(
        description="Concise 1-2 sentence plain-text spoken reply as F.R.I.D.A.Y. (no markdown).",
    )


_CLIENT: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """
    Return a reusable `google.genai.Client` configured to fail fast (`attempts=1`)
    so 503 spikes switch instantly to the fallback model instead of stalling.
    """
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
    """Build F.R.I.D.A.Y.'s system prompt with live device map and relay states."""
    device_map_json = json.dumps(config.DEVICE_MAP, indent=2)
    relay_snapshot = {
        key: current_status.get(key, "UNKNOWN") for key in config.get_all_device_keys()
    }
    status_json = json.dumps(relay_snapshot, indent=2)
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

    return (
        "You are F.R.I.D.A.Y., an intelligent female AI assistant controlling a "
        "4-channel NodeMCU (ESP8266) smart home system. Your tone is crisp, warm, "
        "composed, and subtly witty—never cheesy or verbose.\n"
        "IMPORTANT ADDRESSING RULE: Do NOT say 'Boss' in every reply. Speak naturally "
        "without titles most of the time, and only use 'Boss' occasionally when it "
        "genuinely fits (such as a greeting, farewell, or key moment—at most ~1 in 5 replies).\n"
        f"Current local time: {now_str}\n\n"
        f"Configured Hardware Relay Mapping (`DEVICE_MAP`):\n{device_map_json}\n\n"
        f"Current Live Relay Status:\n{status_json}\n\n"
        "STRICT RULES:\n"
        "1. Greetings & Small Talk: For greetings ('Hello', 'Hey Friday', 'How are you?'), "
        "demos ('Introduce yourself to the external examiner'), or general questions "
        "without hardware intent, return `commands: []` and a natural 1-2 sentence reply.\n"
        "2. Direct & Multi-Device Commands: Extract every requested relay command "
        "(e.g., 'Turn on the fan and switch off the light' -> 2 commands).\n"
        "3. Implicit / Contextual Commands:\n"
        "   - 'It's too dark in here' / 'I can't see' -> light on\n"
        "   - 'It's really hot' / 'I am sweating' -> fan on\n"
        "   - 'I'm freezing' / 'It's too cold' -> fan off\n"
        "   - 'I'm leaving the room' / 'Goodnight Friday' / 'Shut it all down' -> all off\n"
        "   - 'Power everything up' -> all on\n"
        "   - 'What is the status?' / 'What is running?' -> status command or read directly from live status.\n"
        "4. Negation Handling: If the user says 'Don't turn on the fan' or 'Do not switch "
        "off the light', return `commands: []` and confirm you are leaving it untouched.\n"
        "5. Output Style: Keep `reply` strictly plain text (1-2 short sentences, NO markdown, "
        "NO bullet points, NO asterisks, NO emojis) so cloud speech synthesis stays fast."
    )


def _local_backup_brain(
    user_text: str,
    current_status: Dict[str, Any],
) -> FridayBrainOutput:
    """
    Zero-latency local fallback used ONLY when all Gemini cloud models return
    503 / network errors, so live demos never stall or go silent.
    """
    text = user_text.strip().lower()

    # Negation guard
    if re.search(r"\b(?:don't|dont|do\s+not|never)\b", text):
        return FridayBrainOutput(commands=[], reply="Got it, leaving it untouched.")

    # Quick contextual / direct hardware commands
    commands: List[RelayCommand] = []
    clauses = re.split(r"\s*(?:,|and|also|then)\s*", text)
    for clause in clauses:
        if re.search(r"\b(?:dark|can't see|cannot see|need light)\b", clause):
            commands.append(RelayCommand(device="light", action="on"))
            continue
        if re.search(r"\b(?:too bright)\b", clause):
            commands.append(RelayCommand(device="light", action="off"))
            continue
        if re.search(r"\b(?:hot|warm|sweating)\b", clause):
            commands.append(RelayCommand(device="fan", action="on"))
            continue
        if re.search(r"\b(?:cold|chilly|freezing)\b", clause):
            commands.append(RelayCommand(device="fan", action="off"))
            continue
        if re.search(r"\b(?:leaving|goodnight|good night|shut it all down|everything off|all off)\b", clause):
            commands.append(RelayCommand(device="all", action="off"))
            continue
        if re.search(r"\b(?:power everything up|everything on|all on)\b", clause):
            commands.append(RelayCommand(device="all", action="on"))
            continue
        if re.search(r"\b(?:status|what(?:'s|\s+is)\s+(?:on|running))\b", clause):
            commands.append(RelayCommand(device="all", action="status"))
            continue

        action: Optional[Literal["on", "off", "toggle", "status"]] = None
        if re.search(r"\b(?:toggle|flip)\b", clause):
            action = "toggle"
        elif re.search(r"\b(?:off|stop|shut|disable|kill)\b", clause):
            action = "off"
        elif re.search(r"\b(?:on|start|enable|activate)\b", clause):
            action = "on"

        if action:
            for dev_key, meta in config.DEVICE_MAP.items():
                aliases = [dev_key] + [a.lower() for a in meta.get("aliases", [])]
                if any(re.search(rf"\b{re.escape(alias)}\b", clause) for alias in aliases):
                    commands.append(RelayCommand(device=dev_key, action=action))  # type: ignore[arg-type]
                    break

    if commands:
        if len(commands) == 1 and commands[0].action == "status":
            parts = [
                f"{config.get_device_name(k)} is {current_status.get(k, 'OFF')}"
                for k in config.get_all_device_keys()
            ]
            return FridayBrainOutput(
                commands=commands,
                reply="Here's the current status: " + ", ".join(parts) + ".",
            )
        summary = ", ".join(
            f"{config.get_device_name(c.device)} {c.action}" for c in commands
        )
        return FridayBrainOutput(commands=commands, reply=f"Done. {summary}.")

    # Friendly F.R.I.D.A.Y. small talk fallback
    if re.search(r"\b(?:how\s+are\s+you|how(?:'s|\s+is)\s+it\s+going)\b", text):
        return FridayBrainOutput(
            commands=[],
            reply="Running smooth and ready. How are you holding up?",
        )
    if re.search(r"\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b", text):
        return FridayBrainOutput(
            commands=[],
            reply="Hey Boss, everything is ready when you are.",
        )
    if re.search(r"\b(?:introduce\s+yourself|who\s+are\s+you)\b", text):
        return FridayBrainOutput(
            commands=[],
            reply=(
                "I'm Friday, an AI voice assistant managing this four-channel "
                "NodeMCU smart home system over Wi-Fi."
            ),
        )
    if re.search(r"\b(?:what\s+time|current\s+time)\b", text):
        now_str = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        return FridayBrainOutput(commands=[], reply=f"It's {now_str}.")
    if re.search(r"\b(?:thank\s+you|thanks)\b", text):
        return FridayBrainOutput(commands=[], reply="Anytime.")

    return FridayBrainOutput(
        commands=[],
        reply="Cloud AI is briefly busy, so I'm on local backup. Give me a room command.",
    )


def understand_command(
    user_text: str,
    current_status: Dict[str, Any],
) -> FridayBrainOutput:
    """
    Understand natural language commands and small talk with automatic
    multi-model failover (`gemini-3.8-flash` -> `gemini-3.5-flash-lite` ->
    `gemini-3.1-flash-lite`) and instant local backup on 503 outages.
    """
    cleaned = user_text.strip()
    if not cleaned:
        return FridayBrainOutput(commands=[], reply="Didn't catch that.")

    if not config.has_valid_gemini_key():
        return FridayBrainOutput(
            commands=[],
            reply=(
                "My Gemini API key isn't configured yet. "
                "Set GEMINI_API_KEY in your environment or dot env file."
            ),
        )

    client = _get_client()
    system_prompt = _build_system_instruction(current_status)
    schema = FridayBrainOutput.model_json_schema()
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
            return FridayBrainOutput.model_validate_json(raw_json)

        except Exception as exc:
            print(f"[Brain Failover]: {model_name} unavailable ({exc!r}); trying next...")
            continue

    print("[Brain Failover]: All Gemini cloud models temporarily unavailable; using local backup.")
    return _local_backup_brain(cleaned, current_status)
