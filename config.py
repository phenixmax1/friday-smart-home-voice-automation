"""
config.py
---------
Central configuration, Gemini API settings, F.R.I.D.A.Y. cloud voice profiles,
NodeMCU Wi-Fi auto-discovery settings, and dynamic 4-channel relay mapping.

Note: Uses current Gemini 3.8 models (`gemini-3.8-flash` and
`gemini-3.8-flash-lite-tts`) via the official `google-genai` Interactions API
in place of legacy/deprecated `gemini-2.5-*` models.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Gemini API & Cloud Voice Settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE").strip()

# Current Gemini 3.8 / 3.5 models (replaces deprecated gemini-2.5-flash / 2.5-tts)
LLM_MODEL: str = "gemini-3.8-flash"
LLM_FALLBACK_MODELS: List[str] = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
LLM_THINKING_LEVEL: str = "low"  # Lowest supported latency level ("low", "medium", "high")
LLM_TIMEOUT_MS: int = 8000       # Fast 8s timeout (ms) before failing over to next model

TTS_MODEL: str = "gemini-3.8-flash-lite-tts"
TTS_FALLBACK_MODELS: List[str] = ["gemini-3.8-flash-tts"]

# Crisp female studio voice for Gemini TTS.
# Other female voices available: "Aoede", "Leda"
GEMINI_VOICE_NAME: str = "Kore"
GEMINI_VOICE_STYLE: str = (
    "crisp, warm, intelligent female AI assistant like FRIDAY; "
    "natural pacing, slightly witty, confident"
)

# High-speed Irish female neural voice (Marvel F.R.I.D.A.Y. accent)
EDGE_FRIDAY_VOICE: str = "en-IE-EmilyNeural"
EDGE_VOICE_RATE: str = "+8%"

# False -> Uses Gemini Cloud TTS first, with automatic fallback to Edge Neural TTS
#          on rate-limit / quota / network errors.
# True  -> Uses Edge Neural TTS directly for ultra-fast (~300ms) live demo speech.
PREFER_FAST_EDGE_VOICE: bool = False

# ---------------------------------------------------------------------------
# NodeMCU (ESP8266) Shared Wi-Fi Settings
# ---------------------------------------------------------------------------
NODEMCU_HOST: str = "smarthome.local"       # mDNS hostname advertised on Wi-Fi
NODEMCU_FALLBACK_IP: str = "192.168.1.50"   # Static/last-known IP from 16x2 I2C LCD
EXPECTED_DEVICE_ID: str = "nodemcu_smarthome"

REQUEST_TIMEOUT: float = 2.5                # HTTP request timeout (seconds)
DISCOVERY_TIMEOUT: float = 0.5              # Per-host timeout during /24 scan
DISCOVERY_MAX_WORKERS: int = 60             # Parallel threads for subnet discovery

# ---------------------------------------------------------------------------
# Speech Recognition (STT) Settings
# ---------------------------------------------------------------------------
STT_LANGUAGE: str = "en-IN"
AMBIENT_NOISE_DURATION: float = 0.6         # Calibrated ONCE at startup
STT_PAUSE_THRESHOLD: float = 0.65           # Snappy end-of-speech detection
STT_NON_SPEAKING_DURATION: float = 0.4
LISTEN_TIMEOUT: int = 6
PHRASE_TIME_LIMIT: int = 10

# ---------------------------------------------------------------------------
# Double-Clap Standby / Wake Settings (Iron Man style — no startup menu)
# ---------------------------------------------------------------------------
CLAP_WAKE_RESPONSE: str = "All systems online."
STANDBY_GOODBYE_RESPONSE: str = "Bye, it was great talking to you."

CLAP_SAMPLE_RATE: int = 44100
CLAP_CHUNK_SIZE: int = 1024                 # ~23 ms per audio frame at 44.1 kHz
CLAP_CALIBRATION_DURATION: float = 0.5      # Ambient noise floor calibration
CLAP_MIN_RMS: float = 1800.0                # Minimum RMS amplitude for a clap
CLAP_THRESHOLD_MULTIPLIER: float = 4.0      # Multiplier above ambient noise floor
CLAP_MAX_DURATION_SEC: float = 0.24         # Reject sustained speech (> 240 ms)
CLAP_COUNT_REQUIRED: int = 2                # Double clap to wake (set 1 for single clap)
CLAP_MIN_GAP_SEC: float = 0.12              # Echo rejection between claps
CLAP_MULTI_WINDOW_SEC: float = 0.85         # Max gap between clap 1 and clap 2
CLAP_COOLDOWN_SEC: float = 0.8              # Brief pause before re-arming standby

# ---------------------------------------------------------------------------
# Dynamic Relay-to-Appliance Mapping (Edit freely to reassign hardware relays)
# ---------------------------------------------------------------------------
DEVICE_MAP: Dict[str, Dict[str, Any]] = {
    "light": {
        "name": "Room Light",
        "aliases": ["light", "lights", "bulb", "lamp", "tube light", "led", "relay 1"],
    },
    "fan": {
        "name": "Ceiling Fan",
        "aliases": ["fan", "ceiling fan", "cooler", "exhaust", "blower", "relay 2"],
    },
    "relay3": {
        "name": "Power Socket",
        "aliases": ["socket", "plug", "charger", "ac", "air conditioner", "relay 3"],
    },
    "relay4": {
        "name": "Aux Appliance",
        "aliases": ["tv", "television", "heater", "geyser", "motor", "relay 4"],
    },
}


def has_valid_gemini_key() -> bool:
    """Return True if a non-placeholder GEMINI_API_KEY is configured."""
    return bool(GEMINI_API_KEY) and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"


def get_device_name(device_key: str) -> str:
    """Return the friendly display name for a relay endpoint key."""
    if device_key == "all":
        return "All Relays"
    entry = DEVICE_MAP.get(device_key)
    if isinstance(entry, dict) and entry.get("name"):
        return str(entry["name"])
    return device_key.title()


def get_all_device_keys() -> List[str]:
    """Return all configured NodeMCU relay endpoint keys."""
    return list(DEVICE_MAP.keys())
