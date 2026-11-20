"""
config.py
---------
Master configuration for F.R.I.D.A.Y. Lab Voice Automation System:
- Controls the Master ESP8266 controller running at `http://192.168.4.1`
  (Lab Automation: Master Controls, Zone A [12 relays], Zone B [3 fans, 5 lights],
   Zone C [8 relays], and 4 motorized curtains).
- Marvel's F.R.I.D.A.Y. Persona & Voice:
  * Irish female neural voice (`en-IE-EmilyNeural` / Gemini TTS `Kore`/`Aoede`)
  * Poised, intelligent, concise, subtly witty tone (never spamming 'Boss')
- Zero on-device robotic TTS (`pyttsx3` is completely prohibited).
- Instant double-clap wake from silent standby (Iron Man style, zero startup menus).
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

# Gemini Models
LLM_MODEL: str = "gemini-3.8-flash"
LLM_FALLBACK_MODELS: List[str] = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
LLM_THINKING_LEVEL: str = "low"  # Fast latency level
LLM_TIMEOUT_MS: int = 4000       # Fast 4s timeout before auto-failover

TTS_MODEL: str = "gemini-3.8-flash-lite-tts"
TTS_FALLBACK_MODELS: List[str] = [
    "gemini-3.8-flash-tts",
    "gemini-2.5-flash-preview-tts",
]

# Crisp female studio voice for Gemini TTS
GEMINI_VOICE_NAME: str = "Kore"  # "Aoede" and "Leda" also available
GEMINI_VOICE_STYLE: str = (
    "crisp, warm, intelligent female AI assistant like Marvel's FRIDAY; "
    "natural pacing, composed, slightly witty, confident"
)

# Authentic Irish female neural voice (matches Kerry Condon's F.R.I.D.A.Y.)
EDGE_FRIDAY_VOICE: str = "en-IE-EmilyNeural"
EDGE_VOICE_RATE: str = "+8%"

# Default to True for instant sub-300ms speech with authentic MCU Irish female voice.
# Zero 404 errors, zero rate limits, and instant audio playback.
PREFER_FAST_EDGE_VOICE: bool = True

# ---------------------------------------------------------------------------
# Master ESP (ESP8266) Lab Automation Network Settings
# ---------------------------------------------------------------------------
# Default IP of Master ESP (Lab Access Point / Static IP)
NODEMCU_HOST: str = "192.168.4.1"
NODEMCU_FALLBACK_IP: str = "192.168.4.1"
NODEMCU_PORT: int = 80
EXPECTED_TITLE: str = "Lab Automation"

REQUEST_TIMEOUT: float = 1.8       # Fast 1.8s timeout for HTTP requests to ESP
DISCOVERY_TIMEOUT: float = 0.5     # Subnet scan per-host timeout
DISCOVERY_MAX_WORKERS: int = 60    # Subnet discovery threads

# ---------------------------------------------------------------------------
# Speech Recognition (STT) Settings
# ---------------------------------------------------------------------------
STT_LANGUAGE: str = "en-IN"
AMBIENT_NOISE_DURATION: float = 0.6  # Calibrated once at startup
STT_PAUSE_THRESHOLD: float = 0.45    # Snappy end-of-speech detection (0.45s)
STT_NON_SPEAKING_DURATION: float = 0.3 # 0.3s non-speaking window
LISTEN_TIMEOUT: int = 6
PHRASE_TIME_LIMIT: int = 10

# ---------------------------------------------------------------------------
# Double-Clap Standby / Wake Settings (Iron Man hands-free activation)
# ---------------------------------------------------------------------------
CLAP_WAKE_RESPONSE: str = "All lab systems online. What do you need?"
STANDBY_GOODBYE_RESPONSE: str = "Standing by. Call me when you need me."

CLAP_SAMPLE_RATE: int = 44100
CLAP_CHUNK_SIZE: int = 1024
CLAP_CALIBRATION_DURATION: float = 0.5
CLAP_MIN_RMS: float = 1800.0
CLAP_THRESHOLD_MULTIPLIER: float = 4.0
CLAP_MAX_DURATION_SEC: float = 0.24
CLAP_COUNT_REQUIRED: int = 2
CLAP_MIN_GAP_SEC: float = 0.12
CLAP_MULTI_WINDOW_SEC: float = 0.85
CLAP_COOLDOWN_SEC: float = 0.8

# ---------------------------------------------------------------------------
# Lab Hardware Layout & Endpoints (From Master ESP `index.html`)
# ---------------------------------------------------------------------------
# 1. Master Controls:
#    GET /master?type={lights|fans|all}&s={1|0}
# 2. Zone Controls:
#    GET /zone?z={1|2|3}&s={1|0}
# 3. Individual Relays:
#    GET /relay?z={1|2|3}&r={1..12}
# 4. Curtains:
#    GET /curtain?c={1..4}&a={1|2|0}  (1=Up, 2=Down, 0=Stop)
#    GET /allcurtains?a={1|2|0}

LAB_ZONES: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Zone A",
        "description": "General Lab Relays 1 to 12",
        "relays": {i: f"Relay {i}" for i in range(1, 13)},
    },
    2: {
        "name": "Zone B",
        "description": "Lab Fans 1-3 & Lights 1-5",
        "relays": {
            1: "Fan 1",
            2: "Fan 2",
            3: "Fan 3",
            4: "Light 1",
            5: "Light 2",
            6: "Light 3",
            7: "Light 4",
            8: "Light 5",
        },
    },
    3: {
        "name": "Zone C",
        "description": "Lab Relays 1 to 8",
        "relays": {i: f"Relay {i}" for i in range(1, 9)},
    },
}

LAB_CURTAINS: Dict[int, str] = {
    1: "Curtain 1",
    2: "Curtain 2",
    3: "Curtain 3",
    4: "Curtain 4",
}


def has_valid_gemini_key() -> bool:
    """Return True if a non-placeholder GEMINI_API_KEY is configured."""
    return bool(GEMINI_API_KEY) and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
