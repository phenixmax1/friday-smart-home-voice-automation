"""
speech_engine.py
----------------
Cloud-only studio female TTS (ZERO `pyttsx3` / robotic on-device TTS) and
once-calibrated Speech-to-Text (`recognize_google`, `language="en-IN"`).

Cloud Voice Engine:
  1. Google Gemini Cloud TTS (`gemini-3.8-flash-lite-tts` / `gemini-3.8-flash-tts`)
     via `google-genai` Interactions API using `api_version="v1beta"`, female
     voice `"Kore"`, and `speech_metadata` F.R.I.D.A.Y. styling.
  2. Microsoft Edge Neural TTS (`en-IE-EmilyNeural`, Irish female F.R.I.D.A.Y.
     voice at `+8%` rate) as an ultra-fast option or automatic fallback if
     Gemini TTS is unavailable or rate-limited.

Audio Playback:
  - Uses `pygame.mixer` initialized once at startup (or `soundfile + sounddevice`
    if `pygame` isn't installed) with local `%TEMP%` caching for instant replay.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import re
import tempfile
import threading
import time
import wave
from typing import Any, List, Optional

from google import genai
import speech_recognition as sr

import config

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None  # type: ignore[assignment]

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Initialize pygame.mixer ONCE at import/startup for zero-overhead playback
# ---------------------------------------------------------------------------
_MIXER_READY: bool = False
if pygame is not None:
    try:
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        _MIXER_READY = True
    except Exception as exc:
        print(f"[Audio Warning]: pygame.mixer.init failed ({exc!r}).")

_SPEAK_LOCK = threading.Lock()
_GEMINI_TTS_CLIENT: Optional[genai.Client] = None
_UNAVAILABLE_TTS_MODELS: set[str] = set()
_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_LOOP_LOCK = threading.Lock()
_TTS_CACHE_DIR = os.path.join(tempfile.gettempdir(), "friday_cloud_tts_cache")

# Once-calibrated microphone recognizer (avoids re-calibrating before every command)
_RECOGNIZER: Optional[sr.Recognizer] = None
_MICROPHONE: Optional[Any] = None


def _get_gemini_client() -> Optional[genai.Client]:
    """
    Return a reusable `google.genai.Client` configured with `v1beta` and
    fail-fast retry options (`attempts=1`) so a 503 switches instantly to Edge TTS.
    """
    global _GEMINI_TTS_CLIENT
    if not config.has_valid_gemini_key():
        return None
    if _GEMINI_TTS_CLIENT is None:
        _GEMINI_TTS_CLIENT = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options={
                "api_version": "v1beta",
                "timeout": config.LLM_TIMEOUT_MS,
                "retry_options": {"attempts": 1},
            },
        )
    return _GEMINI_TTS_CLIENT


def _get_async_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background asyncio loop for fast edge-tts synthesis."""
    global _ASYNC_LOOP
    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="friday-edge-tts", daemon=True
            ).start()
            _ASYNC_LOOP = loop
        return _ASYNC_LOOP


def _cache_file(cache_key: str, ext: str) -> str:
    """Return a deterministic cache file path inside local `%TEMP%`."""
    os.makedirs(_TTS_CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(_TTS_CACHE_DIR, f"{digest}.{ext}")


# ---------------------------------------------------------------------------
# In-Memory Audio Buffer Helpers
# ---------------------------------------------------------------------------
def _ensure_wav_buffer(audio_bytes: bytes, mime_type: str = "") -> io.BytesIO:
    """
    Wrap raw 24 kHz 16-bit mono PCM in a WAV RIFF header via `io.BytesIO` +
    `wave` if needed; if already a valid RIFF WAV (default on Gemini 3.8 TTS),
    return it directly as an in-memory `io.BytesIO` buffer.
    """
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return io.BytesIO(audio_bytes)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit PCM
        wf.setframerate(24000)   # 24 kHz Gemini TTS native rate
        wf.writeframes(audio_bytes)
    wav_buf.seek(0)
    return wav_buf


def _play_in_memory(audio_buf: io.BytesIO, namehint: str) -> bool:
    """
    Play an in-memory WAV or MP3 `io.BytesIO` buffer using `pygame.mixer.music`
    (or `soundfile + sounddevice` if `pygame` isn't installed).
    """
    audio_buf.seek(0)

    if _MIXER_READY and pygame is not None:
        try:
            pygame.mixer.music.load(audio_buf, namehint)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.02)
            pygame.mixer.music.unload()
            return True
        except Exception as exc:
            print(f"[Audio Warning]: pygame playback error ({exc!r}).")

    if sf is not None and sd is not None:
        try:
            audio_buf.seek(0)
            samples, sample_rate = sf.read(audio_buf, dtype="float32")
            sd.play(samples, samplerate=sample_rate, blocking=True)
            return True
        except Exception as exc:
            print(f"[Audio Warning]: soundfile playback error ({exc!r}).")

    return False


# ---------------------------------------------------------------------------
# Cloud TTS Backends (Gemini Cloud TTS + Edge Neural F.R.I.D.A.Y. Fallback)
# ---------------------------------------------------------------------------
def _candidate_gemini_tts_models() -> List[str]:
    """Return ordered Gemini TTS models that have not returned 404 in this session."""
    models = [config.TTS_MODEL] + list(getattr(config, "TTS_FALLBACK_MODELS", []))
    return [m for m in models if m and m not in _UNAVAILABLE_TTS_MODELS]


def _speak_with_gemini_tts(text: str) -> bool:
    """
    Synthesize speech via Google Gemini Cloud TTS (`v1beta` Interactions API)
    using voice `config.GEMINI_VOICE_NAME` ("Kore") and play from memory.
    """
    client = _get_gemini_client()
    if client is None:
        return False

    models = _candidate_gemini_tts_models()
    if not models:
        return False

    for model_name in models:
        cache_key = f"gemini|{model_name}|{config.GEMINI_VOICE_NAME}|{config.GEMINI_VOICE_STYLE}|{text}"
        cached = _cache_file(cache_key, "wav")
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            try:
                with open(cached, "rb") as f:
                    return _play_in_memory(io.BytesIO(f.read()), namehint="wav")
            except OSError:
                pass

        try:
            interaction = client.interactions.create(
                model=model_name,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                                "annotations": [
                                    {
                                        "type": "speech_metadata",
                                        "style": config.GEMINI_VOICE_STYLE,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [{"voice": config.GEMINI_VOICE_NAME}]
                },
            )
            audio_obj = getattr(interaction, "output_audio", None)
            if not audio_obj or not getattr(audio_obj, "data", None):
                continue

            raw_bytes = base64.b64decode(audio_obj.data)
            mime_type = str(getattr(audio_obj, "mime_type", ""))
            wav_buf = _ensure_wav_buffer(raw_bytes, mime_type)

            try:
                with open(cached, "wb") as f:
                    f.write(wav_buf.getvalue())
            except OSError:
                pass

            return _play_in_memory(wav_buf, namehint="wav")

        except Exception as exc:
            err_text = str(exc)
            if "404" in err_text or "not_found" in err_text.lower():
                _UNAVAILABLE_TTS_MODELS.add(model_name)
                print(
                    f"[TTS Info]: Model '{model_name}' not enabled for this API key; "
                    "remembering and falling back cleanly."
                )
            else:
                print(f"[TTS Fallback]: Gemini TTS error ({exc!r}); switching to Edge Neural TTS.")

    return False


_EDGE_VOICE_FALLBACKS: List[str] = [
    config.EDGE_FRIDAY_VOICE,   # en-IE-EmilyNeural (Irish F.R.I.D.A.Y.)
    "en-GB-SoniaNeural",        # Crisp British female
    "en-GB-LibbyNeural",        # Warm British female
    "en-US-AriaNeural",         # Clear US female
]


def _sanitize_for_tts(text: str) -> str:
    """Clean technical tokens (like underscores or `.env`) that can trip SSML parsers."""
    cleaned = text.replace(".env", "dot env")
    cleaned = cleaned.replace("F.R.I.D.A.Y.", "Friday")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"[*`#<>]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _speak_with_edge_tts(text: str) -> bool:
    """
    Synthesize F.R.I.D.A.Y.'s female neural voice entirely in memory using
    `edge-tts`, with text sanitization, voice fallbacks (to prevent
    `NoAudioReceived`), and local `%TEMP%` caching.
    """
    if edge_tts is None:
        print("[TTS Error]: edge-tts is not installed.")
        return False

    spoken_text = _sanitize_for_tts(text)
    if not spoken_text:
        return False

    # Deduplicate voice order while keeping config.EDGE_FRIDAY_VOICE first
    voices = list(dict.fromkeys([config.EDGE_FRIDAY_VOICE] + _EDGE_VOICE_FALLBACKS))

    last_exc: Optional[Exception] = None
    for voice_name in voices:
        for rate_value in (config.EDGE_VOICE_RATE, "+0%"):
            cache_key = f"edge|{voice_name}|{rate_value}|{spoken_text}"
            cached = _cache_file(cache_key, "mp3")
            if os.path.exists(cached) and os.path.getsize(cached) > 0:
                try:
                    with open(cached, "rb") as f:
                        return _play_in_memory(io.BytesIO(f.read()), namehint="mp3")
                except OSError:
                    pass

            async def _stream_mp3(v: str = voice_name, r: str = rate_value) -> bytes:
                communicate = edge_tts.Communicate(text=spoken_text, voice=v, rate=r)
                buf = bytearray()
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio":
                        buf.extend(chunk["data"])
                return bytes(buf)

            try:
                future = asyncio.run_coroutine_threadsafe(_stream_mp3(), _get_async_loop())
                mp3_bytes = future.result(timeout=12.0)
                if mp3_bytes:
                    try:
                        with open(cached, "wb") as f:
                            f.write(mp3_bytes)
                    except OSError:
                        pass
                    return _play_in_memory(io.BytesIO(mp3_bytes), namehint="mp3")
            except Exception as exc:
                last_exc = exc
                # If default rate failed with NoAudioReceived, try next voice cleanly
                continue

    print(f"[TTS Error]: Edge Neural TTS failed ({last_exc!r}).")
    return False


def speak(text: str) -> None:
    """
    Print and speak F.R.I.D.A.Y.'s response using Cloud Neural TTS ONLY
    (zero offline/robotic `pyttsx3`).
    """
    cleaned = text.strip()
    if not cleaned:
        return

    print(f"\n[F.R.I.D.A.Y.]: {cleaned}")

    with _SPEAK_LOCK:
        if config.PREFER_FAST_EDGE_VOICE:
            if _speak_with_edge_tts(cleaned):
                return
            _speak_with_gemini_tts(cleaned)
        else:
            if _speak_with_gemini_tts(cleaned):
                return
            _speak_with_edge_tts(cleaned)


# ---------------------------------------------------------------------------
# Microphone Calibration (Once at Startup) & Fast Listening
# ---------------------------------------------------------------------------
class _SoundDeviceStream:
    """Fallback PCM reader when PyAudio is unavailable (e.g., Python 3.14 Windows)."""

    def __init__(self, raw_stream: Any) -> None:
        self._raw_stream = raw_stream

    def read(self, size: int) -> bytes:
        data, _overflowed = self._raw_stream.read(size)
        return bytes(data)

    def close(self) -> None:
        try:
            self._raw_stream.stop()
        finally:
            self._raw_stream.close()


class _SoundDeviceMicrophone(sr.AudioSource):
    """Drop-in `sr.AudioSource` backed by `sounddevice` when PyAudio is absent."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024) -> None:
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = 2
        self.CHUNK = chunk_size
        self.stream: Optional[_SoundDeviceStream] = None

    def __enter__(self) -> "_SoundDeviceMicrophone":
        if sd is None:
            raise OSError("Neither PyAudio nor sounddevice is installed.")
        raw_stream = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK,
            channels=1,
            dtype="int16",
        )
        raw_stream.start()
        self.stream = _SoundDeviceStream(raw_stream)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            finally:
                self.stream = None


def _create_microphone() -> Any:
    """Use `sr.Microphone()` (PyAudio) when installed, else `sounddevice`."""
    try:
        return sr.Microphone()
    except (OSError, AttributeError):
        if sd is not None:
            return _SoundDeviceMicrophone()
        raise


def init_microphone() -> bool:
    """
    Initialize the microphone and calibrate ambient noise ONCE at startup so
    every subsequent `listen()` call starts immediately with zero calibration lag.
    """
    global _RECOGNIZER, _MICROPHONE

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = config.STT_PAUSE_THRESHOLD
    recognizer.non_speaking_duration = config.STT_NON_SPEAKING_DURATION

    try:
        microphone = _create_microphone()
        with microphone as source:
            print("[Mic]: Calibrating ambient noise (once at startup)...")
            recognizer.adjust_for_ambient_noise(
                source, duration=config.AMBIENT_NOISE_DURATION
            )
            recognizer.energy_threshold = max(250.0, recognizer.energy_threshold * 1.2)
        _RECOGNIZER = recognizer
        _MICROPHONE = microphone
        print(f"[Mic]: Ready (energy threshold={recognizer.energy_threshold:.0f}).")
        return True
    except Exception as exc:
        print(f"[Mic Warning]: Microphone initialization failed ({exc!r}).")
        _RECOGNIZER = None
        _MICROPHONE = None
        return False


def listen() -> str:
    """
    Capture a single spoken command immediately (using the pre-calibrated
    recognizer) and transcribe via Google Speech Recognition (`language="en-IN"`).
    """
    if _RECOGNIZER is None or _MICROPHONE is None:
        if not init_microphone():
            return ""

    assert _RECOGNIZER is not None and _MICROPHONE is not None

    try:
        with _MICROPHONE as source:
            print("[Mic]: Listening...")
            audio = _RECOGNIZER.listen(
                source,
                timeout=config.LISTEN_TIMEOUT,
                phrase_time_limit=config.PHRASE_TIME_LIMIT,
            )
    except sr.WaitTimeoutError:
        return ""
    except Exception as exc:
        print(f"[Mic Error]: Audio capture failed ({exc!r}).")
        return ""

    try:
        text = _RECOGNIZER.recognize_google(audio, language=config.STT_LANGUAGE).strip()
        if text:
            print(f"[You]: {text}")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        print(f"[STT Error]: Google Speech Recognition unreachable ({exc!r}).")
        return ""
