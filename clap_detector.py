"""
clap_detector.py
----------------
Real-time double-clap detector with automatic ambient-noise calibration,
short-impulse filtering (rejects continuous speech), and minimum-gap echo
rejection so a single clap never double-triggers.

Supports both `sounddevice` (recommended for Python 3.14+ on Windows) and
`PyAudio` automatically.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Callable, Optional

import config

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional backend
    sd = None  # type: ignore[assignment]

try:
    import pyaudio
except ImportError:  # pragma: no cover - optional backend
    pyaudio = None  # type: ignore[assignment]


def compute_rms(pcm_bytes: bytes) -> float:
    """
    Compute the Root Mean Square (RMS) energy of a 16-bit signed little-endian
    mono PCM audio buffer using only the Python standard library.
    """
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", pcm_bytes[: sample_count * 2])
    sum_squares = sum(sample * sample for sample in samples)
    return math.sqrt(sum_squares / sample_count)


class ClapPulseDetector:
    """
    Stateful frame-by-frame impulse analyzer that identifies sharp, brief
    hand claps while rejecting sustained speech, constant noise, and echoes.
    """

    def __init__(
        self,
        ambient_rms: float,
        frame_duration_sec: float,
        min_rms: float = config.CLAP_MIN_RMS,
        threshold_multiplier: float = config.CLAP_THRESHOLD_MULTIPLIER,
        max_duration_sec: float = config.CLAP_MAX_DURATION_SEC,
        required_claps: int = config.CLAP_COUNT_REQUIRED,
        min_gap_sec: float = config.CLAP_MIN_GAP_SEC,
        multi_window_sec: float = config.CLAP_MULTI_WINDOW_SEC,
    ) -> None:
        self.ambient_rms = max(ambient_rms, 1.0)
        self.frame_duration_sec = max(frame_duration_sec, 0.001)
        self.min_rms = min_rms
        self.threshold_multiplier = threshold_multiplier
        self.max_impulse_frames = max(
            1, int(round(max_duration_sec / self.frame_duration_sec))
        )
        self.required_claps = max(1, required_claps)
        self.min_gap_sec = max(0.0, min_gap_sec)
        self.multi_window_sec = multi_window_sec

        self._in_spike = False
        self._spike_frames = 0
        self._clap_timestamps: list[float] = []

    @property
    def threshold(self) -> float:
        """Dynamic RMS threshold above both the noise floor and absolute minimum."""
        return max(self.min_rms, self.ambient_rms * self.threshold_multiplier)

    @property
    def current_clap_count(self) -> int:
        """Number of valid claps currently accumulated within the active window."""
        return len(self._clap_timestamps)

    def feed_rms(self, rms: float, now: Optional[float] = None) -> bool:
        """
        Feed one audio frame's RMS value.

        Returns True when `required_claps` distinct claps occur within
        `multi_window_sec` (respecting `min_gap_sec` echo rejection).
        """
        timestamp = time.monotonic() if now is None else now
        high_threshold = self.threshold
        release_threshold = high_threshold * 0.55

        # Expire stale claps outside the multi-clap window
        if self._clap_timestamps:
            cutoff = timestamp - self.multi_window_sec
            self._clap_timestamps = [t for t in self._clap_timestamps if t >= cutoff]

        if not self._in_spike:
            if rms >= high_threshold:
                self._in_spike = True
                self._spike_frames = 1
            else:
                # Slowly adapt ambient noise floor during quiet frames
                self.ambient_rms = (self.ambient_rms * 0.96) + (rms * 0.04)
            return False

        # Currently inside an acoustic spike
        if rms >= release_threshold:
            self._spike_frames += 1
            return False

        # Spike ended: classify as a clap only if it was brief (impulsive)
        was_valid_clap = 1 <= self._spike_frames <= self.max_impulse_frames
        self._in_spike = False
        self._spike_frames = 0

        if not was_valid_clap:
            return False

        # Reject acoustic echoes too close to the previous clap
        if self._clap_timestamps and (timestamp - self._clap_timestamps[-1]) < self.min_gap_sec:
            return False

        self._clap_timestamps.append(timestamp)

        if len(self._clap_timestamps) >= self.required_claps:
            self._clap_timestamps.clear()
            return True

        return False


def _detect_from_reader(
    read_chunk: Callable[[], bytes],
    frame_duration: float,
    timeout: Optional[float],
) -> bool:
    """Shared calibration and double-clap detection loop for any PCM reader."""
    cal_frames = max(1, int(config.CLAP_CALIBRATION_DURATION / frame_duration))
    rms_samples = []
    for _ in range(cal_frames):
        rms_samples.append(compute_rms(read_chunk()))

    ambient_rms = sum(rms_samples) / len(rms_samples) if rms_samples else 100.0
    detector = ClapPulseDetector(
        ambient_rms=ambient_rms,
        frame_duration_sec=frame_duration,
    )

    print(
        f"[Standby]: Armed — clap twice to activate "
        f"(noise floor={ambient_rms:.0f}, threshold={detector.threshold:.0f})."
    )

    start_time = time.monotonic()
    last_reported_count = 0

    while True:
        if timeout is not None and (time.monotonic() - start_time) >= timeout:
            return False

        rms = compute_rms(read_chunk())
        triggered = detector.feed_rms(rms)

        if triggered:
            print("[Standby]: Double clap detected.")
            return True

        count = detector.current_clap_count
        if count > last_reported_count and config.CLAP_COUNT_REQUIRED > 1:
            print(f"[Standby]: Clap {count}/{config.CLAP_COUNT_REQUIRED}...")
        last_reported_count = count


def wait_for_clap(timeout: Optional[float] = None) -> bool:
    """
    Listen silently on the default microphone until a double clap is detected.

    The audio stream is closed before returning so subsequent TTS output
    ("All systems online.") cannot feed back into the microphone.
    """
    sample_rate = config.CLAP_SAMPLE_RATE
    chunk_size = config.CLAP_CHUNK_SIZE
    frame_duration = chunk_size / float(sample_rate)

    if sd is not None:
        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=chunk_size,
                channels=1,
                dtype="int16",
            ) as stream:
                def _read_sd() -> bytes:
                    data, _overflowed = stream.read(chunk_size)
                    return bytes(data)

                return _detect_from_reader(_read_sd, frame_duration, timeout)
        except Exception as exc:
            print(f"[Standby Error]: sounddevice microphone error ({exc}).")
            return False

    if pyaudio is not None:
        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )

            def _read_pa() -> bytes:
                return stream.read(chunk_size, exception_on_overflow=False)

            return _detect_from_reader(_read_pa, frame_duration, timeout)
        except OSError as exc:
            print(f"[Standby Error]: PyAudio microphone error ({exc}).")
            return False
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            audio.terminate()

    print("[Standby Error]: No microphone backend installed (sounddevice or PyAudio).")
    return False
