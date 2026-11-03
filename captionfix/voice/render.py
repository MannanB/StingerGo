"""Turn a scenario script into one WAV: spoken lines, overlap, and noise."""

from __future__ import annotations

import io
import wave

import numpy as np

from captionfix.models import Scenario
from captionfix.voice.client import synthesize

RATE = 16000
VOICES = ("ara", "leo", "rex", "eve", "sal")


def render_wav(scenario: Scenario) -> bytes:
    clips: list[tuple[float, np.ndarray]] = []
    for index, utterance in enumerate(scenario.utterances):
        voice = VOICES[index % len(VOICES)]
        audio = synthesize(utterance.text, voice)
        clips.append((utterance.start, _pcm(audio)))
    return _mix(scenario, clips)


def _pcm(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Grok TTS returned {sample_width * 8}-bit audio, expected 16-bit wav")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != RATE:
        target = int(len(samples) * RATE / rate)
        source_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        target_x = np.linspace(0.0, 1.0, target, endpoint=False)
        samples = np.interp(target_x, source_x, samples).astype(np.float32)
    return samples


def _mix(scenario: Scenario, clips: list[tuple[float, np.ndarray]]) -> bytes:
    end = scenario.duration + 0.3
    for start, samples in clips:
        end = max(end, start + len(samples) / RATE + 0.1)
    total = int(end * RATE)
    mix = np.zeros(total, dtype=np.float32)
    for start, samples in clips:
        offset = int(start * RATE)
        stop = min(total, offset + len(samples))
        mix[offset:stop] += samples[: stop - offset]
    speech = mix[np.abs(mix) > 1]
    speech_rms = float(np.sqrt(np.mean(speech**2))) if speech.size else 1.0
    rng = np.random.default_rng(scenario.seed)
    noise = rng.normal(0.0, 1.0, total).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(noise**2))) or 1.0
    target_rms = speech_rms / (10 ** (scenario.snr_db / 20))
    mix += noise * (target_rms / noise_rms)
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 32000:
        mix *= 32000 / peak
    pcm = np.clip(mix, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
