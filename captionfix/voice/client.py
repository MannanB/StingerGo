"""Grok Voice API: text-to-speech and speech-to-text."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import certifi

STT_URL = "https://api.x.ai/v1/stt"
TTS_URL = "https://api.x.ai/v1/tts"
MODEL = "grok-voice-transcribe-2.0"


def api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("XAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("Set XAI_API_KEY. Grok Voice will not transcribe without it.")


def synthesize(text: str, voice_id: str) -> bytes:
    payload = {
        "text": text,
        "voice_id": voice_id,
        "language": "en",
        "output_format": {"codec": "wav", "sample_rate": 16000},
    }
    request = urllib.request.Request(
        TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _read(request)


def transcribe(wav: bytes) -> dict:
    body, content_type = _multipart(
        [
            ("model", MODEL),
            ("language", "en"),
            ("diarize", "true"),
            ("format", "false"),
        ],
        field_name="file",
        filename="scene.wav",
        content=wav,
        mime="audio/wav",
    )
    request = urllib.request.Request(
        STT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    raw = _read(request)
    return json.loads(raw.decode("utf-8"))


def _read(request: urllib.request.Request) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Grok Voice returned {exc.code}: {detail}") from exc


def _multipart(
    fields: list[tuple[str, str]],
    field_name: str,
    filename: str,
    content: bytes,
    mime: str,
) -> tuple[bytes, str]:
    boundary = f"----captionfix{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(content)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
