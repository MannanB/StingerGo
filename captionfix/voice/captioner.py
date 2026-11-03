"""Caption a scenario by listening to audio with Grok Voice."""

from __future__ import annotations

import json

from captionfix.models import Scenario, Transcript, Word
from captionfix.paths import AUDIO
from captionfix.segment import reference_transcript, to_cues
from captionfix.voice.align import align_speakers
from captionfix.voice.client import transcribe
from captionfix.voice.render import render_wav


class GrokVoiceCaptioner:
    name = "grok-voice-transcribe-2.0"

    def __init__(self, refresh: bool = False) -> None:
        self.refresh = refresh

    def transcribe(self, scenario: Scenario) -> Transcript:
        AUDIO.mkdir(parents=True, exist_ok=True)
        wav_path = AUDIO / f"{scenario.id}.wav"
        stt_path = AUDIO / f"{scenario.id}.stt.json"
        if self.refresh or not wav_path.exists():
            wav_path.write_bytes(render_wav(scenario))
            if stt_path.exists():
                stt_path.unlink()
        if self.refresh or not stt_path.exists():
            payload = transcribe(wav_path.read_bytes())
            stt_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            payload = json.loads(stt_path.read_text(encoding="utf-8"))
        heard = _words(payload)
        aligned = align_speakers(reference_transcript(scenario).words, heard)
        return Transcript(tuple(aligned), tuple(to_cues(aligned)))


def _words(payload: dict) -> list[Word]:
    rows = payload.get("words") or []
    if rows:
        return [
            Word(
                speaker=str(row.get("speaker", "?")),
                text=str(row.get("text", "")),
                start=float(row.get("start", 0.0)),
                end=float(row.get("end", 0.0)),
                energy=1.0,
                accent="",
            )
            for row in rows
            if str(row.get("text", "")).strip()
        ]
    text = str(payload.get("text", "")).strip()
    if not text:
        return []
    duration = float(payload.get("duration") or 0.0)
    tokens = text.split()
    span = (duration or 1.0) / len(tokens)
    return [
        Word("?", token, index * span, (index + 1) * span, 1.0, "")
        for index, token in enumerate(tokens)
    ]
