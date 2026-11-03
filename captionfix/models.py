"""Shared records for scenarios, captions, and scores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    speaker: str
    accent: str
    start: float
    end: float
    text: str
    energy: float = 0.75


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    tags: tuple[str, ...]
    noise_kind: str
    snr_db: float
    seed: int
    utterances: tuple[Utterance, ...]

    @property
    def duration(self) -> float:
        if not self.utterances:
            return 0.0
        return max(item.end for item in self.utterances)


@dataclass(frozen=True)
class Word:
    speaker: str
    text: str
    start: float
    end: float
    energy: float
    accent: str


@dataclass(frozen=True)
class Cue:
    speaker: str
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    words: tuple[Word, ...]
    cues: tuple[Cue, ...]


@dataclass(frozen=True)
class Assessment:
    scenario_id: str
    wer: float
    overlap_recall: float
    accent_error_rate: float
    hallucination_rate: float
    speaker_error_rate: float
    passed: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "wer": self.wer,
            "overlap_recall": self.overlap_recall,
            "accent_error_rate": self.accent_error_rate,
            "hallucination_rate": self.hallucination_rate,
            "speaker_error_rate": self.speaker_error_rate,
            "passed": self.passed,
            "notes": list(self.notes),
        }
