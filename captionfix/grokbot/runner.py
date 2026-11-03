"""Grokbot runs the scenario suite and records a repeatable score."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from captionfix.assess.metrics import assess
from captionfix.captioner.studio import StudioCaptioner
from captionfix.models import Scenario
from captionfix.paths import LATEST, RUNS
from captionfix.scenarios import load_all
from captionfix.segment import reference_transcript, to_vtt


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    title: str
    tags: tuple[str, ...]
    wer: float
    overlap_recall: float
    accent_error_rate: float
    hallucination_rate: float
    speaker_error_rate: float
    passed: bool
    notes: tuple[str, ...]
    stable: bool
    reference_vtt: str
    hypothesis_vtt: str
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "tags": list(self.tags),
            "wer": self.wer,
            "overlap_recall": self.overlap_recall,
            "accent_error_rate": self.accent_error_rate,
            "hallucination_rate": self.hallucination_rate,
            "speaker_error_rate": self.speaker_error_rate,
            "passed": self.passed,
            "notes": list(self.notes),
            "stable": self.stable,
            "reference_vtt": self.reference_vtt,
            "hypothesis_vtt": self.hypothesis_vtt,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass(frozen=True)
class RunReport:
    run_id: str
    repeat: int
    tag: str | None
    results: tuple[ScenarioResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    @property
    def stable(self) -> bool:
        return all(result.stable for result in self.results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "repeat": self.repeat,
            "tag": self.tag,
            "stable": self.stable,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": len(self.results),
            "results": [result.to_dict() for result in self.results],
        }


def run(
    repeat: int = 1,
    tag: str | None = None,
    write: bool = True,
    captioner: StudioCaptioner | None = None,
    refresh: bool = False,
) -> RunReport:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    scenarios = load_all(tag)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios for tag {tag!r}" if tag else "No scenarios found")
    captioner = captioner or StudioCaptioner(refresh=refresh)
    results = tuple(_run_scenario(captioner, scenario, repeat) for scenario in scenarios)
    report = RunReport(
        run_id=_run_id(),
        repeat=repeat,
        tag=tag,
        results=results,
    )
    if write:
        _write(report)
    return report


def _run_id() -> str:
    now = time.time()
    millis = int(now * 1000) % 1000
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{millis:03d}"


def load_latest() -> RunReport | None:
    if not LATEST.exists():
        return None
    return _from_dict(json.loads(LATEST.read_text(encoding="utf-8")))


def format_report(report: RunReport) -> str:
    header = (
        f"Grokbot {report.run_id}  repeat={report.repeat}  "
        f"stable={'yes' if report.stable else 'NO'}"
    )
    columns = (
        f"{'scenario':<22} {'wer':>6} {'overlap':>8} {'accent':>7} "
        f"{'halluc':>7} {'result':>6}"
    )
    lines = [header, columns]
    for result in report.results:
        lines.append(
            f"{result.scenario_id:<22} {result.wer:6.2f} {result.overlap_recall:8.2f} "
            f"{result.accent_error_rate:7.2f} {result.hallucination_rate:7.2f} "
            f"{'pass' if result.passed else 'fail':>6}"
        )
    lines.append(
        f"{report.passed_count}/{len(report.results)} passed, "
        f"{report.failed_count} failed."
    )
    return "\n".join(lines)


def _run_scenario(captioner: StudioCaptioner, scenario: Scenario, repeat: int) -> ScenarioResult:
    started = time.perf_counter()
    transcripts = [captioner.transcribe(scenario) for _ in range(repeat)]
    elapsed_ms = (time.perf_counter() - started) * 1000
    signature = [tuple((word.speaker, word.text, word.start) for word in item.words) for item in transcripts]
    score = assess(scenario, transcripts[0])
    reference = reference_transcript(scenario)
    return ScenarioResult(
        scenario_id=scenario.id,
        title=scenario.title,
        tags=scenario.tags,
        wer=score.wer,
        overlap_recall=score.overlap_recall,
        accent_error_rate=score.accent_error_rate,
        hallucination_rate=score.hallucination_rate,
        speaker_error_rate=score.speaker_error_rate,
        passed=score.passed,
        notes=score.notes,
        stable=all(item == signature[0] for item in signature),
        reference_vtt=to_vtt(reference.cues),
        hypothesis_vtt=to_vtt(transcripts[0].cues),
        elapsed_ms=elapsed_ms,
    )


def _write(report: RunReport) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2) + "\n"
    (RUNS / f"{report.run_id}.json").write_text(payload, encoding="utf-8")
    LATEST.write_text(payload, encoding="utf-8")


def _from_dict(raw: dict) -> RunReport:
    results = tuple(
        ScenarioResult(
            scenario_id=item["scenario_id"],
            title=item["title"],
            tags=tuple(item["tags"]),
            wer=item["wer"],
            overlap_recall=item["overlap_recall"],
            accent_error_rate=item["accent_error_rate"],
            hallucination_rate=item["hallucination_rate"],
            speaker_error_rate=item["speaker_error_rate"],
            passed=item["passed"],
            notes=tuple(item["notes"]),
            stable=item["stable"],
            reference_vtt=item["reference_vtt"],
            hypothesis_vtt=item["hypothesis_vtt"],
            elapsed_ms=item["elapsed_ms"],
        )
        for item in raw["results"]
    )
    return RunReport(
        run_id=raw["run_id"],
        repeat=raw["repeat"],
        tag=raw["tag"],
        results=results,
    )
