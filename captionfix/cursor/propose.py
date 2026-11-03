"""Cursor's half of the loop: regression tests and proposed patches."""

from __future__ import annotations

import ast
import difflib
import importlib.util
import io
import json
import shutil
import unittest
from dataclasses import dataclass

from captionfix.cursor.repairs import KIND_METRIC, REPAIRS
from captionfix.grokbot.runner import RunReport, _from_dict, format_report, load_latest, run
from captionfix.paths import LATEST, PATCHES, PRISTINE, PROPOSAL, REGRESSIONS, REPO, RULES, RUNS


@dataclass(frozen=True)
class Finding:
    kind: str
    scenario_id: str
    metric: str
    value: float
    summary: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "scenario_id": self.scenario_id,
            "metric": self.metric,
            "value": self.value,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class Proposal:
    findings: tuple[Finding, ...]
    patches: tuple[str, ...]
    tests: tuple[str, ...]
    diffs: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "patches": list(self.patches),
            "tests": list(self.tests),
            "diffs": self.diffs,
        }


def propose(report: RunReport | None = None) -> Proposal:
    report = report or load_latest()
    if report is None:
        raise FileNotFoundError("No Grokbot run to read. Run `captionfix grokbot run` first.")
    findings = _findings(report)
    PATCHES.mkdir(parents=True, exist_ok=True)
    REGRESSIONS.mkdir(parents=True, exist_ok=True)
    source = RULES.read_text(encoding="utf-8")
    patches: list[str] = []
    tests: list[str] = []
    diffs: dict[str, str] = {}
    cursor = source
    for finding in findings:
        repair = REPAIRS[finding.kind]
        if _already(cursor, repair.function, repair.source):
            continue
        updated = _replace_function(cursor, repair.function, repair.source)
        diff = _diff(cursor, updated)
        patch_path = PATCHES / f"{finding.kind}.diff"
        test_path = REGRESSIONS / f"test_{finding.kind}.py"
        patch_path.write_text(diff, encoding="utf-8")
        test_path.write_text(_regression_test(finding), encoding="utf-8")
        patches.append(patch_path.relative_to(REPO).as_posix())
        tests.append(test_path.relative_to(REPO).as_posix())
        diffs[finding.kind] = diff
        cursor = updated
    proposal = Proposal(findings, tuple(patches), tuple(tests), diffs)
    PROPOSAL.write_text(json.dumps(proposal.to_dict(), indent=2) + "\n", encoding="utf-8")
    return proposal


def apply_proposal(proposal: Proposal | None = None) -> list[str]:
    proposal = proposal or load_proposal()
    if proposal is None or not proposal.findings:
        return []
    source = RULES.read_text(encoding="utf-8")
    applied: list[str] = []
    for finding in proposal.findings:
        repair = REPAIRS[finding.kind]
        if _already(source, repair.function, repair.source):
            continue
        source = _replace_function(source, repair.function, repair.source)
        applied.append(repair.function)
    if applied:
        _validate(source)
        RULES.write_text(source, encoding="utf-8")
    return applied


def load_proposal() -> Proposal | None:
    if not PROPOSAL.exists():
        return None
    raw = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    findings = tuple(Finding(**item) for item in raw["findings"])
    return Proposal(
        findings,
        tuple(raw["patches"]),
        tuple(raw["tests"]),
        dict(raw["diffs"]),
    )


def close_loop(repeat: int = 2, tag: str | None = None) -> dict:
    """Run Grokbot, propose patches, apply them, and run Grokbot again."""
    before = run(repeat=repeat, tag=tag)
    proposal = propose(before)
    applied = apply_proposal(proposal)
    after = run(repeat=repeat, tag=tag)
    regressions = verify_regressions(proposal)
    return {
        "before": before.to_dict(),
        "proposal": proposal.to_dict(),
        "applied": applied,
        "regressions": regressions,
        "after": after.to_dict(),
    }


def reset_rules() -> None:
    """Restore the known-failing captioner so the lab can be run again."""
    shutil.copyfile(PRISTINE, RULES)


def rules_are_pristine() -> bool:
    if not PRISTINE.exists() or not RULES.exists():
        return False
    return RULES.read_text(encoding="utf-8") == PRISTINE.read_text(encoding="utf-8")


def format_proposal(proposal: Proposal) -> str:
    if not proposal.findings:
        return "Cursor found nothing to patch. Grokbot is clean for this suite."
    lines = ["Cursor proposal"]
    for finding in proposal.findings:
        lines.append(
            f"  {finding.kind:<8} {finding.scenario_id:<22} "
            f"{finding.metric}={finding.value:.2f}"
        )
        lines.append(f"           {finding.summary}")
    if proposal.tests:
        lines.append("Regression tests")
        lines.extend(f"  {path}" for path in proposal.tests)
    if proposal.patches:
        lines.append("Patches")
        lines.extend(f"  {path}" for path in proposal.patches)
    return "\n".join(lines)


def format_loop(payload: dict) -> str:
    proposal = Proposal(
        tuple(Finding(**item) for item in payload["proposal"]["findings"]),
        tuple(payload["proposal"]["patches"]),
        tuple(payload["proposal"]["tests"]),
        dict(payload["proposal"]["diffs"]),
    )
    regression_lines = ["Regression tests"]
    for item in payload.get("regressions", []):
        mark = "pass" if item["passed"] else "fail"
        regression_lines.append(f"  {mark}  {item['test']}")
    if len(regression_lines) == 1:
        regression_lines.append("  none")
    return "\n\n".join(
        [
            "Grokbot, before",
            format_report(_from_dict(payload["before"])),
            format_proposal(proposal),
            f"Applied: {', '.join(payload['applied']) or 'nothing'}",
            "\n".join(regression_lines),
            "Grokbot, after",
            format_report(_from_dict(payload["after"])),
        ]
    )


def verify_regressions(proposal: Proposal) -> list[dict]:
    results: list[dict] = []
    for relative in proposal.tests:
        path = REPO / relative
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            results.append({"test": relative, "passed": False, "ran": 0})
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        outcome = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        results.append(
            {
                "test": relative,
                "passed": outcome.wasSuccessful(),
                "ran": outcome.testsRun,
            }
        )
    return results


def _findings(report: RunReport) -> tuple[Finding, ...]:
    chosen: list[Finding] = []
    for kind, metric in KIND_METRIC.items():
        pool = [
            result
            for result in report.results
            if kind in result.tags and not result.passed and _tripped(result, kind)
        ]
        if not pool:
            continue
        pure = [result for result in pool if list(result.tags) == [kind]]
        result = _worst(pure or pool, metric)
        chosen.append(
            Finding(
                kind=kind,
                scenario_id=result.scenario_id,
                metric=metric,
                value=getattr(result, metric),
                summary=REPAIRS[kind].summary,
            )
        )
    return tuple(chosen)


def _tripped(result, kind: str) -> bool:
    if kind == "overlap":
        return result.overlap_recall < 0.95
    if kind == "accent":
        return result.accent_error_rate > 0.05
    if kind == "noise":
        return result.hallucination_rate > 0.02 or result.wer > 0.08
    return False


def _worst(results, metric: str):
    if metric == "overlap_recall":
        return min(results, key=lambda result: getattr(result, metric))
    return max(results, key=lambda result: getattr(result, metric))


def _regression_test(finding: Finding) -> str:
    metric = finding.metric
    if metric == "overlap_recall":
        comparison = "score.overlap_recall >= 0.95 and score.wer <= 0.08"
        detail = "(score.overlap_recall, score.wer)"
    elif metric == "accent_error_rate":
        comparison = "score.accent_error_rate <= 0.05 and score.wer <= 0.08"
        detail = "(score.accent_error_rate, score.wer)"
    else:
        comparison = "score.hallucination_rate <= 0.02 and score.wer <= 0.08"
        detail = "(score.hallucination_rate, score.wer)"
    return f'''"""Regression for the {finding.kind} failure Grokbot found on {finding.scenario_id}."""

import unittest

from captionfix.assess.metrics import assess
from captionfix.captioner.studio import StudioCaptioner
from captionfix.scenarios import load


class Test{finding.kind.title()}Regression(unittest.TestCase):
    def test_{finding.scenario_id}(self) -> None:
        scenario = load("{finding.scenario_id}")
        score = assess(scenario, StudioCaptioner().transcribe(scenario))
        self.assertTrue(
            {comparison},
            f"{finding.kind} regression failed: {{{detail}}}",
        )


if __name__ == "__main__":
    unittest.main()
'''


def _replace_function(source: str, name: str, new_def: str) -> str:
    start, end = _function_span(source, name)
    block = new_def if new_def.endswith("\n") else new_def + "\n"
    if end < len(source) and not block.endswith("\n\n"):
        block += "\n"
    return source[:start] + block + source[end:]


def _already(source: str, name: str, new_def: str) -> bool:
    start, end = _function_span(source, name)
    return _compact(source[start:end]) == _compact(new_def)


def _function_span(source: str, name: str) -> tuple[int, int]:
    marker = f"def {name}("
    start = source.find(marker)
    if start == -1:
        raise LookupError(f"{name} is not defined in {RULES.relative_to(REPO).as_posix()}")
    line_start = source.rfind("\n", 0, start)
    start = 0 if line_start == -1 else line_start + 1
    rest = source[start + len(marker) :]
    next_def = rest.find("\ndef ")
    end = len(source) if next_def == -1 else start + len(marker) + next_def + 1
    return start, end


def _diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{RULES.relative_to(REPO).as_posix()}",
        tofile=f"b/{RULES.relative_to(REPO).as_posix()}",
        n=3,
    )
    text = "".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _validate(source: str) -> None:
    ast.parse(source)


def _compact(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def clear_generated() -> None:
    if PATCHES.exists():
        shutil.rmtree(PATCHES)
    if REGRESSIONS.exists():
        shutil.rmtree(REGRESSIONS)
    if RUNS.exists():
        shutil.rmtree(RUNS)
    LATEST.unlink(missing_ok=True)
