"""Load the repeatable caption scenarios Grokbot runs."""

from __future__ import annotations

import json

from captionfix.models import Scenario, Utterance
from captionfix.paths import SCENARIOS


def load_all(tag: str | None = None) -> list[Scenario]:
    scenarios = [_read(path) for path in sorted(SCENARIOS.glob("*.json"))]
    if tag is None:
        return scenarios
    return [scenario for scenario in scenarios if tag in scenario.tags]


def load(scenario_id: str) -> Scenario:
    path = SCENARIOS / f"{scenario_id}.json"
    if not path.exists():
        known = ", ".join(scenario.id for scenario in load_all()) or "(none)"
        raise FileNotFoundError(f"Unknown scenario {scenario_id!r}. Known: {known}")
    return _read(path)


def _read(path) -> Scenario:
    raw = json.loads(path.read_text(encoding="utf-8"))
    utterances = tuple(
        Utterance(
            speaker=item["speaker"],
            accent=item["accent"],
            start=float(item["start"]),
            end=float(item["end"]),
            text=item["text"],
            energy=float(item.get("energy", 0.75)),
        )
        for item in raw["utterances"]
    )
    return Scenario(
        id=raw["id"],
        title=raw["title"],
        tags=tuple(raw["tags"]),
        noise_kind=raw["noise_kind"],
        snr_db=float(raw["snr_db"]),
        seed=int(raw["seed"]),
        utterances=utterances,
    )
