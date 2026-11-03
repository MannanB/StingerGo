"""The patch Grokbot can apply to Graption's one-face caption rule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Repair:
    kind: str
    function: str
    summary: str
    source: str


REPAIRS: dict[str, Repair] = {
    "overlap": Repair(
        kind="overlap",
        function="attribute_words",
        summary="Keep each speaker on their own caption instead of anchoring the whole chunk to one face.",
        source='''def attribute_words(words):
    """Keep the speaker that actually said each word."""
    return list(words)
''',
    ),
}


KIND_METRIC = {
    "overlap": "overlap_recall",
}
