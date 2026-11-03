"""Match diarized speaker ids onto the script's speaker names."""

from __future__ import annotations

from captionfix.models import Word


def align_speakers(reference: list[Word] | tuple[Word, ...], hypothesis: list[Word]) -> list[Word]:
    hyp_ids = list(dict.fromkeys(word.speaker for word in hypothesis))
    ref_ids = list(dict.fromkeys(word.speaker for word in reference))
    scores = {
        (hyp, ref): _overlap(hypothesis, reference, hyp, ref)
        for hyp in hyp_ids
        for ref in ref_ids
    }
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for hyp, ref in sorted(scores, key=lambda pair: scores[pair], reverse=True):
        if hyp in mapping or ref in used or scores[(hyp, ref)] <= 0:
            continue
        mapping[hyp] = ref
        used.add(ref)
    return [
        word.__class__(
            mapping.get(word.speaker, word.speaker),
            word.text,
            word.start,
            word.end,
            word.energy,
            word.accent,
        )
        for word in hypothesis
    ]


def _overlap(hypothesis: list[Word], reference: list[Word] | tuple[Word, ...], hyp: str, ref: str) -> float:
    total = 0.0
    for left in hypothesis:
        if left.speaker != hyp:
            continue
        for right in reference:
            if right.speaker != ref:
                continue
            total += max(0.0, min(left.end, right.end) - max(left.start, right.start))
    return total
