"""Score a hypothesis caption against a scenario's reference speech."""

from __future__ import annotations

from collections import Counter

from captionfix.models import Assessment, Scenario, Transcript, Word
from captionfix.segment import reference_transcript

# A scenario passes only when every gate holds.
WER_MAX = 0.08
OVERLAP_RECALL_MIN = 0.95
ACCENT_ERROR_MAX = 0.05
HALLUCINATION_MAX = 0.02


def assess(scenario: Scenario, hypothesis: Transcript) -> Assessment:
    reference = reference_transcript(scenario)
    ref_tokens = [_token(word.text) for word in reference.words]
    hyp_tokens = [_token(word.text) for word in hypothesis.words]
    wer = word_error_rate(ref_tokens, hyp_tokens)
    overlap_recall = _overlap_recall(scenario, hypothesis.words)
    accent_error_rate = _accent_error_rate(scenario, hypothesis.words)
    hallucination_rate = _hallucination_rate(scenario, reference.words, hypothesis.words)
    speaker_error_rate = _speaker_error_rate(reference.words, hypothesis.words)
    notes = _notes(wer, overlap_recall, accent_error_rate, hallucination_rate)
    passed = not notes
    return Assessment(
        scenario_id=scenario.id,
        wer=wer,
        overlap_recall=overlap_recall,
        accent_error_rate=accent_error_rate,
        hallucination_rate=hallucination_rate,
        speaker_error_rate=speaker_error_rate,
        passed=passed,
        notes=tuple(notes),
    )


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    substitutions, deletions, insertions = _edit_counts(reference, hypothesis)
    return (substitutions + deletions + insertions) / len(reference)


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    # cost, substitutions, deletions, insertions
    table = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            substitute = table[i - 1][j - 1]
            delete = table[i - 1][j]
            insert = table[i][j - 1]
            candidates = (
                (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            table[i][j] = min(candidates)
    _, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions


def _overlap_recall(scenario: Scenario, hypothesis: tuple[Word, ...] | list[Word]) -> float:
    windows = _overlap_windows(scenario)
    if not windows:
        return 1.0
    recalled = 0
    for start, end, speakers in windows:
        present = {
            word.speaker
            for word in hypothesis
            if word.speaker in speakers and word.start < end and word.end > start
        }
        if speakers <= present:
            recalled += 1
    return recalled / len(windows)


def _overlap_windows(scenario: Scenario) -> list[tuple[float, float, frozenset[str]]]:
    windows: list[tuple[float, float, frozenset[str]]] = []
    utterances = scenario.utterances
    for index, left in enumerate(utterances):
        for right in utterances[index + 1 :]:
            if left.speaker == right.speaker:
                continue
            start = max(left.start, right.start)
            end = min(left.end, right.end)
            if end - start > 0.05:
                windows.append((start, end, frozenset({left.speaker, right.speaker})))
    return windows


_NEUTRAL_ACCENTS = {"", "general_american", "neutral"}


def _accent_error_rate(scenario: Scenario, hypothesis: list[Word]) -> float:
    """Word errors inside an accented line that was actually captioned.

    Reference words are spread evenly across a script window, so they are
    scored against the whole utterance, not each synthetic slice. A missing
    speaker is an overlap failure, not an accent failure.
    """
    targets = 0
    errors = 0
    for utterance in scenario.utterances:
        if utterance.accent in _NEUTRAL_ACCENTS:
            continue
        reference = [token for token in (_token(part) for part in utterance.text.split()) if token]
        if not reference:
            continue
        heard = [
            _token(word.text)
            for word in hypothesis
            if word.speaker == utterance.speaker
            and word.start < utterance.end
            and word.end > utterance.start
            and _token(word.text)
        ]
        if not heard:
            continue
        targets += len(reference)
        substitutions, deletions, insertions = _edit_counts(reference, heard)
        errors += substitutions + deletions + insertions
    if targets == 0:
        return 0.0
    return errors / targets


def _hallucination_rate(
    scenario: Scenario, reference: list[Word], hypothesis: list[Word]
) -> float:
    if not hypothesis:
        return 0.0
    ref_counts = Counter(_token(word.text) for word in reference)
    hyp_counts = Counter(_token(word.text) for word in hypothesis)
    extra = sum(max(0, hyp_counts[token] - ref_counts[token]) for token in hyp_counts)
    labeled = sum(1 for word in hypothesis if word.speaker == "noise" or word.accent == "noise")
    return max(extra, labeled) / len(hypothesis)


def _speaker_error_rate(reference: list[Word], hypothesis: list[Word]) -> float:
    compared = 0
    mismatches = 0
    for word in hypothesis:
        if word.speaker == "noise":
            continue
        match = _best_reference(word, reference)
        if match is None:
            continue
        compared += 1
        if match.speaker != word.speaker:
            mismatches += 1
    if compared == 0:
        return 0.0
    return mismatches / compared


def _best_reference(word: Word, reference: list[Word]) -> Word | None:
    best: Word | None = None
    best_overlap = 0.0
    for candidate in reference:
        overlap = _time_overlap(word, candidate)
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best


def _time_overlap(left: Word, right: Word) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _notes(
    wer: float,
    overlap_recall: float,
    accent_error_rate: float,
    hallucination_rate: float,
) -> list[str]:
    notes: list[str] = []
    if wer > WER_MAX:
        notes.append(f"word error rate {wer:.2f} is above {WER_MAX:.2f}")
    if overlap_recall < OVERLAP_RECALL_MIN:
        notes.append(
            f"overlap recall {overlap_recall:.2f} is below {OVERLAP_RECALL_MIN:.2f}"
        )
    if accent_error_rate > ACCENT_ERROR_MAX:
        notes.append(
            f"accent error rate {accent_error_rate:.2f} is above {ACCENT_ERROR_MAX:.2f}"
        )
    if hallucination_rate > HALLUCINATION_MAX:
        notes.append(
            f"hallucination rate {hallucination_rate:.2f} is above {HALLUCINATION_MAX:.2f}"
        )
    return notes


def _token(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum() or char == "'")
