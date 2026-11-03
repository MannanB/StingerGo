"""Turn scenario utterances into timed words and caption cues."""

from __future__ import annotations

from captionfix.models import Cue, Scenario, Transcript, Utterance, Word


def expand(scenario: Scenario) -> list[Word]:
    words: list[Word] = []
    for utterance in scenario.utterances:
        words.extend(_utterance_words(utterance))
    words.sort(key=lambda word: (word.start, word.speaker))
    return words


def reference_transcript(scenario: Scenario) -> Transcript:
    words = expand(scenario)
    return Transcript(tuple(words), tuple(to_cues(words)))


def to_cues(words: list[Word]) -> list[Cue]:
    if not words:
        return []
    ordered = sorted(words, key=lambda word: (word.start, word.speaker, word.end))
    groups: list[list[Word]] = [[ordered[0]]]
    for word in ordered[1:]:
        previous = groups[-1][-1]
        same_voice = word.speaker == previous.speaker
        close = word.start - previous.end <= 0.35
        if same_voice and close:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [_cue(group) for group in groups]


def to_vtt(cues: tuple[Cue, ...] | list[Cue]) -> str:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{_stamp(cue.start)} --> {_stamp(cue.end)}")
        lines.append(f"<{cue.speaker}> {cue.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def words_from_cues(cues: list[Cue]) -> list[Word]:
    """Rebuild a word stream from caption cues posted by an outside captioner."""
    words: list[Word] = []
    for cue in cues:
        tokens = cue.text.split()
        if not tokens:
            continue
        span = max(cue.end - cue.start, 0.01) / len(tokens)
        for index, token in enumerate(tokens):
            start = cue.start + index * span
            words.append(
                Word(
                    speaker=cue.speaker,
                    text=token,
                    start=start,
                    end=start + span,
                    energy=1.0,
                    accent="",
                )
            )
    words.sort(key=lambda word: (word.start, word.speaker))
    return words


def _utterance_words(utterance: Utterance) -> list[Word]:
    tokens = utterance.text.split()
    if not tokens:
        return []
    span = (utterance.end - utterance.start) / len(tokens)
    return [
        Word(
            speaker=utterance.speaker,
            text=token,
            start=utterance.start + index * span,
            end=utterance.start + (index + 1) * span,
            energy=utterance.energy,
            accent=utterance.accent,
        )
        for index, token in enumerate(tokens)
    ]


def _cue(words: list[Word]) -> Cue:
    return Cue(
        speaker=words[0].speaker,
        text=" ".join(word.text for word in words),
        start=words[0].start,
        end=words[-1].end,
    )


def _stamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
