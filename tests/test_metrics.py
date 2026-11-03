"""Scores for caption text, overlap, accent, and noise."""

import unittest

from captionfix.assess.metrics import assess, word_error_rate
from captionfix.scenarios import load


class WordErrorRateTest(unittest.TestCase):
    def test_substitution(self) -> None:
        rate = word_error_rate(["the", "cat", "sat"], ["the", "cap", "sat"])
        self.assertAlmostEqual(rate, 1 / 3)

    def test_deletion(self) -> None:
        rate = word_error_rate(["a", "b", "c"], ["a", "c"])
        self.assertAlmostEqual(rate, 1 / 3)

    def test_perfect(self) -> None:
        self.assertEqual(word_error_rate(["one"], ["one"]), 0.0)


class CleanScenarioTest(unittest.TestCase):
    def test_wrong_caption_fails(self) -> None:
        from captionfix.models import Cue, Transcript
        from captionfix.segment import to_cues, words_from_cues

        scenario = load("clean_briefing")
        cues = [Cue("A", "totally different words", 0.2, 3.4)]
        words = words_from_cues(cues)
        score = assess(scenario, Transcript(tuple(words), tuple(to_cues(words))))
        self.assertFalse(score.passed)
        self.assertGreater(score.wer, 0.5)

    def test_matching_accent_line_passes(self) -> None:
        from captionfix.models import Cue, Transcript
        from captionfix.segment import to_cues, words_from_cues

        scenario = load("accent_indian")
        utterance = scenario.utterances[0]
        cues = [Cue(utterance.speaker, utterance.text, utterance.start, utterance.end)]
        words = words_from_cues(cues)
        score = assess(scenario, Transcript(tuple(words), tuple(to_cues(words))))
        self.assertEqual(score.accent_error_rate, 0.0)
        self.assertTrue(score.passed)
