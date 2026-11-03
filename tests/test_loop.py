"""The closed loop: Grokbot fails, Cursor patches, Grokbot passes."""

import threading
import unittest
import urllib.request

from captionfix.api.server import serve_in_thread
from captionfix.cursor.propose import clear_generated, reset_rules
from captionfix.grokbot.runner import run
from captionfix.models import Transcript, Word
from captionfix.scenarios import load
from captionfix.segment import reference_transcript, to_cues
from captionfix.voice.align import align_speakers


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_rules()
        clear_generated()

    def tearDown(self) -> None:
        reset_rules()
        clear_generated()

    def test_dropped_speaker_fails_overlap(self) -> None:
        scenario = load("overlap_cafe")
        reference = reference_transcript(scenario)
        kept = [word for word in reference.words if word.speaker == "A"]
        transcript = Transcript(tuple(kept), tuple(to_cues(kept)))

        class OnlyA:
            def transcribe(self, _scenario):
                return transcript

        report = run(repeat=1, write=False, captioner=OnlyA())
        cafe = next(result for result in report.results if result.scenario_id == "overlap_cafe")
        self.assertFalse(cafe.passed)
        self.assertLess(cafe.overlap_recall, 0.95)

    def test_diarized_ids_map_back_to_script_speakers(self) -> None:
        reference = [
            Word("A", "hello", 0.0, 0.5, 1.0, ""),
            Word("B", "there", 0.6, 1.0, 1.0, ""),
        ]
        heard = [
            Word("0", "hello", 0.0, 0.5, 1.0, ""),
            Word("1", "there", 0.6, 1.0, 1.0, ""),
        ]
        aligned = align_speakers(reference, heard)
        self.assertEqual([word.speaker for word in aligned], ["A", "B"])


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_rules()
        self.server, port = serve_in_thread(0)
        self.base = f"http://127.0.0.1:{port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        reset_rules()
        clear_generated()

    def test_transcribe_and_assess(self) -> None:
        wrong = _post(
            self.base + "/api/assess",
            {
                "scenario_id": "clean_briefing",
                "cues": [
                    {"speaker": "A", "text": "totally different words", "start": 0.2, "end": 3.4}
                ],
            },
        )
        self.assertFalse(wrong["assessment"]["passed"])
        self.assertGreater(wrong["assessment"]["wer"], 0.5)

    def test_state(self) -> None:
        payload = _get(self.base + "/api/state")
        ids = {item["id"] for item in payload["scenarios"]}
        self.assertIn("overlap_cafe", ids)
        self.assertIn("noise_platform", ids)


def _post(url: str, payload: dict) -> dict:
    import json

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str) -> dict:
    import json

    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
