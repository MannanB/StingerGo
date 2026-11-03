"""HTTP surface for transcription, assessment, Grokbot, and the Cursor loop."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from captionfix.paths import AUDIO

from captionfix.assess.metrics import assess
from captionfix.captioner.studio import StudioCaptioner
from captionfix.cursor.propose import (
    apply_proposal,
    close_loop,
    load_proposal,
    propose,
    reset_rules,
    rules_are_pristine,
)
from captionfix.grokbot.ask import ask
from captionfix.grokbot.runner import load_latest, run
from captionfix.models import Cue, Transcript
from captionfix.scenarios import load, load_all
from captionfix.segment import reference_transcript, to_cues, to_vtt, words_from_cues

HTML = Path(__file__).with_name("index.html")


def lab_state() -> dict:
    report = load_latest()
    proposal = load_proposal()
    return {
        "pristine": rules_are_pristine(),
        "scenarios": [_scenario_dict(scenario) for scenario in load_all()],
        "report": None if report is None else report.to_dict(),
        "proposal": None if proposal is None else proposal.to_dict(),
    }


def transcribe_scenario(scenario_id: str) -> dict:
    scenario = load(scenario_id)
    transcript = StudioCaptioner().transcribe(scenario)
    return _bundle(scenario, transcript)


def assess_cues(scenario_id: str, cues: list[dict]) -> dict:
    scenario = load(scenario_id)
    parsed = [
        Cue(
            speaker=item["speaker"],
            text=item["text"],
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in cues
    ]
    words = words_from_cues(parsed)
    transcript = Transcript(tuple(words), tuple(to_cues(words)))
    return _bundle(scenario, transcript, score_transcript=transcript)


def _bundle(scenario, transcript: Transcript, score_transcript: Transcript | None = None) -> dict:
    scored = transcript if score_transcript is None else score_transcript
    reference = reference_transcript(scenario)
    return {
        "scenario_id": scenario.id,
        "reference_vtt": to_vtt(reference.cues),
        "hypothesis_vtt": to_vtt(transcript.cues),
        "reference": [_cue_dict(cue) for cue in reference.cues],
        "hypothesis": [_cue_dict(cue) for cue in transcript.cues],
        "assessment": assess(scenario, scored).to_dict(),
    }


def _scenario_dict(scenario) -> dict:
    return {
        "id": scenario.id,
        "title": scenario.title,
        "tags": list(scenario.tags),
        "noise_kind": scenario.noise_kind,
        "snr_db": scenario.snr_db,
        "utterances": [
            {
                "speaker": item.speaker,
                "accent": item.accent,
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "energy": item.energy,
            }
            for item in scenario.utterances
        ],
    }


def _cue_dict(cue: Cue) -> dict:
    return {
        "speaker": cue.speaker,
        "text": cue.text,
        "start": round(cue.start, 3),
        "end": round(cue.end, 3),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CaptionFix/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._bytes(HTML.read_bytes(), "text/html; charset=utf-8", cache=False)
            return
        if path == "/api/state":
            self._json(lab_state())
            return
        if path == "/api/scenarios":
            self._json({"scenarios": lab_state()["scenarios"]})
            return
        if path == "/api/runs/latest":
            report = load_latest()
            if report is None:
                self._error(404, "No Grokbot run yet")
                return
            self._json(report.to_dict())
            return
        if path.startswith("/audio/"):
            self._audio(path.removeprefix("/audio/"))
            return
        self._error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/grokbot/run":
                report = run(repeat=int(body.get("repeat", 2)), tag=body.get("tag"))
                self._json(report.to_dict())
                return
            if path == "/api/grokbot/ask":
                self._json(ask(str(body.get("message", ""))))
                return
            if path == "/api/meetings":
                from backend.src.dashboard.meetings import load_meetings

                self._json({"meetings": load_meetings()})
                return
            if path == "/api/transcribe":
                self._json(transcribe_scenario(body["scenario_id"]))
                return
            if path == "/api/assess":
                self._json(assess_cues(body["scenario_id"], body.get("cues", [])))
                return
            if path == "/api/cursor/propose":
                self._json(propose().to_dict())
                return
            if path == "/api/cursor/apply":
                applied = apply_proposal()
                self._json({"applied": applied, "pristine": rules_are_pristine()})
                return
            if path == "/api/loop":
                self._json(close_loop(repeat=int(body.get("repeat", 2)), tag=body.get("tag")))
                return
            if path == "/api/reset":
                reset_rules()
                self._json({"pristine": rules_are_pristine()})
                return
        except KeyError as exc:
            self._error(400, f"Missing {exc.args[0]}")
            return
        except (FileNotFoundError, LookupError, ValueError, json.JSONDecodeError) as exc:
            self._error(400, str(exc))
            return
        self._error(404, "Not found")

    def log_message(self, fmt: str, *args) -> None:
        print("api", fmt % args)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def _json(self, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _bytes(self, payload: bytes, content_type: str, cache: bool = True) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _audio(self, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+\.wav", name):
            self._error(404, "Not found")
            return
        path = (AUDIO / name).resolve()
        if path.parent != AUDIO.resolve() or not path.is_file():
            self._error(404, "Not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None:
                self._error(416, "Bad range")
                return
            left, right = match.groups()
            if left == "" and right == "":
                self._error(416, "Bad range")
                return
            if left == "":
                length = int(right)
                start = max(0, size - length)
            else:
                start = int(left)
                end = int(right) if right else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206
        length = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            payload = handle.read(length)
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(payload)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self.wfile.write(payload)


def serve(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CaptionFix API  http://127.0.0.1:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def serve_in_thread(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    bound = server.server_address[1]
    return server, bound
