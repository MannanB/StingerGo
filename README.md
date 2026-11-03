# CaptionFix CI

Grokbot scores Graption, the BDBI live-captioning project cloned in `graption/`.

Graption anchors each ElevenLabs caption to one face. When two people talk, the other speaker is missing from the caption. Grokbot replays those scenes, the API scores them, and Cursor writes a regression plus a patch against `graption/backend/src/caption/attribute.py`.

Meetings the phone uploads are stored as-is. The dashboard does not fill in demo speakers or fake tasks.

## Run

From this directory:

```bash
python3 -m captionfix grokbot run
python3 -m captionfix loop
python3 -m captionfix api
```

The API listens on [http://127.0.0.1:8765](http://127.0.0.1:8765).

`grokbot run` exits 1 while the suite is failing, which is what you want in CI. `loop` applies the proposed repairs and exits 0 when the rerun and the new regression tests pass. `reset` puts the failing captioner back:

```bash
python3 -m captionfix reset
```

## What the suite is looking for

| Fault | What breaks | Where it is fixed |
| --- | --- | --- |
| Overlap | The quieter speaker disappears, and equal-level voices cancel each other | `resolve_overlap` |
| Accent | Indian English and Scottish tokens are rewritten through a mishear table | `normalize_accent` |
| Noise | Quiet words are dropped and the noise bed is captioned as words | `denoise` |

`clean_briefing` is the control. It passes before any patch.

## API

- `POST /api/transcribe` with `{"scenario_id": "overlap_cafe"}` runs the captioner and returns the score.
- `POST /api/assess` with `{"scenario_id": "...", "cues": [{"speaker", "text", "start", "end"}]}` scores captions you supply.
- `POST /api/grokbot/run` runs the suite.
- `POST /api/loop` runs Grokbot, writes tests and diffs, applies them, and runs Grokbot again.
- `POST /api/reset` restores the failing rules.
- `GET /api/state` is the lab snapshot the page renders.

Generated runs, diffs, and regression tests land in `runs/`, `patches/proposed/`, and `tests/regressions/`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
