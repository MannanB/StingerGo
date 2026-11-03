"""Repository locations shared by Grokbot, the API, and the Cursor loop."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = REPO / "scenarios"
RUNS = REPO / "runs"
PATCHES = REPO / "patches" / "proposed"
REGRESSIONS = REPO / "tests" / "regressions"
GRAPTION_ROOT = REPO / "graption"
RULES = GRAPTION_ROOT / "backend" / "src" / "caption" / "attribute.py"
PRISTINE = GRAPTION_ROOT / "backend" / "src" / "caption" / "attribute_pristine.py"
AUDIO = REPO / "audio_cache"
LATEST = RUNS / "latest.json"
PROPOSAL = PATCHES / "proposal.json"
