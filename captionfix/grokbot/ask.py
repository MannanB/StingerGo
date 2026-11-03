"""Open the Grok Bot app. This does not answer as the bot."""

from __future__ import annotations

import shutil
import subprocess


def ask(message: str) -> dict:
    opened = _open_app()
    text = " ".join(message.strip().split())
    if opened:
        reply = "Grok Bot is open. Tell it what you need in that app."
    else:
        reply = "Grok Bot is installed, but it did not come to the front. Open Grok Bot yourself."
    if text:
        reply += "\n\nThis box does not talk to the app. Paste this there:\n" + text
    return {"reply": reply, "report": None, "loop": None, "meetings": None, "opened": opened}


def _open_app() -> bool:
    opener = shutil.which("open")
    if opener is None:
        return False
    try:
        subprocess.run([opener, "-a", "Grok Bot"], check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return True
