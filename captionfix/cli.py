"""CaptionFix command line."""

from __future__ import annotations

import argparse

from captionfix.api.server import serve
from captionfix.cursor.propose import (
    apply_proposal,
    close_loop,
    format_loop,
    format_proposal,
    propose,
    reset_rules,
)
from captionfix.grokbot.ask import ask
from captionfix.grokbot.runner import format_report, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captionfix",
        description="CaptionFix CI — Grokbot finds caption failures, the API scores them, Cursor proposes the patch.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    grokbot = commands.add_parser("grokbot", help="Run repeatable caption scenarios")
    grokbot_commands = grokbot.add_subparsers(dest="grokbot_command", required=True)
    grokbot_run = grokbot_commands.add_parser("run", help="Run the scenario suite")
    grokbot_run.add_argument("--repeat", type=int, default=1)
    grokbot_run.add_argument("--fresh", action="store_true")
    grokbot_run.add_argument("--tag", default=None)
    grokbot_run.set_defaults(func=_grokbot_run)
    grokbot_ask = grokbot_commands.add_parser("ask", help="Tell Grokbot what to run")
    grokbot_ask.add_argument("message", nargs="+")
    grokbot_ask.set_defaults(func=_grokbot_ask)

    api = commands.add_parser("api", help="Serve transcription and assessment")
    api.add_argument("--port", type=int, default=8765)
    api.set_defaults(func=_api)

    cursor = commands.add_parser("cursor", help="Write regression tests and patches")
    cursor_commands = cursor.add_subparsers(dest="cursor_command", required=True)
    cursor_propose = cursor_commands.add_parser("propose", help="Propose from the latest Grokbot run")
    cursor_propose.set_defaults(func=_cursor_propose)
    cursor_apply = cursor_commands.add_parser("apply", help="Apply the current proposal")
    cursor_apply.set_defaults(func=_cursor_apply)

    loop = commands.add_parser("loop", help="Run Grokbot, patch, and run Grokbot again")
    loop.add_argument("--repeat", type=int, default=2)
    loop.add_argument("--tag", default=None)
    loop.set_defaults(func=_loop)

    reset = commands.add_parser("reset", help="Restore the failing captioner")
    reset.set_defaults(func=_reset)

    args = parser.parse_args(argv)
    return args.func(args)


def _grokbot_ask(args: argparse.Namespace) -> int:
    print(ask(" ".join(args.message))["reply"])
    return 0


def _grokbot_run(args: argparse.Namespace) -> int:
    report = run(repeat=args.repeat, tag=args.tag, refresh=args.fresh)
    print(format_report(report))
    return 0 if report.failed_count == 0 else 1


def _api(args: argparse.Namespace) -> int:
    serve(args.port)
    return 0


def _cursor_propose(_args: argparse.Namespace) -> int:
    proposal = propose()
    print(format_proposal(proposal))
    return 0 if proposal.findings else 0


def _cursor_apply(_args: argparse.Namespace) -> int:
    applied = apply_proposal()
    if applied:
        print("Applied " + ", ".join(applied))
    else:
        print("Nothing to apply.")
    return 0


def _loop(args: argparse.Namespace) -> int:
    payload = close_loop(repeat=args.repeat, tag=args.tag)
    print(format_loop(payload))
    regressions_ok = all(item["passed"] for item in payload["regressions"])
    return 0 if payload["after"]["failed"] == 0 and regressions_ok else 1


def _reset(_args: argparse.Namespace) -> int:
    reset_rules()
    print("Restored Graption's one-face caption rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
