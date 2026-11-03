"""Cursor loop: regression tests and patches for Grokbot failures."""

from captionfix.cursor.propose import (
    apply_proposal,
    close_loop,
    format_loop,
    format_proposal,
    load_proposal,
    propose,
    reset_rules,
    rules_are_pristine,
)

__all__ = [
    "apply_proposal",
    "close_loop",
    "format_loop",
    "format_proposal",
    "load_proposal",
    "propose",
    "reset_rules",
    "rules_are_pristine",
]
