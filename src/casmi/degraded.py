"""Fallbacks that fired during a run.

Code that keeps going without some input (a molecule without fingerprint logits, an unparsable
candidate, a model on the CPU, a molecule that gets the filler guess) records it here with ``note``.
Nothing of this goes into the submission; the notebook prints ``report()`` as its last line, a
``DEGRADED`` warning with the counts when anything fired. A missing wheel, dataset or model stops the
notebook instead (``find`` / ``ensure`` / the model loaders raise).
"""

from __future__ import annotations

_EVENTS: dict[str, int] = {}


def note(kind: str, n: int = 1) -> None:
    """Count ``n`` occurrences of the fallback ``kind`` (n ≤ 0 is ignored)."""
    if n > 0:
        _EVENTS[kind] = _EVENTS.get(kind, 0) + int(n)


def events() -> dict[str, int]:
    return dict(_EVENTS)


def reset() -> None:
    _EVENTS.clear()


def report() -> str:
    """``DEGRADED: kind × n; ...`` when a fallback fired, else ``no fallback fired``."""
    if not _EVENTS:
        return "no fallback fired"
    return "DEGRADED: " + "; ".join(f"{k} × {n}" for k, n in sorted(_EVENTS.items()))
