from casmi import degraded
from casmi.submit import FILLER_SMILES, format_guesses


def test_report_counts_fallbacks() -> None:
    degraded.reset()
    assert degraded.report() == "no fallback fired"
    degraded.note("x", 2)
    degraded.note("x")
    degraded.note("y", 0)
    assert degraded.events() == {"x": 3} and degraded.report() == "DEGRADED: x × 3"
    degraded.reset()


def test_filler_guess_is_recorded() -> None:
    degraded.reset()
    assert format_guesses([]) == FILLER_SMILES
    assert degraded.events() == {"molecule given the filler guess": 1}
    degraded.reset()
