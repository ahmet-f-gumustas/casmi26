import numpy as np
import pandas as pd

from casmi import degraded
from casmi.forward import score as fs


class Echo:
    """Predicts the same peaks for every SMILES except 'bad' (None); 'good' gets the query's own peaks."""

    def __init__(self, peaks):
        self.peaks, self.calls = peaks, []

    def predict(self, smiles, adduct, energies, instrument="QTOF", top=100):
        self.calls.append((tuple(smiles), adduct, tuple(energies), instrument))
        out = []
        for s in smiles:
            if s == "bad":
                out.append(None)
                continue
            mz, it = self.peaks if s == "good" else (np.array([51.0, 77.0]), np.array([1.0, 1.0], np.float32))
            out.append([(mz.astype(np.float64), it.astype(np.float32)) for _ in energies])
        return out


def _queries():
    mz, it = np.array([80.0, 105.0, 122.0]), np.array([0.3, 1.0, 0.5])
    rows = [("m1", "[M+H]+", np.array([20.0, 40.0]), "timsTOF"), ("m1", "[M-H]-", np.array([20.0]), "timsTOF"),
            ("m2", "[M+NH4]+", np.array([40.0]), "timsTOF")]
    q = pd.DataFrame([{"molecule_id": m, "adduct": a, "collision_energy_ev": e, "instrument_type": i,
                       "ms2_mzs": mz, "ms2_normalized_intensities": it, "precursor_mz": 200.0}
                      for m, a, e, i in rows])
    return q, (mz, it)


def test_instrument_classes():
    assert fs.instrument_of("timsTOF") == "QTOF"
    assert fs.instrument_of("LC-ESI-QFT") == fs.instrument_of("Orbitrap") == fs.instrument_of("Hybrid FT") == \
        "Orbitrap"
    assert fs.instrument_of(None) == "QTOF"


def test_scores_only_supported_spectra_and_candidates():
    q, peaks = _queries()
    model = Echo(peaks)
    smiles = {"a": "good", "b": "other", "c": "bad"}
    scores, info = fs.forward_scores(model, q, {"m1": ["a", "b", "c"], "m2": ["a"]}, smiles)
    assert scores[("m1", "a")] > 0.99 > scores[("m1", "b")]
    assert ("m1", "c") not in scores and ("m2", "a") not in scores  # unhandled candidate; [M+NH4]+ only
    assert info["unscorable"] == 1 and info["failed"] == 2  # c at 20 and 40 eV
    assert {c[1] for c in model.calls} == {"[M+H]+"} and {c[2] for c in model.calls} == {(20.0,), (40.0,)}


def test_cache_and_budget():
    q, peaks = _queries()
    model, cache = Echo(peaks), {}
    fs.forward_scores(model, q, {"m1": ["a"]}, {"a": "good"}, cache=cache)
    n = len(model.calls)
    fs.forward_scores(model, q, {"m1": ["a"]}, {"a": "good"}, cache=cache)
    assert len(model.calls) == n  # all from the cache
    degraded.reset()
    scores, _ = fs.forward_scores(Echo(peaks), q, {"m1": ["a"]}, {"a": "good"}, budget_s=-1.0)
    assert not scores and "forward model out of time" in degraded.report()
    degraded.reset()


def test_rrf_reorders_only_the_head():
    m = np.array(["x"] * 5)
    base = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    fw = np.array([0.1, np.nan, 0.9, 0.5, 0.99])  # last row is outside the head (top 3)
    s = fs.rrf(m, base, [fw], k=1, top=3)
    order = np.argsort(-s, kind="stable")
    assert order[:3].tolist() == [0, 2, 1] and order[3:].tolist() == [3, 4]  # 2 passes 1, the nan row


def test_top_keys_and_feature_columns():
    m = np.array(["x", "x", "y", "x"])
    keys = np.array(["a", "b", "c", "d"])
    assert fs.top_keys(m, keys, np.array([0.1, 0.5, 0.2, 0.3]), k=2) == {"x": ["b", "d"], "y": ["c"]}
    f = fs.feature_columns(m, np.array([0.2, np.nan, 0.4, 0.5]))
    assert f["fw_rank"].tolist()[0] == 2 and np.isnan(f["fw_rank"].iloc[1])
    assert np.allclose(f["fw_minus_max"].to_numpy()[[0, 2, 3]], [-0.3, 0.0, 0.0])


def test_rrf_two_models_agreeing_move_a_row_up():
    m = np.array(["x"] * 3)
    base = np.array([0.9, 0.8, 0.7])
    one, two = np.array([0.1, 0.2, 0.9]), np.array([0.3, 0.1, 0.8])
    assert np.argmax(fs.rrf(m, base, [one], k=1)) == 0
    assert np.argmax(fs.rrf(m, base, [one, two], k=1)) == 2
