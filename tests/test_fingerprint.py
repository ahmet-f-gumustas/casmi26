import numpy as np
import pandas as pd
import pytest

from casmi.channels.fingerprint import LAYOUTS, descriptor, descriptors, fingerprint_channel, score


def test_descriptor_layouts() -> None:
    full = descriptor("c1ccccc1O", "ap6930")
    assert full.shape == (LAYOUTS["ap6930"],) and set(np.unique(full)) <= {0, 1}
    bits = np.array([0, 5, 10406])
    assert descriptor("c1ccccc1O", "ap6930", bits).tolist() == full[bits].tolist()
    ours = descriptor("c1ccccc1O", "casmi6311")
    assert ours.shape == (6311,) and ours.sum() > 0
    assert descriptor("not a smiles", "casmi6311") is None
    with pytest.raises(ValueError):
        descriptor("C", "nope")


def test_score_is_shifted_dot_product() -> None:
    packed = descriptors(["CCO", "c1ccccc1O", "xx"], "casmi6311")
    assert packed[2] is None
    z = np.zeros(6311, np.float32)
    on = np.unpackbits(packed[1], count=6311).astype(bool)
    z[on] = 1.0
    s = score(z, packed, 6311)
    assert s[1] > s[0] > 0 and s[2] == 0  # phenol matches z best; the lowest valid score is 1e-6
    assert s[0] == pytest.approx(1e-6)


def test_fingerprint_channel_scores_window_candidates() -> None:
    smiles_of = {"A": "CCO", "B": "c1ccccc1O"}
    packed = descriptors(["c1ccccc1O"], "casmi6311")[0]
    z = np.where(np.unpackbits(packed, count=6311).astype(bool), 1.0, -1.0).astype(np.float32)
    cache: dict = {}
    out = fingerprint_channel({"m": z}, {"m": ["A", "B"], "n": ["A"]}, smiles_of, "casmi6311", cache=cache)
    assert set(out) == {"m"} and out["m"]["B"] > out["m"]["A"]
    assert set(cache) == {"A", "B"}


def test_fpnet_roundtrip_and_bundle(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from casmi.fpmodel import APBundle, FPNet, merge_peaks, prep_peaks_ap

    net = FPNet(16, d=32, layers=1, heads=4)
    for name in ("fp_single_x.pt", "fp_merged_x.pt"):
        torch.save({"model": net.state_dict(), "nbits": 16, "d": 32, "layers": 1}, tmp_path / name)
    bundle = APBundle.load(sorted(tmp_path.glob("fp_*.pt")), bits=np.arange(16), device="cpu")
    assert len(bundle.single) == 1 and len(bundle.merged) == 1
    spectra = pd.DataFrame({"ms2_mzs": [np.array([50.0, 80.0]), np.array([60.0])],
                            "ms2_normalized_intensities": [np.array([1.0, 0.5]), np.array([1.0])],
                            "precursor_mz": [100.0, 100.0], "adduct": ["[M+H]+", "[M+H]+"],
                            "instrument_type": ["timsTOF", None], "collision_energy_ev": [[20.0], None],
                            "ionization_mode": ["positive", "positive"]})
    z = bundle.logits(spectra)
    assert z.shape == (16,) and np.isfinite(z).all()
    mz, it = prep_peaks_ap(np.array([10.0, 20.0, 200.0]), np.array([1.0, 0.0001, 1.0]), 100.0)
    assert mz.tolist() == [10.0] and it.tolist() == [1.0]  # above precursor + 1.5 and below the floor dropped
    mz, it = merge_peaks([np.array([50.0, 60.0]), np.array([50.001])], [np.array([1.0, 0.5]), np.array([2.0])])
    assert mz.tolist() == pytest.approx([50.001, 60.0]) and it.tolist() == [1.0, 0.5]  # tie keeps the later peak
    with pytest.raises(ValueError):
        APBundle.load(sorted(tmp_path.glob("fp_*.pt")), bits=np.arange(8), device="cpu")


def test_specformer_bundle_and_peak_prep(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from casmi.fpmodel import CasmiBundle, SpecFormer, prep_peaks

    mz, it = prep_peaks(np.array([100.004, 100.001, 50.0, 300.0]), np.array([0.2, 0.5, 1.0, 0.0]))
    assert mz.tolist() == [50.0, 100.0] and it.tolist() == pytest.approx([1.0, 0.5 ** 0.5])
    mz, _ = prep_peaks(np.arange(1.0, 400.0), np.arange(1.0, 400.0), max_peaks=256)
    assert len(mz) == 256 and mz[0] == 144.0  # the 256 strongest peaks, in m/z order

    cfg = {"nbits": 16, "d": 32, "layers": 1, "heads": 4}
    net = SpecFormer(cfg["nbits"], d=32, layers=1, heads=4)
    torch.save({"model": net.state_dict(), "config": cfg, "step": 1}, tmp_path / "fp_casmi.pt")
    bundle = CasmiBundle.load([tmp_path / "fp_casmi.pt"], device="cpu")
    spectra = pd.DataFrame({"ms2_mzs": [np.array([50.0, 80.0]), np.array([])],
                            "ms2_normalized_intensities": [np.array([1.0, 0.5]), np.array([])],
                            "precursor_mz": [100.0, 100.0], "adduct": ["[M+H]+", "[M+Na]+"],
                            "instrument_type": ["timsTOF", "timsTOF"], "collision_energy_ev": [None, [20.0]],
                            "ionization_mode": ["positive", "positive"]})
    z = bundle.logits(spectra)  # the empty spectrum is skipped; unknown CE is fed as NaN
    assert z.shape == (16,) and np.isfinite(z).all()
    assert bundle.logits(spectra.iloc[1:]) is None
