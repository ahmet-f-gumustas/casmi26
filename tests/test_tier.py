import pytest

from casmi.tier import TierConfig, prior, pub_candidates, slot_merge, tier_merge, union_scores


def test_union_scores_rank_normalise_over_the_union() -> None:
    keys = ["A", "B", "P"]
    s = union_scores(keys, {"analog": {"A": 0.9, "B": 0.1, "P": 0.5}, "fingerprint": {"P": 2.0}},
                     {"analog": 1.0, "fingerprint": 1.0})
    assert s["A"] == pytest.approx(1.0) and s["B"] == pytest.approx(1 / 3)
    assert s["P"] == pytest.approx(2 / 3 + 1.0)


def test_prior() -> None:
    cfg = TierConfig(beta=1.0, sid_cap=1000.0)
    assert prior(0, cfg) == 0.0 and prior(5000, cfg) == 1.0
    assert 0 < prior(10, cfg) < 1
    assert prior(0, TierConfig(beta=0.0)) == 1.0


def test_tier_merge_gate_and_slots() -> None:
    main = [f"M{i}" for i in range(30)]
    common = {m: 1.0 - i / 100 for i, m in enumerate(main)}  # M0 1.00, M2 0.98, ...
    cfg = TierConfig(n_slots=3, gate_k=3)
    strong = {"P1": 0.99, "P2": 0.5, "P3": 0.4, "P4": 0.1}
    assert tier_merge(main, strong, common, cfg) == main[:22] + ["P1", "P2", "P3"]
    two = {"P1": 0.99, "P2": 0.5}  # fewer PubChem candidates than slots: the main list keeps the rest
    assert tier_merge(main, two, common, cfg) == main[:23] + ["P1", "P2"]
    weak = {"P1": 0.97}  # below the 3rd main candidate (0.98)
    assert tier_merge(main, weak, common, cfg) == main[:25]
    assert tier_merge(main, {}, common, cfg) == main[:25]
    assert tier_merge(main[:2], two, common, cfg) == main[:2] + ["P1", "P2"]  # short list: no gate


def test_popularity_order_and_threshold() -> None:
    common = {"P1": 0.9, "P2": 0.4, "P3": 0.8, "P4": 0.0}
    sid = {"P1": 10, "P2": 1000, "P3": 500, "P4": 9999}
    cfg = TierConfig(order="popularity", min_common=0.5)
    assert pub_candidates(list(common), common, sid, cfg) == [("P3", 0.8), ("P1", 0.9)]
    main = [f"M{i}" for i in range(30)]
    mc = {m: 1.0 - i / 100 for i, m in enumerate(main)}
    out = tier_merge(main, pub_candidates(list(common), common, sid, TierConfig(n_slots=2, gate_k=13, order="popularity",
                                                                               min_common=0.5)), mc, TierConfig(
        n_slots=2, gate_k=13))
    assert out == main[:23] + ["P3", "P1"]  # the gate compares the best union score (0.9 > 0.88)


def test_slot_merge_tail_and_merit() -> None:
    main = [f"M{i}" for i in range(30)]
    ms = {m: 1.0 - i / 100 for i, m in enumerate(main)}  # M0 1.00 ... M24 0.76
    extra = {"A": 0.995, "B": 0.5, "C": 0.9}
    assert slot_merge(main, ms, extra, n_slots=5, gate_k=25, placement="tail") == main[:23] + ["A", "C"]
    merit = slot_merge(main, ms, extra, n_slots=5, gate_k=25, placement="merit")
    assert merit[:3] == ["M0", "A", "M1"] and merit.index("C") == 12 and len(merit) == 25 and "M22" in merit and "M23" not in merit
    assert slot_merge(main, ms, extra, n_slots=1, gate_k=1, placement="merit") == main[:25]  # gate: beat M0
    assert slot_merge(main, ms, {}, n_slots=5, gate_k=5) == main[:25]
