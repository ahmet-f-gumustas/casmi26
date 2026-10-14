"""EXP-000: check ``casmi.metric`` against the official Kaggle scorer.

1. RDKit version must be exactly the pinned one.
2. Key agreement on N random train structures: official key vs ours, vs a naive
   ``InChIKey[:14]``, vs our first scaffold (tautomer canonicalization without
   stripping stereo, raw-molecule fallback) and vs the train ``inchikey14`` column.
3. Fuzzed submissions (stereo-free / random-rooted SMILES of the answer, same-formula
   decoys, invalid strings, duplicates, empty slots) scored per molecule by the
   official ``score()`` and by ``casmi.metric.evaluate``.

Needs the official notebook in external/official_metric:
    kaggle kernels pull metric/casmi-mean-reciprocal-rank -p external/official_metric

    python scripts/check_metric.py [--n 50000] [--fuzz 3000] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi.config import RDKIT_VERSION, REPO_ROOT
from casmi.data import read_frame
from casmi.metric import evaluate, load_official_scorer, scorer_key
from casmi.report import md_table

OFFICIAL = load_official_scorer()
_TAUT = rdMolStandardize.TautomerEnumerator()


def naive_key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    key = Chem.MolToInchiKey(mol) if mol is not None else ""
    return key[:14] or None


def scaffold_key(smiles: str) -> str | None:
    """The first version of ``scorer_key`` (no stereo strip, raw-molecule fallback)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _TAUT.Canonicalize(mol)
    except Exception:  # noqa: BLE001
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] or None


def all_keys(smiles: str) -> tuple[str | None, str | None, str | None, str | None, float]:
    t0 = time.perf_counter()
    official = OFFICIAL.canonicalize_smiles_to_ik14(smiles)
    dt = time.perf_counter() - t0
    return official, scorer_key(smiles), naive_key(smiles), scaffold_key(smiles), dt


def score_one(case: tuple[str, str, list[str]]) -> tuple[str, float, float]:
    mol_id, answer, guesses = case
    solution = pd.DataFrame({"molecule_id": [mol_id], "smiles": [answer], "Usage": ["Public"]})
    submission = pd.DataFrame({"molecule_id": [mol_id], "smiles": [";".join(guesses)]})
    official = OFFICIAL.score(solution, submission, "molecule_id")
    ours = evaluate({mol_id: guesses}, {mol_id: answer})["mrr@25"]
    return mol_id, official, ours


def key_section(sample: pd.DataFrame, workers: int) -> tuple[list[str], pd.DataFrame]:
    with Pool(workers) as pool:
        res = pool.map(all_keys, sample["normalized_smiles"].tolist(), chunksize=200)
    keys = pd.DataFrame(res, columns=["official", "ours", "naive", "scaffold", "seconds"], index=sample.index)
    keys["train_ik14"] = sample["inchikey14"]
    rows = []
    for col in ["ours", "naive", "scaffold", "train_ik14"]:
        diff = keys[col] != keys["official"]
        rows.append({"key": col, "mismatch": int(diff.sum()), "mismatch_frac": float(diff.mean())})
    lines = [
        f"## Key agreement on {len(sample):,} random train structures",
        f"- official key unparseable: {keys['official'].isna().sum()}; canonicalization time per structure: "
        f"median {keys['seconds'].median() * 1e3:.1f} ms, p99 {keys['seconds'].quantile(0.99) * 1e3:.0f} ms, "
        f"max {keys['seconds'].max():.1f} s",
        md_table(pd.DataFrame(rows)),
    ]
    examples = keys[keys["scaffold"] != keys["official"]].head(5)
    if len(examples):
        lines += ["Scaffold vs official examples:",
                  md_table(sample.loc[examples.index, ["normalized_smiles"]].join(examples[["official", "scaffold"]]))]
    return lines, keys


def fuzz_cases(sample: pd.DataFrame, n: int, rng: random.Random) -> list[tuple[str, str, list[str]]]:
    smiles = sample["normalized_smiles"].tolist()
    formula_of = dict(zip(sample["normalized_smiles"], sample["molecular_formula"]))
    by_formula = sample.groupby("molecular_formula")["normalized_smiles"].apply(list).to_dict()
    cases = []
    for i in range(n):
        answer = rng.choice(smiles)
        mol = Chem.MolFromSmiles(answer)
        decoys = [s for s in by_formula.get(formula_of[answer], []) if s != answer]
        pool = decoys + rng.sample(smiles, 10) + ["xx", "C1CC", "", " ", "c1ccccc1;"]
        guesses = [rng.choice(pool) for _ in range(rng.randint(0, 24))]
        guesses += [rng.choice(guesses)] if guesses else []  # a duplicate
        if rng.random() < 0.7:  # the answer, written differently
            variant = rng.choice([
                answer,
                Chem.MolToSmiles(mol, isomericSmiles=False),
                Chem.MolToSmiles(mol, doRandom=True),
                f"  {answer}  ",
            ])
            guesses.insert(rng.randint(0, len(guesses)), variant)
        guesses = [g for g in guesses if g is not None]
        # the scorer splits on ';', so keep at most 25 non-empty pieces
        while len([p for g in guesses for p in g.split(";") if p.strip()]) > 25:
            guesses.pop()
        if not [p for g in guesses for p in g.split(";") if p.strip()]:
            guesses = ["C"]
        cases.append((f"m_{i:05d}", answer, guesses))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--fuzz", type=int, default=3_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "METRIC_CHECK.md")
    args = ap.parse_args()
    if OFFICIAL is None:
        raise SystemExit("official metric notebook missing — see the module docstring")

    lines = [
        "# EXP-000 — metric check against the official scorer",
        f"- rdkit.__version__ = {rdkit.__version__} (pinned {RDKIT_VERSION}): "
        f"{'OK' if rdkit.__version__ == RDKIT_VERSION else 'MISMATCH'}",
    ]
    train = read_frame("train", ["normalized_smiles", "inchikey14", "molecular_formula"])
    structures = train.drop_duplicates("normalized_smiles")
    sample = structures.sample(n=min(args.n, len(structures)), random_state=args.seed)

    section, _ = key_section(sample, args.workers)
    lines += section

    rdBase.SeedRandomNumberGenerator(args.seed)  # for doRandom SMILES
    cases = fuzz_cases(sample, args.fuzz, random.Random(args.seed))
    with Pool(args.workers) as pool:
        scored = pool.map(score_one, cases, chunksize=20)
    df = pd.DataFrame(scored, columns=["molecule_id", "official", "ours"])
    diff = ~np.isclose(df["official"], df["ours"], rtol=0, atol=1e-12)
    lines += [
        f"## Fuzzed submissions ({len(df):,} molecules)",
        f"- official MRR = {df['official'].mean():.6f}, ours = {df['ours'].mean():.6f}",
        f"- per-molecule disagreements: {int(diff.sum())}",
    ]
    if diff.any():
        bad = [c for c, d in zip(cases, diff) if d][:5]
        lines.append(md_table(pd.DataFrame(bad, columns=["molecule_id", "answer", "guesses"]).astype(str)))

    report = "\n\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
