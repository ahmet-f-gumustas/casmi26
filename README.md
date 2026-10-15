# CASMI 2026: Molecule ID from MS/MS Spectra

My work on the Kaggle competition
**[Enveda CASMI 2026: Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)**.

| | |
|---|---|
| **Competition** | Enveda × CASMI 2026 (Critical Assessment of Small Molecule Identification) |
| **Type** | Code competition: offline notebook, no internet, ≤ 9 h runtime |
| **Deadline** | 14 December 2026 |
| **Metric** | MRR@25 per molecule |

---

## Problem

Given one or more LC-MS/MS spectra of an unknown molecule (several collision
energies and adducts), predict up to **25 SMILES**, best guess first.

| | |
|---|---|
| Input | MS/MS spectra per molecule (`ms2_mzs`, `ms2_normalized_intensities`, precursor m/z, adduct, CE, instrument) |
| Output | ≤ 25 ranked SMILES per molecule |
| Metric | **MRR@25**, averaged over molecules (not spectra) |
| Correct hit | Tautomer-canonical **InChIKey14** match under RDKit **2026.03.3**. Stereo and tautomers do not matter |
| Train | ~2.54M spectra, ~275.8k unique InChIKey14 (`inchikey14` is the group key; there is no `molecule_id`) |
| Test | ~400 molecules, ~1,500 unpublished spectra (1–16 per molecule, median 3), all Bruker timsTOF, 10 adducts, neutral mass 157–1,159 Da |

### Novelty classes

| Class | Share* | Meaning | Reachable by retrieval? |
|---|---|---|---|
| 1 | ~16 % | Reference spectrum exists in a public library | Yes |
| 2 | ~45 % | Structure is in PubChem / COCONUT, no spectrum | Yes |
| 3 | ~39 % | Novel natural product | **No** |

\* Class definitions come from the host. The shares are community estimates
from leaderboard probing. Pure retrieval tops out around **0.61** MRR.

---

## Setup

```bash
git clone https://github.com/ahmet-f-gumustas/casmi26.git && cd casmi26

# Python 3.11 env (uv shown; conda works too)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # core + pytest/kaggle/ipykernel
uv pip install -e ".[ml]"         # torch, matchms, ms_entropy

pytest -q
```

The official scorer is the Kaggle notebook [`metric/casmi-mean-reciprocal-rank`](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank).
`casmi.metric` mirrors it exactly. Pull it once to enable the comparison test and script:

```bash
kaggle kernels pull metric/casmi-mean-reciprocal-rank -p external/official_metric
```

> **Pin RDKit.** Tautomer canonicalization changes between RDKit releases.
> The scorer strips stereo, canonicalizes the tautomer, then takes InChIKey14. A naive
> `InChIKey[:14]` (and the train `inchikey14` column) disagrees with it on ~1.7 % of train structures.
> On Kaggle, install the wheel from the `metric/rdkit-2026-3-3-wheel` dataset.

### Data

Kaggle CLI 2.x reads its token from `~/.kaggle/access_token` (the old
`kaggle.json` returns 401). Accept the competition rules on the website first.

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data
unzip -q data/*.zip -d data
```

> The visible `test.parquet` is a placeholder: its 400 molecules are enveda-180 training spectra. The hidden
> test replaces it at re-run, so never tune on it. `enveda-np-examples` is the closest public proxy.

Paths come from `casmi.config`. On Kaggle they point to `/kaggle/input/...` and
`/kaggle/working`. Locally you can override them with `CASMI_DATA`,
`CASMI_EXTERNAL` and `CASMI_WORK`. The defaults are `data/`, `external/` and
`outputs/`.

---

## Layout

```
casmi26/
├── data/            # competition data (git-ignored)
├── external/        # COCONUT download, candidate pool (external/pool/), official metric (git-ignored)
├── src/casmi/
│   ├── config.py    # paths (Kaggle vs. local), TOP_K, pinned RDKit version
│   ├── metric.py    # exact mirror of the official scorer: scorer_key, evaluate, dedup_by_key
│   ├── chem.py      # adduct table, neutral mass ↔ precursor m/z, ppm windows
│   ├── data.py      # column-wise parquet readers
│   ├── clean.py     # test-style cleaning rules: precursor+2 Da peak cut, base peak, mass verdicts
│   ├── structures.py # cached scorer key + exact mass per train SMILES
│   ├── submit.py    # submission.csv builder + validation (scorer's rejection rules)
│   ├── split.py     # molecule-level validation split, Class 1/2/3 simulation, leakage checks
│   ├── pipeline.py  # end-to-end ranking shared by validation and the Kaggle notebook
│   ├── harness.py   # load a split's queries / mask / pool, channel scores, MRR@25 per class
│   ├── ranker.py    # ranker v0 (weighted rank-normalised sum) and v1 (library gate / two-stage)
│   ├── ranker_v2.py # learned ranker: per-candidate features, bagged HistGradientBoosting fitted in place
│   ├── difficulty.py # nn_tanimoto / analog counts (blocked matmul), difficulty bins, profile tilt
│   ├── candidates.py # mass-sorted candidate pool, ppm windows with instrument offset, exclusions
│   ├── pubchem.py   # reader for the mass-sorted PubChem SMILES index, NP-likeness / scaffold filter
│   ├── tier.py      # PubChem tier (last N slots, gate, popularity prior / order); gated extra slots (slot_merge)
│   ├── adducts.py   # alternative-adduct hypotheses and their mass windows (recall probe 2b)
│   ├── fpmodel.py   # spectrum → fingerprint models: SpecFormer (ours), FPNet (prvsiyan checkpoints)
│   ├── channels/    # library (entropy similarity), analog (mass-shifted analogs), fragment (bond breaks),
│   │                # fingerprint (candidate f·z against the model's bit logits)
│   └── generate/    # Class 3: analog editing, de novo generation
├── validation/      # molecule-level holdout splits, class simulation
├── kaggle/          # generated submission notebook + kernel-metadata.json
├── scripts/         # EDA and one-off utilities
└── tests/
```

### Scripts

| Script | Output (local, git-ignored) |
|---|---|
| `scripts/check_adducts.py` | `notes/ADDUCTS.md`: adduct coverage, precursor ppm errors, mislabels; caches structure masses |
| `scripts/eda.py` | `notes/EDA.md`: train/test distributions, shift, isomer pool, submission format |
| `scripts/clean_train.py` | `outputs/train_clean.parquet` (test-style cleaning, ppm-verified adduct fixes), `outputs/train_rejected.parquet`, `notes/CLEAN.md` |
| `scripts/build_pool.py` | `external/pool/`: COCONUT ∪ train, deduped by scorer key, mass-sorted `masses.npy`, Morgan-2048 `fps.npy`, `meta.parquet` (~212 MB); `notes/POOL.md` |
| `scripts/fork_prvsiyan.py` | `outputs/kaggle/prvsiyan_v33_fork/`: prvsiyan's scored V33 cell (Apache 2.0), byte-for-byte, as a private kernel folder |
| `scripts/analog_difficulty.py` | `notes/ANALOG_DIFFICULTY_<prefix>.md`: Class 2 MRR by nn_tanimoto bin, NP-proxy profile, LB-implied target profile (`outputs/difficulty_profile.json`) |
| `scripts/build_pool.py --pubchem` | `external/pool_pubchem/`: pool + PubChem rows filtered by NP-likeness / COCONUT scaffold (≤ 3M); `notes/POOL_PUBCHEM.md` |
| `scripts/error_analysis.py` | `notes/ERROR_ANALYSIS.md`, `outputs/error_analysis_<prefix>.parquet`: Class 2 failures of the analog channel by cluster (near-isomer / retrieval / intermediate), NP subset |
| `scripts/build_fp_data.py` | `outputs/fp_data/`: library spectra as fingerprint-model inputs (m/z rounded to 0.01 Da, ≤ 256 peaks), casmi6311 pool fingerprints, ±10 ppm decoys, per-split-family training masks |
| `scripts/train_fp.py` | `outputs/fp_models/<name>/fp_casmi.pt`: SpecFormer (8 × 512 transformer, CLS) trained with BCE (rare-bit pos_weight) + soft-Tanimoto ranking loss over 32 decoys; `--np-weight` upweights natural products, `--eval-splits` scores every epoch (C1-protected) and keeps the best, `--patience` stops early |
| `scripts/recall_diagnosis.py` | `notes/DIAGNOSIS.md`: test columns, in-library share of Class 2 window candidates vs wrong analog top-1s, analog channel without self matches (alone and C1-protected) |
| `scripts/build_popularity.py` | `<pool>/popularity.parquet`: PubChem substance / PubMed record counts per scorer key (CC0 `prvsiyan/pubchem-popularity-counts`) |
| `scripts/check_metric.py` | `notes/METRIC_CHECK.md`: `casmi.metric` vs the official Kaggle scorer (EXP-000) |
| `validation/candidate_recall.py` | `notes/RECALL_<split>.md`: is the answer in its mass window, per class (C3 must be 0) |
| `validation/eval_analog.py` | `notes/EXP002_analog_*.md`: analog-channel grid (power, top-K, aggregation) per class; needs the Flash index in `outputs/analog_index` |
| `validation/eval_fragment.py` | `notes/EXP003_fragment_*.md`: fragmentation-channel MRR per class, time per candidate, agreement with the analog channel |
| `validation/eval_fp.py` | `notes/EXP_fp_<model>_<prefix>.md`: fingerprint channel alone (f·z, soft-Tanimoto), with the analog channel and C1-protected (library gate τ = 0.8 first), per class over the v2 or v3 seeds; saves fingerprint features |
| `validation/eval_pubchem_tier.py` | `notes/EXP_pubchem_tier_*.md`, `outputs/tier_v1.json`: PubChem tier (last N slots, gate, popularity prior) — cost on validation, recovery of simulated PubChem-only Class 2; `--grid aggressive` (N = 12, popularity order) → `outputs/tier_aggressive_<weights>.json` |
| `validation/eval_alt_adducts.py` | `notes/EXP_alt_adducts.md`, `outputs/alt_adducts_v1.json`: alternative-adduct slots (tail / merit, gate) — cost on validation, slot loss on the placeholder test, recovery of simulated mislabelled adducts |
| `validation/ranker_features.py` | `outputs/ranker_features/<split>.parquet`: per-(molecule, candidate) ranker-v2 features for the v2 seeds and the ranker splits r_seed10–14 |
| `validation/eval_ranker_v2.py` | `notes/EXP_ranker_v2*.md`, `outputs/ranker_v2*.json`, `outputs/ranker_train*/`: learned ranker vs rank average per class over the v2 seeds (training on r_seed10–14); `--splits v3` uses v3_seed0–2 / r3_seed10–14 |
| `validation/eval_pubchem.py` | `notes/EXP011_pubchem_*.md`: recall, candidates, analog C2 cost and the Class 2b (PubChem-only) break-even, old vs PubChem pool |
| `validation/eval_gate.py` | `notes/EXP010_gate_*.md`, `outputs/ranker_v1.json`: library gate / two-stage vs analog only per class over seeds; saves channel features per split |
| `validation/eval_ranker.py` | `notes/EXP004_ranker_*.md`, `outputs/ranker_v0.json`: ranker-v0 weights tuned on half A, tested on half B |
| `validation/seed_noise.py` | `notes/EXP005_seed_noise.md`: library / analog MRR@25 per class across split seeds, mean ± std |
| `validation/eval_library.py` | `notes/EXP001_library.md`: library-channel MRR@25 per class / source (needs `.[ml]`) |
| `scripts/make_kaggle_notebook.py` | `kaggle/submission/`: submission notebook with `casmi` embedded; `--local-check` runs it on the placeholder test |
| `scripts/kaggle_datasets.py` | stages and uploads the Kaggle datasets above |
| `validation/make_split.py` | `outputs/splits/{seedN,v2_seedN,v3_seedN,final}/`: `val_ids.parquet`, `library_mask.parquet`, `config.json`; `--difficulty` = split v2, `--decoys-from` = split v3 (+ `decoy_keys.parquet`) |

### Validation split

`validation/make_split.py` builds a seeded, deterministic holdout. Molecules are grouped by **scorer key**,
not by the train `inchikey14` column, so tautomers can't leak. The target is the hidden test as the host
describes it (timsTOF, the 10 test adducts, 157–1,159 Da), with the chemistry, adduct mix and CE mix of
`enveda-np-examples`:

- every eligible NP example (~245) is a validation molecule;
- enveda-180 is capped at `--max-drug-share` (default 50 % → ~490 molecules), drawn with mass-bin quotas
  matching the NP mass distribution and adduct / CE importance weights;
- `--final` writes a mask that keeps every spectrum in the library, for the submission run.
- `--difficulty` (split v2) draws Class 2 by nn_tanimoto bin (default 0 / 50 / 35 / 15 % for 0–0.3 / 0.3–0.5 / 0.5–0.7 / 0.7+), capping easy C2; analog C2 MRR 0.75 → 0.66, still about twice the LB-implied value.
- `--decoys-from v2_seedN` (split v3) also removes the spectra of a random 30 % of the non-validation library
  structures (81k structures, ~0.7M spectra). The pool is unchanged, so "no library spectra, not in COCONUT"
  no longer marks the held-out answer (70 % → 1.8 % of such ranker training rows), and Class 2 windows hold
  fewer structures with library spectra (60 % → 42 %).

| Class | Share | Library (`in_library`) | Candidate pool |
|---|---|---|---|
| C1 | 16 % | timsTOF spectra removed, other-library references kept | kept |
| C2 | 45 % | all spectra removed | kept |
| C3 | 39 % | all spectra removed | **removed** |

## Kaggle

The submission notebook is generated from this repo: `scripts/make_kaggle_notebook.py` embeds `src/casmi`
(base64 tar.gz), installs RDKit / ms_entropy from wheel datasets and finds its inputs by file name
(Kaggle mounts datasets under `/kaggle/input/datasets/<owner>/<slug>/` and runs Python 3.13).

| Dataset (private unless noted) | Content |
|---|---|
| `red6161/casmi26-pool` | candidate pool (`masses.npy`, `fps.npy`, `meta.parquet`, 212 MB) |
| `red6161/casmi26-library` | cleaned library `train_clean.parquet` (3.4 GB) |
| `red6161/casmi26-wheels` | ms_entropy 1.5.3 wheels, CPython 3.10–3.13 |
| `red6161/casmi26-pool-pubchem` | pool + 1.57M NP-like PubChem structures (2.28M, `fps.npz` compressed, 183 MB) |
| `red6161/casmi26-analog-index` | Flash entropy index of the cleaned library (2.8 GB, files flattened as `<mode>__<file>`) |
| `metric/rdkit-2026-3-3-wheel` (public) | RDKit 2026.3.3 wheels used by the official scorer |

```bash
python scripts/kaggle_datasets.py pool library index wheels  # first upload (--update -m msg for new versions)
python scripts/make_kaggle_notebook.py --local-check         # library-only notebook, run locally on the placeholder
python scripts/make_kaggle_notebook.py --mode combined [--pool pubchem]  # ranker config from outputs/ranker_v1.json
python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_fp_ap.json --fp ap  # + fingerprint channel, T4 GPU
python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_fp.json --fp casmi  # our fingerprint model
python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_fp.json --fp casmi --pool pubchem --tier outputs/tier_v1.json
python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_v2.json --fp casmi  # ranker v2, fitted in the notebook
python scripts/make_kaggle_notebook.py --mode combined --pool pubchem --tier outputs/tier_aggressive_analog.json  # probe 2a
python scripts/make_kaggle_notebook.py --mode combined --alt-adducts outputs/alt_adducts_v1.json  # probe 2b
kaggle kernels push -p kaggle/submission
kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra \
    -k red6161/casmi26-submission -v <version> -f submission.csv -m "<message>"
```

| Submission | Public LB | Val MRR@25 | Kaggle runtime / peak RSS |
|---|---|---|---|
| library-only v1 | 0.137 | 0.144 (seed0); 0.136 ± 0.007 (3 seeds) | 110 s / 9.4 GB (CPU) |
| ranker v0 = analog only (p3, K100, mean) | 0.285 | 0.492 (seed0); 0.479 ± 0.012 (3 seeds) | 266 s / 3.9 GB (CPU, 4 workers) |
| analog only + PubChem pool (2.28M) | 0.258 | 0.424 ± 0.009 (split v2) | 207 s / 5.0 GB (CPU, 4 workers) |
| analog + prvsiyan FP checkpoints (rank average, CC BY-NC-SA test) | 0.292 | — (trained on all of train) | 261 s / 4.8 GB (T4 GPU) |
| analog + our SpecFormer FP (rank average) | 0.262 | 0.443 ± 0.012 (split v2; C2 NP 0.567) | 244 s / 4.8 GB (T4 GPU; logits 6.5 s / 1213 spectra) |
| ranker v2 (learned, all features) + our FP | 0.275 | 0.526 ± 0.013 (split v2; shortcut-inflated, test-like 0.437) | 373 s / 13.3 GB (T4 GPU) |
| reference: prvsiyan V33 public notebook, unchanged private fork | 0.344 | — | 3879 s (T4 GPU) |

Validation tracks the LB for Class 1 but overestimates Class 2 about twofold. Neither analog difficulty
(nn_tanimoto does not predict analog-channel success; split v2 only brings val C2 from 0.75 to 0.66) nor pool
coverage (adding 1.57M NP-like PubChem isomers costs 0.027 on the LB, as predicted for no PubChem-only answers)
explains the gap. The reference notebook runs analog propagation, then a PubChem popularity prior, a forward-model
isomer check (GPU) and a gated PubChem tier in fixed slots; this pipeline has only the first step so far.

A second trap for learned rankers: the pool is COCONUT ∪ training structures, so in a simulation a candidate with
no library spectra and no COCONUT entry can only be the held-out answer — a shortcut that never occurs on the
test. `validation/eval_ranker_v2.py` drops it from training and decides on a test-like metric (C1 and the Class 2
answers that are in COCONUT).

A third one is the self match: a candidate with library spectra is its own best analog (Tanimoto 1). 81 % of the
analog channel's wrong Class 2 top-1s have library spectra, against 60 % of the wrong candidates in the window
(NP: 66 % vs 31 %). Dropping self matches trades Class 1 for Class 2 (C1 −0.23, C2 +0.04, total −0.02) and,
with Class 1 protected by the library gate, leaves the total unchanged. Split v3's decoys shrink the bias. On
split v3 the learned ranker still gains on the total, but on natural products only without its popularity and
source features (test-like +0.016 over analog only). Upweighting natural products threefold in fingerprint
training does not move Class 2 NP past the analog channel (+0.004 on v3, −0.030 on v2).

Since validation cannot measure recall, the leaderboard is now spent on recall probes against analog only
(0.285): a larger PubChem tier ordered by popularity (2a) and gated slots for alternative-adduct windows (2b).

### Quick usage

```python
from casmi.metric import evaluate, dedup_by_key
from casmi.chem import neutral_mass

neutral_mass(195.087652, "[M+H]+")          # -> 194.080376 (caffeine)
preds = {"mol1": dedup_by_key(candidate_smiles)}
evaluate(preds, {"mol1": true_smiles})       # {'mrr@25': ..., 'hits@1': ..., ...}
```

---

## Ground rules

1. **Deduplicate by InChIKey14, not SMILES.** Otherwise 25 tautomers of the same structure fill 24 slots with nothing new.
2. **Do not upload a CSV.** In this code competition you submit a notebook *version*. The downloadable `submission.csv` only covers the placeholder test.
3. **Keep adduct math in one place.** Every neutral-mass conversion goes through `casmi.chem.ADDUCTS`.
4. **Avoid leakage.** Never split spectra of the same InChIKey14 across train and validation.
5. **The candidate pool is mostly isomers.** COCONUT (Oct 2026) plus train structures give 714k structures after scorer-key dedup. On validation, a ±10 ppm window (+2 ppm timsTOF offset) returns a median of ~70 candidates (p90 ~200–250), and a median of 71 % of them share the answer's formula.
6. **Seed noise is about ±0.015 on total MRR@25** (measured std over 3 split seeds: 0.007–0.012; per class ±0.04 for C1, ±0.02 for C2). Smaller differences are not measurable.

---

## Roadmap

- [ ] **Baseline & harness.** EDA, exact metric check against the official scorer, reproduce a public pipeline, and a molecule-holdout validation that simulates Class 1/2/3.
- [ ] **Class 2.** Spectrum encoder (peak set → transformer → fingerprint) and a contrastive spectrum↔molecule embedding with same-formula hard negatives.
- [ ] **Class 3, the main edge.** Analog-conditioned editing with biotransformation rules (–OH, –OMe, glycosyl, –CH₃, prenyl, sulfate…) plus formula-constrained de novo generation. Rank with the fingerprint net and a fragmentation score.
- [ ] **Lock-in.** Profile on a T4 within 9 h. Pick two finals: safe retrieval and a Class-3-weighted variant.

---

## References

- Competition: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>
- MS2Mol (Enveda): <https://chemrxiv.org/engage/chemrxiv/article-details/648dead54f8b1884b77082da>
- Data sources: COCONUT (CC0), PubChem via `prvsiyan/pubchem-npformula-massindex` (CC0), MassBank / MoNA / GNPS
- Third-party code and models, with their licences: see `NOTICE` (the prvsiyan fingerprint checkpoints are CC BY-NC-SA 4.0)
- Tools: RDKit, matchms, ms_entropy, MIST, SIRIUS / CSI:FingerID
