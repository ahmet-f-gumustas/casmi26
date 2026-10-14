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
│   ├── ranker.py    # ranker v0: weighted sum of per-molecule rank-normalised channel scores
│   ├── difficulty.py # nn_tanimoto / analog counts (blocked matmul), difficulty bins, profile tilt
│   ├── candidates.py # mass-sorted candidate pool, ppm windows with instrument offset, exclusions
│   ├── channels/    # library (entropy similarity), analog (mass-shifted analogs), fragment (bond breaks)
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
| `scripts/check_metric.py` | `notes/METRIC_CHECK.md`: `casmi.metric` vs the official Kaggle scorer (EXP-000) |
| `validation/candidate_recall.py` | `notes/RECALL_<split>.md`: is the answer in its mass window, per class (C3 must be 0) |
| `validation/eval_analog.py` | `notes/EXP002_analog_*.md`: analog-channel grid (power, top-K, aggregation) per class; needs the Flash index in `outputs/analog_index` |
| `validation/eval_fragment.py` | `notes/EXP003_fragment_*.md`: fragmentation-channel MRR per class, time per candidate, agreement with the analog channel |
| `validation/eval_ranker.py` | `notes/EXP004_ranker_*.md`, `outputs/ranker_v0.json`: ranker-v0 weights tuned on half A, tested on half B |
| `validation/seed_noise.py` | `notes/EXP005_seed_noise.md`: library / analog MRR@25 per class across split seeds, mean ± std |
| `validation/eval_library.py` | `notes/EXP001_library.md`: library-channel MRR@25 per class / source (needs `.[ml]`) |
| `scripts/make_kaggle_notebook.py` | `kaggle/submission/`: submission notebook with `casmi` embedded; `--local-check` runs it on the placeholder test |
| `scripts/kaggle_datasets.py` | stages and uploads the Kaggle datasets above |
| `validation/make_split.py` | `outputs/splits/{seedN,v2_seedN,final}/`: `val_ids.parquet`, `library_mask.parquet`, `config.json`; `--difficulty` = split v2 |

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
| `red6161/casmi26-analog-index` | Flash entropy index of the cleaned library (2.8 GB, files flattened as `<mode>__<file>`) |
| `metric/rdkit-2026-3-3-wheel` (public) | RDKit 2026.3.3 wheels used by the official scorer |

```bash
python scripts/kaggle_datasets.py pool library index wheels  # first upload (--update -m msg for new versions)
python scripts/make_kaggle_notebook.py --local-check         # library-only notebook, run locally on the placeholder
python scripts/make_kaggle_notebook.py --mode combined       # ranker v0 with weights from outputs/ranker_v0.json
kaggle kernels push -p kaggle/submission
kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra \
    -k red6161/casmi26-submission -v <version> -f submission.csv -m "<message>"
```

| Submission | Public LB | Val MRR@25 | Kaggle runtime / peak RSS |
|---|---|---|---|
| library-only v1 | 0.137 | 0.144 (seed0); 0.136 ± 0.007 (3 seeds) | 110 s / 9.4 GB (CPU) |
| ranker v0 = analog only (p3, K100, mean) | 0.285 | 0.492 (seed0); 0.479 ± 0.012 (3 seeds) | 266 s / 3.9 GB (CPU, 4 workers) |

Validation tracks the LB for Class 1 but overestimates Class 2 about twofold: simulated C2 molecules are common
compounds whose spectra were removed, so close analogs abound, while the hidden C2 structures have no public spectra
and may only exist in PubChem (not in this pool).

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
- Data sources: COCONUT (CC0), PubChem, MassBank / MoNA / GNPS
- Tools: RDKit, matchms, ms_entropy, MIST, SIRIUS / CSI:FingerID
