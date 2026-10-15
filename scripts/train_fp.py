"""Train the spectrum → fingerprint model (``casmi.fpmodel.SpecFormer``, casmi6311 bits).

Data: ``outputs/fp_data`` (``scripts/build_fp_data.py``); spectra allowed by ``--family``
(``v2``: library rows of every v2 split, so no validation molecule is seen; ``all``: everything,
for the submission model). Each spectrum is a training example; a molecule's spectra are
averaged only at inference.

Loss = BCE over bits (pos_weight = clip(sqrt(neg/pos), 1, 30) for rare bits)
     + λ × softmax cross-entropy over the true structure and 32 decoys from its mass window,
       scored by soft-Tanimoto(sigmoid(z), f) / τ.

``--np-weight w``: an epoch is len(spectra) draws with replacement, NP spectra (structure in
COCONUT, or an enveda-np-examples spectrum) w times as likely as the others. ``--eval-splits``:
after every epoch the model re-ranks those splits' validation molecules with the cached analog /
library scores of their ranker features (``casmi.harness.CachedSplit``) under the C1-protected
rank average (``--eval-rule``); the epoch with the best mean Class 2 NP MRR is kept as
``fp_casmi.pt`` (every epoch is saved as ``epoch_<n>.pt``), and ``--patience`` epochs without a
gain stop training. ``--stop-after`` ends a run early while the LR schedule still spans ``--epochs``
(the final model replays the selected epoch count).

    python scripts/train_fp.py --out outputs/fp_models/smoke --subset 200000 --max-minutes 60
    python scripts/train_fp.py --out outputs/fp_models/v2_full --epochs 1
    python scripts/train_fp.py --out outputs/fp_models/np_v2 --family rank --epochs 12 --np-weight 3 \\
        --eval-splits v3_seed0 v3_seed1 v3_seed2 --patience 3
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from casmi.candidates import POOL_DIR
from casmi.config import WORK_DIR
from casmi.fpmodel import CasmiBundle, SpecFormer
from casmi.harness import CLEAN_LIBRARY

DATA = WORK_DIR / "fp_data"
NBITS = 6311
NP_LIB = "enveda-np-examples"


def np_flags(struct: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """NP spectrum: its structure is in COCONUT, or it comes from enveda-np-examples."""
    coconut = pd.read_parquet(POOL_DIR / "meta.parquet", columns=["in_coconut"])["in_coconut"].to_numpy(bool)
    lib = pd.read_parquet(CLEAN_LIBRARY, columns=["row", "ingest_lib"])
    np_rows = set(lib.loc[lib["ingest_lib"] == NP_LIB, "row"].tolist())
    return (coconut[np.maximum(struct, 0)] & (struct >= 0)) | np.isin(rows, list(np_rows))


class Data:
    def __init__(self, family: str, subset: int | None, seed: int, np_weight: float = 1.0) -> None:
        load = lambda n: np.load(DATA / f"{n}.npy")  # noqa: E731
        self.mz, self.it, self.off = load("mz"), load("it"), load("off")
        self.prec, self.adduct, self.instr = load("prec"), load("adduct"), load("instr")
        self.ce, self.mode, self.struct = load("ce"), load("mode"), load("struct")
        valid = self.struct >= 0
        ok = valid & load("pool_ok")[np.where(valid, self.struct, 0)] & (np.diff(self.off) > 0)
        if family != "all":
            ok &= load(f"allowed_{family}")
        idx = np.flatnonzero(ok)
        rng = np.random.default_rng(seed)
        if subset and subset < len(idx):
            idx = np.sort(rng.choice(idx, subset, replace=False))
        self.idx = idx
        self.is_np = np_flags(self.struct[idx], load("row")[idx])
        self.p = None
        if np_weight != 1.0:
            w = np.where(self.is_np, np_weight, 1.0)
            self.p = w / w.sum()
        self.lens = np.diff(self.off)
        dstruct = load("decoy_struct")
        self.decoy_row = np.searchsorted(dstruct, np.maximum(self.struct, 0)).astype(np.int64)
        self.decoys = load("decoys")
        self.pool_fp = load("pool_fp")

    def batches(self, batch: int, rng: np.random.Generator, mega: int = 64):
        """Shuffled batches of similar length (sorted within mega-batches); weighted draws if ``p`` is set."""
        perm = rng.permutation(self.idx) if self.p is None else rng.choice(self.idx, len(self.idx), p=self.p)
        out = []
        for s in range(0, len(perm), batch * mega):
            chunk = perm[s:s + batch * mega]
            chunk = chunk[np.argsort(self.lens[chunk], kind="stable")]
            out += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
        rng.shuffle(out)
        return out

    def make(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        lens = self.lens[idx]
        n, b = int(lens.max()), len(idx)
        rows = np.repeat(np.arange(b), lens)
        cols = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        src = np.repeat(self.off[idx], lens) + cols
        mz = np.zeros((b, n), np.float32)
        it = np.zeros((b, n), np.float32)
        pad = np.ones((b, n), bool)
        mz[rows, cols], it[rows, cols], pad[rows, cols] = self.mz[src], self.it[src], False
        return {"mz": mz, "it": it, "pad": pad, "prec": self.prec[idx], "ad": self.adduct[idx].astype(np.int64),
                "ins": self.instr[idx].astype(np.int64), "ce": self.ce[idx],
                "mode": self.mode[idx].astype(np.float32), "struct": self.struct[idx].astype(np.int64),
                "drow": self.decoy_row[idx]}


class EpochEval:
    """Validation re-ranking after each epoch (see the module docstring)."""

    def __init__(self, splits: list[str], rule: str, pool_fp: np.ndarray) -> None:
        from casmi.harness import CachedSplit

        self.cached = {s: CachedSplit.load(s) for s in splits}
        keys = pd.read_parquet(POOL_DIR / "meta.parquet", columns=["key"])["key"]
        ok = np.load(DATA / "pool_ok.npy")
        need = {k for c in self.cached.values() for ks in c.windows.values() for k in ks}
        self.fp_cache = {k: (pool_fp[i] if ok[i] else None) for i, k in enumerate(keys) if k in need}
        self.weights = {"analog": 1.0, "fingerprint": 1.0}
        self.configs = {"analog only": {"mode": "weighted", "weights": {"analog": 1.0}},
                        "fingerprint only": {"mode": "weighted", "weights": {"fingerprint": 1.0}},
                        "rank average": {"mode": "weighted", "weights": self.weights},
                        "protected": {"mode": "protect", "tau": 0.8, "weights": self.weights, "rule": rule}}

    def __call__(self, model: SpecFormer, device: str) -> dict[str, dict[str, float]]:
        from casmi.channels.fingerprint import fingerprint_channel
        from casmi.pipeline import fingerprint_logits

        model.eval()
        bundle = CasmiBundle([model], NBITS, device)
        out: dict[str, list] = {name: [] for name in self.configs}
        for c in self.cached.values():
            fp = fingerprint_channel(fingerprint_logits(c.queries, c.windows, bundle), c.windows, {}, "casmi6311",
                                     cache=self.fp_cache)
            for name, cfg in self.configs.items():
                out[name].append(c.class_rr({"fingerprint": fp}, cfg))
        model.train()
        return {name: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]} for name, rs in out.items()}


def prefetch(data: Data, batches: list, depth: int = 6):
    q: queue.Queue = queue.Queue(depth)

    def work() -> None:
        for b in batches:
            q.put({k: torch.from_numpy(v).pin_memory() for k, v in data.make(b).items()})
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", default="v2")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--n-decoys", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--np-weight", type=float, default=1.0)
    ap.add_argument("--eval-splits", nargs="*", default=[])
    ap.add_argument("--eval-rule", default="switch", help="C1 protection rule of the evaluated rank average")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--stop-after", type=float, default=None, help="epochs to run (default: --epochs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    t0 = time.time()

    data = Data(args.family, args.subset, args.seed, args.np_weight)
    pool_fp = torch.from_numpy(data.pool_fp).to(dev)
    decoys = torch.from_numpy(data.decoys.astype(np.int64)).to(dev)
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def unpack(rows: torch.Tensor) -> torch.Tensor:
        p = pool_fp[rows]
        return ((p.unsqueeze(-1) >> shifts) & 1).reshape(*p.shape[:-1], -1)[..., :NBITS].float()

    structs, counts = np.unique(data.struct[data.idx], return_counts=True)
    freq = torch.zeros(NBITS, device=dev)
    for s in range(0, len(structs), 20000):
        f = unpack(torch.from_numpy(structs[s:s + 20000]).to(dev))
        freq += (f * torch.from_numpy(counts[s:s + 20000]).to(dev).float()[:, None]).sum(0)
    q = freq / counts.sum()
    pos_weight = torch.sqrt((1 - q) / q.clamp(min=1e-9)).clamp(1.0, 30.0)
    print(f"{len(data.idx):,} training spectra ({data.is_np.mean():.1%} NP, weight {args.np_weight}), "
          f"{len(structs):,} structures; bit frequency mean {q.mean():.4f}; {time.time() - t0:.0f}s", flush=True)
    evaluator = EpochEval(args.eval_splits, args.eval_rule, data.pool_fp) if args.eval_splits else None

    model = SpecFormer(NBITS, d=args.d, layers=args.layers, heads=args.heads, drop=args.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    per_epoch = math.ceil(len(data.idx) / args.batch)
    total = max(1, int(per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))
    config = {"nbits": NBITS, "d": args.d, "layers": args.layers, "heads": args.heads}
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "steps": []}

    step, deadline = 0, (time.time() + 60 * args.max_minutes) if args.max_minutes else None
    stop = min(total, int(per_epoch * args.stop_after)) if args.stop_after else total
    model.train()
    t_train, ema = time.time(), None
    best, best_epoch, epoch = -1.0, 0, 0

    def checkpoint() -> dict:
        return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": config,
                "step": step, "family": args.family, "layout": "casmi6311", "np_weight": args.np_weight}

    while step < stop and (deadline is None or time.time() < deadline):
        for b in prefetch(data, data.batches(args.batch, rng)):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(b["mz"], b["it"], b["pad"], b["prec"], b["ad"], b["ins"], b["ce"], b["mode"])
            z = z.float()
            f = unpack(b["struct"])
            bce = F.binary_cross_entropy_with_logits(z, f, pos_weight=pos_weight)
            pick = torch.randint(0, decoys.shape[1], (len(z), args.n_decoys), device=dev)
            cand = torch.cat([b["struct"][:, None], decoys[b["drow"]].gather(1, pick)], 1)
            fc = unpack(cand)
            p = torch.sigmoid(z)
            inter = (fc * p[:, None, :]).sum(-1)
            st = inter / (fc.sum(-1) + p.sum(-1, keepdim=True) - inter).clamp(min=1e-6)
            rank_loss = F.cross_entropy(st / args.tau, torch.zeros(len(z), dtype=torch.long, device=dev))
            loss = bce + args.lam * rank_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            acc = (st.argmax(1) == 0).float().mean().item() if step % 50 == 0 else None
            if acc is not None:
                cur = {"bce": bce.item(), "rank": rank_loss.item(), "top1_of_33": acc}
                ema = cur if ema is None else {k: 0.9 * ema[k] + 0.1 * v for k, v in cur.items()}
            if step % 500 == 0:
                rate = step * args.batch / (time.time() - t_train)
                rec = {"step": step, **{k: round(v, 4) for k, v in ema.items()}, "lr": sched.get_last_lr()[0],
                       "spectra_per_s": round(rate), "min": round((time.time() - t0) / 60, 1)}
                log["steps"].append(rec)
                print(json.dumps(rec), flush=True)
            if step >= stop or (deadline is not None and time.time() >= deadline):
                break
        epoch += 1
        if evaluator is None:
            continue
        torch.save(checkpoint(), args.out / f"epoch_{epoch}.pt")
        res = evaluator(model, dev)
        metric = res["protected"]["C2 NP"]
        log.setdefault("epochs", []).append({"epoch": epoch, "step": step, "min": round((time.time() - t0) / 60, 1),
                                             **{f"{n} {k}": v for n, r in res.items() for k, v in r.items()}})
        print(json.dumps(log["epochs"][-1]), flush=True)
        if metric > best:
            best, best_epoch = metric, epoch
            torch.save(checkpoint(), args.out / "fp_casmi.pt")
        elif args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop: no C2 NP gain for {args.patience} epochs (best epoch {best_epoch})", flush=True)
            break
        (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    if evaluator is None:
        torch.save(checkpoint(), args.out / "fp_casmi.pt")
    else:
        log.update(best_epoch=best_epoch, best_c2_np=best)
    log.update(steps_done=step, steps_planned=total, epochs_done=step / per_epoch,
               minutes=round((time.time() - t0) / 60, 1))
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {args.out / 'fp_casmi.pt'}: {step} steps ({step / per_epoch:.2f} epochs), "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
