"""Train the spectrum → fingerprint model (``casmi.fpmodel.SpecFormer``, casmi6311 bits).

Data: ``outputs/fp_data`` (``scripts/build_fp_data.py``); spectra allowed by ``--family``
(``v2``: library rows of every v2 split, so no validation molecule is seen; ``all``: everything,
for the submission model). Each spectrum is a training example; a molecule's spectra are
averaged only at inference.

Loss = BCE over bits (pos_weight = clip(sqrt(neg/pos), 1, 30) for rare bits)
     + λ × softmax cross-entropy over the true structure and 32 decoys from its mass window,
       scored by soft-Tanimoto(sigmoid(z), f) / τ.

    python scripts/train_fp.py --out outputs/fp_models/smoke --subset 200000 --max-minutes 60
    python scripts/train_fp.py --out outputs/fp_models/v2_full --epochs 1
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
import torch
import torch.nn.functional as F

from casmi.config import WORK_DIR
from casmi.fpmodel import SpecFormer

DATA = WORK_DIR / "fp_data"
NBITS = 6311


class Data:
    def __init__(self, family: str, subset: int | None, seed: int) -> None:
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
        self.lens = np.diff(self.off)
        dstruct = load("decoy_struct")
        self.decoy_row = np.searchsorted(dstruct, np.maximum(self.struct, 0)).astype(np.int64)
        self.decoys = load("decoys")
        self.pool_fp = load("pool_fp")

    def batches(self, batch: int, rng: np.random.Generator, mega: int = 64):
        """Shuffled batches of similar length (sorted within mega-batches)."""
        perm = rng.permutation(self.idx)
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
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    t0 = time.time()

    data = Data(args.family, args.subset, args.seed)
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
    print(f"{len(data.idx):,} training spectra, {len(structs):,} structures; bit frequency mean {q.mean():.4f}; "
          f"{time.time() - t0:.0f}s", flush=True)

    model = SpecFormer(NBITS, d=args.d, layers=args.layers, heads=args.heads, drop=args.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    per_epoch = math.ceil(len(data.idx) / args.batch)
    total = max(1, int(per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))
    config = {"nbits": NBITS, "d": args.d, "layers": args.layers, "heads": args.heads}
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "steps": []}

    step, deadline = 0, (time.time() + 60 * args.max_minutes) if args.max_minutes else None
    model.train()
    t_train, ema = time.time(), None
    while step < total and (deadline is None or time.time() < deadline):
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
            if step >= total or (deadline is not None and time.time() >= deadline):
                break
    ck = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": config, "step": step,
          "family": args.family, "layout": "casmi6311"}
    torch.save(ck, args.out / "fp_casmi.pt")
    log.update(steps_done=step, steps_planned=total, epochs_done=step / per_epoch,
               minutes=round((time.time() - t0) / 60, 1))
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {args.out / 'fp_casmi.pt'}: {step} steps ({step / per_epoch:.2f} epochs), "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
