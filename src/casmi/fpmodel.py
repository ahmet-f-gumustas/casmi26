"""Spectrum → fingerprint models (PyTorch), for ``casmi.channels.fingerprint``.

``FPNet``, ``prep_peaks_ap`` and ``merge_peaks`` reproduce the architecture and input
preparation of prvsiyan's public "Analog Propagation CASMI 2026 baseline" notebook
(Apache 2.0) so that its published checkpoints (``prvsiyan/casmi26-fp-models-v2``,
CC BY-NC-SA 4.0; see NOTICE) load unchanged. Checkpoints are read with
``weights_only=True``; a checkpoint that needs unrestricted unpickling is refused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from casmi import degraded
from casmi.data import mean_ce

ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
DEFAULT_CE = 25.0


def instr_family(s: object) -> int:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return 4
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if any(w in t for w in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive")):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qq" in t:
        return 3
    return 4


def prep_peaks_ap(mz: NDArray, inten: NDArray, prec_mz: float, max_peaks: int = 128, floor: float = 1e-3,
                  win: float = 50.0, per_win: int = 8) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Peaks ≤ precursor + 1.5, ≥ floor × max; window-diversified top-N; m/z order; sqrt intensity."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    empty = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if not len(mz) or it.max() <= 0:
        return empty
    keep = it >= floor * it.max()
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt: dict[int, int] = {}
        sel = []
        for i in order:
            c = cnt.get(bucket[i], 0)
            if c < per_win:
                cnt[bucket[i]] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            if len(rest):
                sel = np.concatenate([sel, rest[: max_peaks - len(sel)]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


def merge_peaks(mzs: Sequence[NDArray], its: Sequence[NDArray], tol: float = 0.005) -> tuple[NDArray, NDArray]:
    """All of a molecule's peaks (each spectrum max-normalised) in one list; near-duplicates keep the stronger."""
    mz = np.concatenate([np.asarray(a, float) for a in mzs])
    it = np.concatenate([np.asarray(b, float) / max(float(np.asarray(b, float).max()), 1e-9) if len(b) else
                         np.zeros(0) for b in its])
    o = np.argsort(mz, kind="stable")
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < tol:
            if it[j] >= it[j - 1]:
                keep[j - 1] = False
            else:
                keep[j] = False
    return mz[keep], it[keep]


class SinEmb(nn.Module):
    """Log-spaced sinusoidal embedding of m/z values."""

    def __init__(self, dim: int, lo: float = -2.0, hi: float = 3.2, power: float = 1.0) -> None:
        super().__init__()
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, dim // 2), power) + lo)
        self.register_buffer("inv", (2 * math.pi) / wav)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, d: int, h: int, drop: float) -> None:
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :])
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    """Peak tokens (m/z, neutral loss, intensity) + one global token (precursor, adduct, instrument, CE, mode)."""

    def __init__(self, nbits: int, d: int = 512, layers: int = 6, heads: int = 8, drop: float = 0.1) -> None:
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode) -> torch.Tensor:
        b = mz.shape[0]
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.norm(x)
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([x[:, 0], mean], -1))


@dataclass
class Batch:
    """Padded model inputs for B spectra (numpy; ``tensors`` moves them to a device)."""

    mz: NDArray[np.float32]
    it: NDArray[np.float32]
    pad: NDArray[np.bool_]
    prec: NDArray[np.float32]
    ad: NDArray[np.int64]
    ins: NDArray[np.int64]
    ce: NDArray[np.float32]
    mode: NDArray[np.float32]

    def tensors(self, device: str | torch.device) -> tuple[torch.Tensor, ...]:
        return tuple(torch.as_tensor(getattr(self, f), device=device)
                     for f in ("mz", "it", "pad", "prec", "ad", "ins", "ce", "mode"))


def make_batch(peaks: Sequence[tuple[NDArray, NDArray]], prec, adduct, instrument, ce, mode) -> Batch:
    n = max(len(a) for a, _ in peaks)
    b = len(peaks)
    mz = np.zeros((b, n), np.float32)
    it = np.zeros((b, n), np.float32)
    pad = np.ones((b, n), bool)
    for i, (a, w) in enumerate(peaks):
        mz[i, : len(a)], it[i, : len(w)], pad[i, : len(a)] = a, w, False
    return Batch(mz, it, pad, np.asarray(prec, np.float32),
                 np.array([ADDUCT_IX.get(a, ADDUCT_IX["<unk>"]) for a in adduct], np.int64),
                 np.array([instr_family(s) for s in instrument], np.int64),
                 np.asarray(ce, np.float32), np.asarray(mode, np.float32))


@dataclass
class APBundle:
    """prvsiyan's FPNet checkpoints: per-spectrum models and merged-peak-list models, averaged."""

    single: list[FPNet]
    merged: list[FPNet]
    nbits: int
    device: str
    bits: NDArray[np.int64]  # fp_bits.npy: ap6930 columns of the full descriptor vector
    layout: str = "ap6930"

    @classmethod
    def load(cls, paths: Sequence[Path], bits: NDArray[np.int64], device: str | None = None) -> APBundle:
        if device is None and not torch.cuda.is_available():
            degraded.note("fingerprint model on the CPU (no GPU)")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        single, merged, nbits = [], [], None
        for p in sorted(paths):
            ck = torch.load(p, map_location="cpu", weights_only=True)
            net = FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"])
            net.load_state_dict(ck["model"])
            net.to(device).eval()
            (merged if "merged" in Path(p).name else single).append(net)
            nbits = ck["nbits"]
        if nbits is None:
            raise FileNotFoundError("no fingerprint checkpoints given")
        if len(bits) != nbits:
            raise ValueError(f"fp_bits has {len(bits)} columns, the checkpoints {nbits}")
        return cls(single, merged, int(nbits), device, np.asarray(bits, dtype=np.int64))

    @torch.no_grad()
    def _run(self, nets: list[FPNet], batch: Batch) -> NDArray[np.float32]:
        args = batch.tensors(self.device)
        return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

    def logits(self, spectra: pd.DataFrame) -> NDArray[np.float32] | None:
        """Per-bit logits for one molecule (rows: its spectra, with the query columns)."""
        rows = list(spectra.itertuples(index=False))
        out = []
        if self.single:
            kept = [(prep_peaks_ap(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)), r) for r in rows]
            kept = [(p, r) for p, r in kept if len(p[0])]
            if kept:
                rs = [r for _, r in kept]
                batch = make_batch([p for p, _ in kept], [r.precursor_mz for r in rs], [r.adduct for r in rs],
                                   [r.instrument_type for r in rs], [mean_ce(r.collision_energy_ev, DEFAULT_CE) for r in rs],
                                   [1.0 if r.ionization_mode == "positive" else -1.0 for r in rs])
                out.append(self._run(self.single, batch))
        if self.merged:
            mz, it = merge_peaks([r.ms2_mzs for r in rows], [r.ms2_normalized_intensities for r in rows])
            prec = float(np.median([r.precursor_mz for r in rows]))
            p = prep_peaks_ap(mz, it, prec)
            if len(p[0]):
                mode = float(np.mean([1.0 if r.ionization_mode == "positive" else -1.0 for r in rows]))
                batch = make_batch([p], [prec], [rows[0].adduct], [rows[0].instrument_type], [DEFAULT_CE], [mode])
                out.append(self._run(self.merged, batch))
        return np.mean(out, axis=0).astype(np.float32) if out else None


# --------------------------------------------------------------------------------------
# Our model (casmi6311 layout): peak tokens with m/z rounded to 0.01 Da, at most 256 per
# spectrum, one CLS token for precursor / adduct / CE / polarity / instrument, CLS pooling.
# --------------------------------------------------------------------------------------
MAX_PEAKS = 256


def prep_peaks(mz: NDArray, inten: NDArray, max_peaks: int = MAX_PEAKS) -> tuple[NDArray[np.float32],
                                                                                  NDArray[np.float32]]:
    """m/z rounded to 0.01 Da (duplicates keep the stronger peak), top ``max_peaks`` by intensity,
    m/z order, sqrt(intensity / max)."""
    mz = np.round(np.asarray(mz, np.float64), 2)
    it = np.asarray(inten, np.float64)
    ok = it > 0
    mz, it = mz[ok], it[ok]
    if not len(mz):
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    o = np.lexsort((-it, mz))  # by m/z, strongest first within an m/z
    mz, it = mz[o], it[o]
    first = np.concatenate([[True], mz[1:] != mz[:-1]])
    mz, it = mz[first], it[first]
    if len(mz) > max_peaks:
        top = np.sort(np.argpartition(-it, max_peaks - 1)[:max_peaks])
        mz, it = mz[top], it[top]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


class SpecFormer(nn.Module):
    """Spectrum → fingerprint logits; ``forward`` takes a ``Batch`` with ``ce`` NaN where unknown."""

    def __init__(self, nbits: int, d: int = 512, layers: int = 8, heads: int = 8, drop: float = 0.1) -> None:
        super().__init__()
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.pol = nn.Embedding(2, d)
        self.ce = nn.Linear(2, d)
        self.cls = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode) -> torch.Tensor:
        b = mz.shape[0]
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        known = torch.isfinite(ce)
        ce_feat = torch.stack([torch.where(known, ce / 100.0, torch.zeros_like(ce)), known.float()], -1)
        g = (self.cls(self.prec_emb(prec)) + self.ad(ad) + self.ins(ins) + self.pol((mode > 0).long())
             + self.ce(ce_feat))
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        return self.head(self.norm(x[:, 0]))


@dataclass
class CasmiBundle:
    """Our SpecFormer checkpoint(s); logits averaged over models and over a molecule's spectra."""

    nets: list[SpecFormer]
    nbits: int
    device: str
    bits: None = None
    layout: str = "casmi6311"

    @classmethod
    def load(cls, paths: Sequence[Path], device: str | None = None) -> CasmiBundle:
        if device is None and not torch.cuda.is_available():
            degraded.note("fingerprint model on the CPU (no GPU)")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        nets, nbits = [], None
        for p in sorted(paths):
            ck = torch.load(p, map_location="cpu", weights_only=True)
            cfg = ck["config"]
            net = SpecFormer(cfg["nbits"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"])
            net.load_state_dict(ck["model"])
            nets.append(net.to(device).eval())
            nbits = cfg["nbits"]
        if not nets:
            raise FileNotFoundError("no fingerprint checkpoints given")
        return cls(nets, int(nbits), device)

    @torch.no_grad()
    def logits(self, spectra: pd.DataFrame) -> NDArray[np.float32] | None:
        rows = [r for r in spectra.itertuples(index=False)]
        kept = [(prep_peaks(r.ms2_mzs, r.ms2_normalized_intensities), r) for r in rows]
        kept = [(p, r) for p, r in kept if len(p[0])]
        if not kept:
            return None
        rs = [r for _, r in kept]
        batch = make_batch([p for p, _ in kept], [r.precursor_mz for r in rs], [r.adduct for r in rs],
                           [r.instrument_type for r in rs], [mean_ce(r.collision_energy_ev) for r in rs],
                           [1.0 if r.ionization_mode == "positive" else -1.0 for r in rs])
        args = batch.tensors(self.device)
        z = [n(*args).float().mean(0).cpu().numpy() for n in self.nets]
        return np.mean(z, axis=0).astype(np.float32)
