"""Forward models (structure → predicted MS/MS) ported from coleygroup/ms-pred (MIT) to plain PyTorch.

``glacier`` and ``iceberg`` load weights-only copies of the MassSpecGym checkpoints and predict
sparse spectra for candidate structures; ``score`` compares them with the query spectra and fuses
the result with the ranker.
"""

from __future__ import annotations

from collections.abc import Callable

FILES = {"iceberg": ("iceberg_msg_all_gen_sd.pt", "iceberg_msg_all_inten_contr_sd.pt"),
         "glacier": ("glacier_msg_sd.pt",)}


def load_model(name: str, find: Callable[[str], str], device: str | None = None):
    """The forward model ``name`` (``FILES``) from the weight files ``find`` locates (it raises when one is
    missing)."""
    if name == "iceberg":
        from casmi.forward.iceberg import Iceberg
        return Iceberg.load(*map(find, FILES[name]), device=device)
    if name == "glacier":
        from casmi.forward.glacier import Glacier
        return Glacier.load(find(FILES[name][0]), device=device)
    raise ValueError(f"unknown forward model {name!r}; known: {sorted(FILES)}")
