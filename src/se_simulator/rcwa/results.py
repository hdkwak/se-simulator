"""RCWAResult dataclass for storing full simulation output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RCWAResult:
    wavelengths_nm: np.ndarray           # shape (Nλ,)
    jones_reflection: np.ndarray         # shape (Nλ, 2, 2), complex: [[Rpp,Rps],[Rsp,Rss]]
    jones_transmission: np.ndarray       # shape (Nλ, 2, 2), complex
    r_orders: np.ndarray | None = None   # shape (Nλ, n_orders), per-order R efficiency
    t_orders: np.ndarray | None = None   # shape (Nλ, n_orders), per-order T efficiency
    order_indices: list[tuple[int, int]] | None = None  # list of (m, n) pairs
    energy_conservation: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # shape (Nλ,) — R_total + T_total, should be ≈ 1 for lossless
