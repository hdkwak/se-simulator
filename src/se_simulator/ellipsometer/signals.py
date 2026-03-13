"""EllipsometryResult dataclass and CSV I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EllipsometryResult:
    wavelengths_nm: np.ndarray       # (Nλ,)
    psi_deg: np.ndarray              # (Nλ,), range [0, 90]
    delta_deg: np.ndarray            # (Nλ,), range (-180, 180]
    alpha: np.ndarray                # (Nλ,), normalized 2δ cosine coefficient
    beta: np.ndarray                 # (Nλ,), normalized 2δ sine coefficient
    chi: np.ndarray                  # (Nλ,), normalized 4δ cosine coefficient
    xi: np.ndarray                   # (Nλ,), normalized 4δ sine coefficient
    jones_reflection: np.ndarray     # (Nλ, 2, 2), complex
    energy_conservation: np.ndarray  # (Nλ,)
    mueller_matrix: np.ndarray | None = field(default=None)  # (Nλ, 4, 4), real

    def export_mueller_csv(self, path: Path) -> None:
        """Write Mueller matrix to CSV.

        Columns: wavelength_nm, m11, m12, m13, m14, m21, m22, m23, m24,
                 m31, m32, m33, m34, m41, m42, m43, m44
        m11 column is always 1.0 (normalized).
        """
        if self.mueller_matrix is None:
            raise ValueError("No Mueller matrix data available.")
        path = Path(path)
        header = "wavelength_nm," + ",".join(
            f"m{i + 1}{j + 1}" for i in range(4) for j in range(4)
        )
        rows = [header]
        for k in range(len(self.wavelengths_nm)):
            vals = [f"{self.wavelengths_nm[k]:.6f}"]
            for i in range(4):
                for j in range(4):
                    vals.append(f"{self.mueller_matrix[k, i, j]:.10f}")
            rows.append(",".join(vals))
        path.write_text("\n".join(rows) + "\n")

    def to_dataframe(self) -> pd.DataFrame:  # type: ignore[name-defined]  # noqa: F821
        """Return a pandas DataFrame with scalar columns."""
        import pandas as pd

        return pd.DataFrame({
            "wavelength_nm": self.wavelengths_nm,
            "psi_deg": self.psi_deg,
            "delta_deg": self.delta_deg,
            "alpha": self.alpha,
            "beta": self.beta,
            "chi": self.chi,
            "xi": self.xi,
            "energy_conservation": self.energy_conservation,
        })

    def to_csv(self, path: Path, metadata: dict | None = None) -> None:
        """Write results to CSV with optional # metadata header lines."""
        path = Path(path)
        lines: list[str] = []
        if metadata:
            for k, v in metadata.items():
                lines.append(f"# {k}={v}\n")
        header = "wavelength_nm,psi_deg,delta_deg,alpha,beta,chi,xi,energy_conservation\n"
        lines.append(header)
        for i in range(len(self.wavelengths_nm)):
            row = (
                f"{self.wavelengths_nm[i]:.6f},"
                f"{self.psi_deg[i]:.6f},"
                f"{self.delta_deg[i]:.6f},"
                f"{self.alpha[i]:.6f},"
                f"{self.beta[i]:.6f},"
                f"{self.chi[i]:.6f},"
                f"{self.xi[i]:.6f},"
                f"{self.energy_conservation[i]:.6f}\n"
            )
            lines.append(row)
        path.write_text("".join(lines))

    @classmethod
    def from_csv(cls, path: Path) -> EllipsometryResult:
        """Load a previously saved EllipsometryResult CSV."""
        path = Path(path)
        # Count leading comment lines so names=True reads the actual header row
        with open(path) as f:
            n_comments = sum(1 for line in f if line.startswith("#"))
        data = np.genfromtxt(path, delimiter=",", skip_header=n_comments, names=True)
        if data.ndim == 0:
            data = data[np.newaxis]
        n = len(data)

        dummy_jones = np.zeros((n, 2, 2), dtype=complex)
        return cls(
            wavelengths_nm=data["wavelength_nm"],
            psi_deg=data["psi_deg"],
            delta_deg=data["delta_deg"],
            alpha=data["alpha"],
            beta=data["beta"],
            chi=data["chi"],
            xi=data["xi"],
            jones_reflection=dummy_jones,
            energy_conservation=data["energy_conservation"],
        )
