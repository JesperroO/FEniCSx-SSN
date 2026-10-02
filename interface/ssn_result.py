"""
SSNResult — data contract between the SSN solver and the XPINN trainer.

The SSN side (FEniCSx) saves an SSNResult to a .npz file.
The XPINN side (PyTorch) loads it without any FEniCSx dependency.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SSNResult:
    # FEM solution at DOF coordinates
    dof_coords:   np.ndarray   # (N, dim) spatial coordinates of all DOFs
    y_star:       np.ndarray   # (N,)     optimal state
    u_star:       np.ndarray   # (N,)     optimal control
    p_star:       np.ndarray   # (N,)     adjoint variable (determines active set)
    chi_A:        np.ndarray   # (N,)     active-set indicator in {0, 1}

    # Free boundary extracted from chi_A
    boundary_pts: np.ndarray   # (M, dim) points on Γ = ∂{chi_A = 1}

    # Problem parameters / metadata. For spatial alpha(x) datasets, alpha stores
    # only the scalar baseline; reconstruct the full alpha field from the design CSV.
    nu:    float               # Tikhonov/L2 control regularisation weight
    alpha: float               # scalar threshold or spatial-alpha baseline

    # Metadata
    mesh_n: int = 64           # mesh resolution (N×N)
    ssn_iters: int = 0         # number of SSN iterations at convergence

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save to .npz. Scalars stored as 0-d arrays."""
        np.savez(
            path,
            dof_coords=self.dof_coords,
            y_star=self.y_star,
            u_star=self.u_star,
            p_star=self.p_star,
            chi_A=self.chi_A,
            boundary_pts=self.boundary_pts,
            nu=np.array(self.nu),
            alpha=np.array(self.alpha),
            mesh_n=np.array(self.mesh_n),
            ssn_iters=np.array(self.ssn_iters),
        )

    @classmethod
    def load(cls, path: str) -> "SSNResult":
        """Load from .npz produced by save()."""
        d = np.load(path)
        # p_star added later; default to zeros for backward-compat with old .npz
        p_star = d["p_star"] if "p_star" in d else np.zeros_like(d["y_star"])
        return cls(
            dof_coords=d["dof_coords"],
            y_star=d["y_star"],
            u_star=d["u_star"],
            p_star=p_star,
            chi_A=d["chi_A"],
            boundary_pts=d["boundary_pts"],
            nu=float(d["nu"]),
            alpha=float(d["alpha"]),
            mesh_n=int(d["mesh_n"]),
            ssn_iters=int(d["ssn_iters"]),
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        n_active   = int(self.chi_A.sum())
        n_inactive = len(self.chi_A) - n_active
        dim        = self.dof_coords.shape[1]
        mesh_str   = f"{self.mesh_n}³" if dim == 3 else f"{self.mesh_n}×{self.mesh_n}"
        return (
            f"SSNResult | mesh {mesh_str} | dim {dim}D | "
            f"DOFs {len(self.dof_coords)} | "
            f"active {n_active} inactive {n_inactive} | "
            f"Γ pts {len(self.boundary_pts)} | "
            f"y* ∈ [{self.y_star.min():.3f}, {self.y_star.max():.3f}] | "
            f"u* ∈ [{self.u_star.min():.3f}, {self.u_star.max():.3f}] | "
            f"p* ∈ [{self.p_star.min():.3f}, {self.p_star.max():.3f}] | "
            f"ν={self.nu} α={self.alpha} | "
            f"SSN iters {self.ssn_iters}"
        )
