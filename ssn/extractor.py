"""
Free-boundary extractor: chi_A → Γ.

Takes the active-set indicator on DOF coordinates and returns a point
cloud on the free boundary Γ = ∂{chi_A = 1}, using matplotlib contour
on a griddata-interpolated regular grid.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def extract_free_boundary(
    chi_A_vals: np.ndarray,
    dof_coords: np.ndarray,
    level: float = 0.5,
    grid_res: int = 128,
) -> tuple[np.ndarray, callable]:
    """
    Parameters
    ----------
    chi_A_vals : (N,) array in {0, 1}
    dof_coords : (N, 2) DOF spatial coordinates
    level      : contour level (0.5 splits active/inactive)
    grid_res   : interpolation grid resolution

    Returns
    -------
    boundary_pts : (M, 2)  points on Γ
    mask_inner   : callable (pts: (K,2)) → (K,) bool  — True inside Ω_A
    """
    dim = dof_coords.shape[1]

    # 3D: contour extraction not applicable — return DOFs on the discrete
    # active/inactive transition as an approximation of Γ.
    if dim == 3:
        boundary_pts = _extract_boundary_3d(chi_A_vals, dof_coords, level)

        def mask_inner_fn(pts: np.ndarray) -> np.ndarray:
            vals = griddata(dof_coords, chi_A_vals, pts, method="linear", fill_value=0.0)
            return vals > level

        return boundary_pts, mask_inner_fn

    # 2D: matplotlib contour on interpolated grid
    xi = np.linspace(dof_coords[:, 0].min(), dof_coords[:, 0].max(), grid_res)
    yi = np.linspace(dof_coords[:, 1].min(), dof_coords[:, 1].max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(dof_coords, chi_A_vals, (Xi, Yi), method="linear", fill_value=0.0)

    fig, ax = plt.subplots()
    cs = ax.contour(Xi, Yi, Zi, levels=[level])
    plt.close(fig)

    pts_list = [p.vertices for p in cs.get_paths()]
    boundary_pts = np.vstack(pts_list) if pts_list else np.empty((0, 2))

    def mask_inner_fn(pts: np.ndarray) -> np.ndarray:
        vals = griddata(dof_coords, chi_A_vals, pts, method="linear", fill_value=0.0)
        return vals > level

    return boundary_pts, mask_inner_fn


def _extract_boundary_3d(
    chi_A_vals: np.ndarray,
    dof_coords: np.ndarray,
    level: float = 0.5,
) -> np.ndarray:
    """
    Approximate 3D free boundary as DOFs in the transition zone:
    chi_A interpolated to a regular grid, thresholded at `level ± 0.3`.
    Returns a point cloud of (M, 3) coordinates near Γ.
    """
    from scipy.spatial import cKDTree

    active   = chi_A_vals > level
    inactive = ~active

    if not active.any() or not inactive.any():
        return np.empty((0, 3))

    # Find active DOFs that have at least one inactive neighbour within radius r
    active_coords   = dof_coords[active]
    inactive_coords = dof_coords[inactive]

    # Estimate neighbourhood radius from mean inter-DOF spacing
    n_sample = min(1000, len(dof_coords))
    sample   = dof_coords[np.random.choice(len(dof_coords), n_sample, replace=False)]
    tree_s   = cKDTree(sample)
    dists, _ = tree_s.query(sample, k=2)
    r        = dists[:, 1].mean() * 3.0

    tree_inactive = cKDTree(inactive_coords)
    near_inactive = tree_inactive.query_ball_point(active_coords, r=r)
    on_boundary   = np.array([len(nb) > 0 for nb in near_inactive])

    return active_coords[on_boundary]
