"""
Evaluate the trained XPINN pair.

predict_at_points  — evaluate at an arbitrary point cloud (2D or 3D)
predict_on_grid    — evaluate on a 2D uniform grid (or z=0.5 slice for 3D)
relative_l2        — relative L² error between prediction and reference
"""

from __future__ import annotations
import numpy as np
import torch
from .network import MLP


def predict_at_points(
    net_A: MLP,
    net_B: MLP,
    mask_inner_fn,
    pts: np.ndarray,        # (N, dim)
    device: str = "cpu",
) -> np.ndarray:
    """
    Evaluate XPINN at an arbitrary point cloud.
    net_A handles the active subdomain, net_B the inactive subdomain.
    Returns (N,) float32 array.
    """
    inner = mask_inner_fn(pts)
    vals  = np.zeros(len(pts), dtype=np.float32)

    with torch.no_grad():
        if inner.any():
            t = torch.tensor(pts[inner].astype(np.float32), device=device)
            vals[inner] = net_A(t).cpu().numpy()
        if (~inner).any():
            t = torch.tensor(pts[~inner].astype(np.float32), device=device)
            vals[~inner] = net_B(t).cpu().numpy()

    return vals


def predict_on_grid(
    net_A: MLP,
    net_B: MLP,
    mask_inner_fn,
    grid_res: int = 64,
    device: str = "cpu",
    dim: int = 2,
    z_slice: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate on a 2D uniform grid (dim=2) or the z=z_slice plane (dim=3).

    Returns
    -------
    Xi, Yi   : (grid_res, grid_res) meshgrid arrays  [x, y coords]
    Z_pred   : (grid_res, grid_res) XPINN prediction
    """
    xi = np.linspace(0, 1, grid_res, dtype=np.float32)
    Xi, Yi = np.meshgrid(xi, xi)
    flat   = np.column_stack([Xi.ravel(), Yi.ravel()])

    if dim == 3:
        zi   = np.full((len(flat), 1), z_slice, dtype=np.float32)
        pts  = np.column_stack([flat, zi])
    else:
        pts = flat

    vals = predict_at_points(net_A, net_B, mask_inner_fn, pts, device)
    return Xi, Yi, vals.reshape(grid_res, grid_res)


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L² error ‖pred − ref‖ / ‖ref‖ (array-shape-agnostic)."""
    num   = np.sqrt(((pred - ref) ** 2).mean())
    denom = np.sqrt((ref ** 2).mean())
    return float(num / (denom + 1e-14))
