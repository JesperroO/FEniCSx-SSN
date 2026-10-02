"""
Entry point: load SSNResult, train XPINNs, evaluate, save figures.

Works for both 2D and 3D — spatial dimension is auto-detected from the .npz.
For 3D, evaluation and plots use the z=0.5 mid-plane slice.
Rel L² error is always computed at SSN DOF coordinates for accuracy.

Can run anywhere with torch + scipy + matplotlib (no dolfinx needed).

Usage:
    python3 run_xpinn.py [--input ssn_result.npz] [--device cpu|cuda]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from interface.ssn_result import SSNResult
from xpinn.network import NetworkConfig
from xpinn.loss import LossConfig
from xpinn.trainer import TrainConfig, train
from xpinn.predictor import predict_at_points, predict_on_grid, relative_l2

THRESHOLD = 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="ssn_result.npz")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--outdir",  default="output_v2")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    result = SSNResult.load(args.input)
    dim    = result.dof_coords.shape[1]

    print("=" * 60)
    print(f"  XPINN trainer  input={args.input}  dim={dim}D  device={args.device}")
    print("=" * 60)
    print(result.summary())

    cfg = TrainConfig(
        net=NetworkConfig(input_dim=dim, hidden_dim=32, n_layers=2),
        loss=LossConfig(lam_pde=0.0, lam_bc=10.0, lam_ic=20.0, lam_reg=50.0),
        optimizer="lbfgs",
        max_iter=3000,
        warm_iters=1000,
        n_pde=600, n_bc=200, n_ic=200,
        seed=args.seed,
        device=args.device,
    )

    net_A, net_B, info = train(result, cfg)
    mask_inner_fn = info["mask_inner_fn"]

    # ── Relative L² error at SSN DOF coordinates (dimension-agnostic) ────
    coords = result.dof_coords.astype(np.float32)
    y_pred = predict_at_points(net_A, net_B, mask_inner_fn,
                                coords, device=args.device)
    rel_err = relative_l2(y_pred, result.y_star)
    passed  = rel_err < THRESHOLD

    print(f"\n  Relative L² error: {rel_err:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}: {rel_err:.4f} < {THRESHOLD} → {passed}\n")

    # ── Grid / slice evaluation for plotting ──────────────────────────────
    GRID = 64
    Xi, Yi, Z_xpinn = predict_on_grid(
        net_A, net_B, mask_inner_fn,
        grid_res=GRID, device=args.device, dim=dim,
    )

    # SSN reference on the same grid / slice
    pts_grid = np.column_stack([Xi.ravel(), Yi.ravel()])
    if dim == 3:
        zi       = np.full((GRID * GRID, 1), 0.5, dtype=np.float32)
        pts_grid = np.column_stack([pts_grid, zi])
    Z_ssn = griddata(
        result.dof_coords, result.y_star, pts_grid,
        method="linear", fill_value=0.0,
    ).reshape(GRID, GRID).astype(np.float32)

    # ── Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    kw = dict(shading="auto", cmap="viridis")

    slice_label = "  (z=0.5 slice)" if dim == 3 else ""

    im0 = axes[0].pcolormesh(Xi, Yi, Z_ssn, **kw)
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"SSN  y*{slice_label}")
    axes[0].set_aspect("equal")

    im1 = axes[1].pcolormesh(Xi, Yi, Z_xpinn, **kw)
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"XPINNs  ŷ{slice_label}")
    axes[1].set_aspect("equal")
    bdy = result.boundary_pts
    if len(bdy):
        if dim == 2:
            axes[1].scatter(bdy[:, 0], bdy[:, 1], c="w", s=2, alpha=0.5)
        else:
            # Show boundary points near z=0.5 slice
            near = np.abs(bdy[:, 2] - 0.5) < 0.05
            if near.any():
                axes[1].scatter(bdy[near, 0], bdy[near, 1],
                                c="w", s=3, alpha=0.6)

    im2 = axes[2].pcolormesh(Xi, Yi, np.abs(Z_xpinn - Z_ssn),
                              cmap="Reds", shading="auto")
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title(f"|ŷ − y*|  (rel L²={rel_err:.3f})")
    axes[2].set_aspect("equal")

    tag = "3D" if dim == 3 else "2D"
    fig.suptitle(f"SSN-Guided XPINNs V2 — {tag} — {result.mesh_n}³ mesh"
                 if dim == 3 else
                 f"SSN-Guided XPINNs V2 — {tag} — {result.mesh_n}×{result.mesh_n} mesh")
    fig.tight_layout()
    path = os.path.join(args.outdir, "xpinn_comparison.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  Saved: {path}")

    # Loss curve
    if info["loss_history"]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(info["loss_history"])
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss (log)")
        ax.set_title("XPINN Training Loss")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path2 = os.path.join(args.outdir, "xpinn_loss.png")
        fig.savefig(path2, dpi=120); plt.close(fig)
        print(f"  Saved: {path2}")

    if not passed:
        raise SystemExit(f"FAIL: rel_err={rel_err:.4f} ≥ {THRESHOLD}")


if __name__ == "__main__":
    main()
