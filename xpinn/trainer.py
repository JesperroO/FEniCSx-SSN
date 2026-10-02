"""
XPINN trainer: warm-start + joint optimisation of (net_A, net_B).

Supports 2D and 3D — spatial dimension is inferred from SSNResult.dof_coords.

Warm-start: each network regresses on SSN DOF data independently.
Joint opt:  L-BFGS (or Adam) minimises total_loss(net_A, net_B).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, replace
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

from .network import MLP, NetworkConfig
from .loss import LossConfig, total_loss


@dataclass
class TrainConfig:
    net:          NetworkConfig = None    # filled by __post_init__
    loss:         LossConfig    = None
    optimizer:    str  = "lbfgs"   # "lbfgs" | "adam"
    lr:           float = 1.0       # L-BFGS step (1.0) or Adam lr (1e-3)
    max_iter:     int   = 3000
    warm_iters:   int   = 1000
    # Curriculum: phase-1 iterations with lam_ic=0 before interface coupling.
    warm_ic_iters: int  = 500
    n_pde:        int   = 600       # collocation pts per subdomain
    n_bc:         int   = 200       # outer boundary pts
    n_ic:         int   = 200       # interface pts
    seed:         int   = 42
    device:       str   = "cpu"

    def __post_init__(self):
        if self.net  is None: self.net  = NetworkConfig()
        if self.loss is None: self.loss = LossConfig()


# ── sampling helpers ──────────────────────────────────────────────────────────

def _sample_interior(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples inside the unit hypercube [0,1]^dim."""
    return rng.uniform(0, 1, (n, dim)).astype(np.float32)


def _sample_boundary(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on the boundary of [0,1]^dim."""
    if dim == 2:
        k = n // 4
        t = rng.uniform(0, 1, k).astype(np.float32)
        return np.vstack([
            np.column_stack([t,           np.zeros(k)]),
            np.column_stack([t,           np.ones(k)]),
            np.column_stack([np.zeros(k), t]),
            np.column_stack([np.ones(k),  t]),
        ])
    else:  # dim == 3: 6 faces of unit cube
        k = max(n // 6, 1)
        t = rng.uniform(0, 1, (k, 2)).astype(np.float32)
        return np.vstack([
            np.column_stack([np.zeros((k, 1)), t]),                           # x=0
            np.column_stack([np.ones((k, 1)),  t]),                           # x=1
            np.column_stack([t[:, :1], np.zeros((k, 1)), t[:, 1:]]),          # y=0
            np.column_stack([t[:, :1], np.ones((k, 1)),  t[:, 1:]]),          # y=1
            np.column_stack([t, np.zeros((k, 1))]),                           # z=0
            np.column_stack([t, np.ones((k, 1))]),                            # z=1
        ])


def _sample_domain(n: int, dim: int, rng: np.random.Generator, mask_fn) -> np.ndarray:
    """Sample n points inside [0,1]^dim that satisfy mask_fn."""
    pts = []
    while sum(len(p) for p in pts) < n:
        cand = _sample_interior(n * 4, dim, rng)
        pts.append(cand[mask_fn(cand)])
    return np.vstack(pts)[:n].astype(np.float32)


def _to(arr: np.ndarray, device: str, grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if grad:
        t.requires_grad_(True)
    return t


# ── warm-start ────────────────────────────────────────────────────────────────

def warm_start(
    net: MLP,
    x_data: np.ndarray,   # (N, dim) SSN DOF coords
    y_data: np.ndarray,   # (N,)     SSN solution values
    max_iter: int = 1000,
    device: str = "cpu",
) -> float:
    """Regress net on SSN data via L-BFGS. Returns final MSE."""
    net.to(device)
    x = _to(x_data, device)
    y = _to(y_data, device)

    opt = LBFGS(net.parameters(), lr=1.0, max_iter=max_iter,
                tolerance_grad=1e-8, tolerance_change=1e-12,
                line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        final_mse = float(((net(x) - y) ** 2).mean())
    return final_mse


# ── joint training ────────────────────────────────────────────────────────────

def train(
    ssn_result,           # SSNResult (loaded from .npz)
    cfg: TrainConfig = None,
) -> tuple[MLP, MLP, dict]:
    """
    Full pipeline: warm-start both networks, then jointly optimise.

    Parameters
    ----------
    ssn_result : SSNResult
    cfg        : TrainConfig (defaults if None)

    Returns
    -------
    net_A, net_B : trained MLP instances
    info         : dict with loss history, timing, warm-start errors
    """
    if cfg is None:
        cfg = TrainConfig()

    device = cfg.device
    rng    = np.random.default_rng(cfg.seed)

    from ssn.extractor import extract_free_boundary

    coords    = ssn_result.dof_coords
    y_star    = ssn_result.y_star
    u_star    = ssn_result.u_star
    chi_A_np  = ssn_result.chi_A
    bdy_pts   = ssn_result.boundary_pts
    dim       = coords.shape[1]

    # Partition SSN data by subdomain
    inner_mask = chi_A_np > 0.5
    x_A_data   = coords[inner_mask].astype(np.float32)
    y_A_data   = y_star[inner_mask].astype(np.float32)
    u_A_data   = u_star[inner_mask].astype(np.float32)   # exact u* at Ω_A DOFs
    x_B_data   = coords[~inner_mask].astype(np.float32)
    y_B_data   = y_star[~inner_mask].astype(np.float32)

    # Build mask_inner_fn from extractor (dim-aware in extractor.py)
    _, mask_inner_fn = extract_free_boundary(chi_A_np, coords)

    # ── Networks ──────────────────────────────────────────────────────────
    # net_B 覆盖 Ω_B（接触 ∂Ω），enforce_bc=True 硬编码 y=0 on ∂Ω
    # net_A 覆盖 Ω_A（内部激活区，不接触 ∂Ω），无需 enforce_bc
    _base = dict(input_dim=dim, hidden_dim=cfg.net.hidden_dim,
                 n_layers=cfg.net.n_layers, activation=cfg.net.activation)
    net_A = MLP(NetworkConfig(**_base, enforce_bc=False)).to(device)
    net_B = MLP(NetworkConfig(**_base, enforce_bc=True )).to(device)

    # ── Warm-start ────────────────────────────────────────────────────────
    mse_A = warm_start(net_A, x_A_data, y_A_data, cfg.warm_iters, device)
    mse_B = warm_start(net_B, x_B_data, y_B_data, cfg.warm_iters, device)
    print(f"  Warm-start MSE: A={mse_A:.3e}  B={mse_B:.3e}")

    # ── Collocation sets (fixed) ───────────────────────────────────────────
    # Ω_A: use SSN DOF coords directly — u* is exact at these points,
    # eliminating the griddata interpolation noise near Γ that forced lam_pde=0.
    if len(x_A_data) > cfg.n_pde:
        idx      = rng.choice(len(x_A_data), cfg.n_pde, replace=False)
        pde_A_np = x_A_data[idx]
        f_A_np   = u_A_data[idx]
    else:
        pde_A_np = x_A_data
        f_A_np   = u_A_data
    pde_B_np = _sample_domain(cfg.n_pde, dim, rng, lambda p: ~mask_inner_fn(p))
    bc_np    = _sample_boundary(cfg.n_bc, dim, rng)

    # BC partition: each network only penalised at its own boundary slice
    inner_bc  = mask_inner_fn(bc_np)
    bc_A_np   = bc_np[inner_bc]
    bc_B_np   = bc_np[~inner_bc]

    # Interface points
    if len(bdy_pts) >= cfg.n_ic:
        ic_np = bdy_pts[rng.choice(len(bdy_pts), cfg.n_ic, replace=False)]
    elif len(bdy_pts) > 0:
        ic_np = bdy_pts
    else:
        # Fallback: circle/sphere of radius 0.15 around centre
        if dim == 2:
            theta = np.linspace(0, 2 * np.pi, cfg.n_ic, endpoint=False)
            ic_np = np.column_stack([0.5 + 0.15 * np.cos(theta),
                                     0.5 + 0.15 * np.sin(theta)]).astype(np.float32)
        else:
            phi   = np.random.uniform(0, np.pi,     cfg.n_ic).astype(np.float32)
            theta = np.random.uniform(0, 2*np.pi,   cfg.n_ic).astype(np.float32)
            r     = 0.15
            ic_np = np.column_stack([
                0.5 + r * np.sin(phi) * np.cos(theta),
                0.5 + r * np.sin(phi) * np.sin(theta),
                0.5 + r * np.cos(phi),
            ]).astype(np.float32)

    ic_np  = ic_np.astype(np.float32)
    f_A_np = f_A_np.astype(np.float32)

    # Convert to tensors (gradient-enabled where needed)
    pde_A = _to(pde_A_np, device, grad=True)
    pde_B = _to(pde_B_np, device, grad=True)
    bc_A  = _to(bc_A_np,  device)
    bc_B  = _to(bc_B_np,  device)
    ic    = _to(ic_np,    device, grad=True)
    f_A   = _to(f_A_np,   device)

    # Anchor targets: warm-start predictions at the same collocation points (detached)
    with torch.no_grad():
        anchor_A_target = net_A(_to(pde_A_np, device)).detach()
        anchor_B_target = net_B(_to(pde_B_np, device)).detach()

    # ── Joint optimisation ────────────────────────────────────────────────
    all_params = list(net_A.parameters()) + list(net_B.parameters())
    loss_history: list[float] = []
    t0 = time.perf_counter()

    # _active_cfg[0] switches from warm-in config (lam_ic=0) to full config.
    _active_cfg = [replace(cfg.loss, lam_ic=0.0) if cfg.warm_ic_iters > 0
                   else cfg.loss]

    def closure():
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()
        loss, breakdown = total_loss(
            net_A, net_B, _active_cfg[0],
            pde_A_pts=pde_A, f_A=f_A,
            pde_B_pts=pde_B,
            bc_A_pts=bc_A,
            bc_B_pts=bc_B,
            ic_pts=ic,
            anchor_A_pts=pde_A, anchor_A_target=anchor_A_target,
            anchor_B_pts=pde_B, anchor_B_target=anchor_B_target,
        )
        loss.backward()
        loss_history.append(breakdown["total"])
        return loss

    _lbfgs_kw = dict(lr=cfg.lr, tolerance_grad=1e-8, tolerance_change=1e-15,
                     line_search_fn="strong_wolfe")

    if cfg.optimizer == "lbfgs":
        # Phase 1: warm-in with lam_ic = 0
        if cfg.warm_ic_iters > 0:
            opt1 = LBFGS(all_params, max_iter=cfg.warm_ic_iters, **_lbfgs_kw)
            opt1.step(closure)
            print(f"  Phase 1 (lam_ic=0): {len(loss_history)} evals, "
                  f"loss={loss_history[-1]:.3e}")
            _active_cfg[0] = cfg.loss   # switch to full lam_ic

        # Phase 2: full lam_ic
        opt2 = LBFGS(all_params, max_iter=cfg.max_iter, **_lbfgs_kw)
        opt2.step(closure)
    else:
        opt = Adam(all_params, lr=cfg.lr)
        total_steps = cfg.warm_ic_iters + cfg.max_iter
        for step in range(total_steps):
            if step == cfg.warm_ic_iters:
                _active_cfg[0] = cfg.loss   # switch to full lam_ic
                print(f"  Phase 2 (full lam_ic={cfg.loss.lam_ic}): "
                      f"step {step}/{total_steps}")
            opt.zero_grad()
            loss, _ = total_loss(
                net_A, net_B, _active_cfg[0],
                pde_A_pts=pde_A, f_A=f_A,
                pde_B_pts=pde_B,
                bc_A_pts=bc_A, bc_B_pts=bc_B, ic_pts=ic,
                anchor_A_pts=pde_A, anchor_A_target=anchor_A_target,
                anchor_B_pts=pde_B, anchor_B_target=anchor_B_target,
            )
            loss.backward()
            opt.step()
            loss_history.append(float(loss))

    wall_time = time.perf_counter() - t0
    print(f"  XPINN training: {len(loss_history)} steps, "
          f"final loss={loss_history[-1]:.3e} ({wall_time:.1f}s)")

    return net_A, net_B, {
        "loss_history":  loss_history,
        "warm_mse_A":    mse_A,
        "warm_mse_B":    mse_B,
        "wall_time":     wall_time,
        "mask_inner_fn": mask_inner_fn,
        "dim":           dim,
    }
