"""
Loss functions and LossConfig for XPINN training.

Per-subdomain variational energy formulation:
  Ω_A (active):   E_A = ½∫_A(|∇y|²+y²−u*·y)dx   — 一阶 autograd，u* 在 A 内插值可靠
  Ω_B (inactive): E_B = ½∫_B(|∇y|²+y²)dx          — u*≡0 精确，无需插值

界面条件（y 和 ∂y/∂n 均连续，因为 RHS 不连续不引起梯度跳跃）:
  L_ic = ‖y_A−y_B‖² + ‖∇y_A−∇y_B‖²  at Γ

Loss = λ_pde·(E_A+E_B) + λ_bc·L_bc + λ_ic·L_ic + λ_reg·L_reg
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LossConfig:
    lam_pde: float = 0.0    # Ω_A 变分能量：与 lam_reg 锚点梯度冲突，实验确认单调恶化精度
    lam_bc:  float = 10.0   # Dirichlet BC（net_B 已 enforce_bc 硬编码，此项约束 net_A）
    lam_ic:  float = 10.0   # interface continuity
    lam_reg: float = 50.0   # warm-start anchor（主导项）


# ── helpers ───────────────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂y/∂x via autograd. x must have requires_grad=True."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


# ── per-subdomain variational energies ────────────────────────────────────────

def pde_loss_A(
    net: nn.Module,
    pts: torch.Tensor,    # (N, dim) Ω_A 内采样点，requires_grad=True
    u_star: torch.Tensor, # (N,)     SSN 提供的 u*（在 A 内部，插值可靠）
) -> torch.Tensor:
    """
    Ω_A 变分能量：E_A = mean(½|∇y|² + ½y² − u*·y)
    最小值点满足 (−Δ+I)y = u*  in Ω_A
    只需一阶 autograd，无 Laplacian。
    """
    y  = net(pts)
    gy = _grad(y, pts)
    return (0.5 * (gy ** 2).sum(dim=-1) + 0.5 * y ** 2 - u_star * y).mean()


def pde_loss_B(
    net: nn.Module,
    pts: torch.Tensor,    # (N, dim) Ω_B 内采样点，requires_grad=True
) -> torch.Tensor:
    """
    Ω_B 变分能量：E_B = mean(½|∇y|² + ½y²)
    最小值点满足 (−Δ+I)y = 0  in Ω_B
    u*≡0 精确，完全不依赖插值。
    """
    y  = net(pts)
    gy = _grad(y, pts)
    return (0.5 * (gy ** 2).sum(dim=-1) + 0.5 * y ** 2).mean()


# ── boundary / interface / anchor ─────────────────────────────────────────────

def bc_loss(
    net: nn.Module,
    bc_pts: torch.Tensor,  # (M, dim) 该网络负责的边界点
) -> torch.Tensor:
    """Dirichlet y=0 on ∂Ω，仅在本网络负责的边界段上惩罚。"""
    if bc_pts.shape[0] == 0:
        return torch.tensor(0.0, device=bc_pts.device)
    return (net(bc_pts) ** 2).mean()


def interface_loss(
    net_A: nn.Module,
    net_B: nn.Module,
    ic_pts: torch.Tensor,  # (K, dim) Γ 上的点，requires_grad=True
) -> torch.Tensor:
    """
    界面连续性：‖y_A−y_B‖² + ‖∇y_A−∇y_B‖²
    y* 和 ∂y*/∂n 在 Γ 两侧均连续（RHS 的跳跃不引起梯度跳跃）。
    """
    yA  = net_A(ic_pts)
    yB  = net_B(ic_pts)
    gyA = _grad(yA, ic_pts)
    gyB = _grad(yB, ic_pts)
    return ((yA - yB) ** 2).mean() + ((gyA - gyB) ** 2).mean()


def anchor_loss(
    net: nn.Module,
    pts: torch.Tensor,
    target: torch.Tensor,  # warm-start 预测值（detached 常量）
) -> torch.Tensor:
    """Warm-start 锚点：‖y−ŷ‖²，辅助初期稳定，不再主导。"""
    return ((net(pts) - target) ** 2).mean()


# ── combined loss ──────────────────────────────────────────────────────────────

def total_loss(
    net_A: nn.Module,
    net_B: nn.Module,
    cfg: LossConfig,
    *,
    pde_A_pts: torch.Tensor, f_A: torch.Tensor,
    pde_B_pts: torch.Tensor,
    bc_A_pts:  torch.Tensor,
    bc_B_pts:  torch.Tensor,
    ic_pts:    torch.Tensor,
    anchor_A_pts: torch.Tensor, anchor_A_target: torch.Tensor,
    anchor_B_pts: torch.Tensor, anchor_B_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    总损失 = λ_pde·(E_A+E_B) + λ_bc·L_bc + λ_ic·L_ic + λ_reg·L_reg

    注意 f_B 参数已移除：Ω_B 的能量不需要 u*。
    """
    # 只对 net_A 施加变分能量：E_A 的唯一最小值是 y*_A（线性项 -u*y 使 y=0 不是驻点）
    # E_B 不使用：½∫(|∇y|²+y²)dx 的最小值是 y≡0，与界面约束竞争，会把 net_B 压向零
    L_pde = (
        pde_loss_A(net_A, pde_A_pts, f_A)
    ) if cfg.lam_pde > 0 else torch.tensor(0.0)

    L_bc = bc_loss(net_A, bc_A_pts) + bc_loss(net_B, bc_B_pts)

    L_ic = interface_loss(net_A, net_B, ic_pts)

    L_reg = (
        anchor_loss(net_A, anchor_A_pts, anchor_A_target) +
        anchor_loss(net_B, anchor_B_pts, anchor_B_target)
    ) if cfg.lam_reg > 0 else torch.tensor(0.0)

    loss = (
        cfg.lam_pde * L_pde +
        cfg.lam_bc  * L_bc  +
        cfg.lam_ic  * L_ic  +
        cfg.lam_reg * L_reg
    )

    return loss, {
        "L_pde": L_pde.detach().item(),
        "L_bc":  L_bc.detach().item(),
        "L_ic":  L_ic.detach().item(),
        "L_reg": L_reg.detach().item(),
        "total": float(loss),
    }
