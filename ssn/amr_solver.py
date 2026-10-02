"""
AMR-SSN: Adaptive Mesh Refinement loop driving the SSN solver.

Each level:
  1. Run SSNSolver on current mesh → converged (y*, u*, p*)
  2. Compute gradient-jump indicator from adjoint p*
  3. Dörfler marking (theta) → local refinement → new mesh
  4. Repeat until max_levels or max(eta) < eta_tol

The final (finest) solver is returned; its get_dof_arrays() gives the
high-quality solution for downstream XPINN training.
"""

from __future__ import annotations
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from .amr_engine import compute_jump_indicator, dorfer_mark, amr_refine
from .solver import SSNSolver


def run_amr_ssn(
    n_coarse: int,
    nu: float,
    alpha_expr,
    yd_expr,
    max_levels: int = 4,
    theta: float = 0.8,
    eta_tol: float = 5e-3,
    max_iter: int = 40,
    tol: float = 1e-10,
    solver_type: str = "mumps",
    dim: int = 2,
    verbose: bool = True,
) -> SSNSolver:
    """
    Run the AMR-SSN loop and return the final solved SSNSolver.

    Parameters
    ----------
    n_coarse    : initial uniform mesh resolution (n × n or n × n × n)
    nu          : Tikhonov regularisation weight
    alpha_expr  : callable(x_ufl) → UFL expression for α(x)
    yd_expr     : callable(x_ufl) → UFL expression for y_d(x)
    max_levels  : maximum AMR refinement levels (0 = no refinement, uniform only)
    theta       : Dörfler bulk fraction (0.7–0.9)
    eta_tol     : stop early if max(eta) < eta_tol
    max_iter    : SSN Newton iteration limit per level
    tol         : SSN residual tolerance
    dim         : spatial dimension (2 or 3)
    verbose     : print per-level progress
    """
    comm = MPI.COMM_WORLD

    if dim == 3:
        domain = dmesh.create_unit_cube(
            comm, n_coarse, n_coarse, n_coarse, dmesh.CellType.tetrahedron
        )
    else:
        domain = dmesh.create_unit_square(
            comm, n_coarse, n_coarse, dmesh.CellType.triangle
        )

    solver = None
    for level in range(max_levels + 1):
        n_cells = domain.topology.index_map(domain.topology.dim).size_global
        if verbose and comm.rank == 0:
            print(f"\n── AMR level {level}  (cells={n_cells}) ──")

        solver = SSNSolver(
            n=n_coarse,
            nu=nu,
            alpha_expr=alpha_expr,
            yd_expr=yd_expr,
            max_iter=max_iter,
            tol=tol,
            domain=domain,
            solver_type=solver_type,
        )
        solver.solve(verbose=verbose)

        if level == max_levels:
            break

        p_h = solver.get_adjoint_func()
        eta_vals, _ = compute_jump_indicator(p_h, domain)

        max_eta = float(np.max(eta_vals)) if len(eta_vals) > 0 else 0.0
        if verbose and comm.rank == 0:
            print(f"  jump indicator: max={max_eta:.3e}  sum={eta_vals.sum():.3e}")

        if max_eta < eta_tol:
            if verbose and comm.rank == 0:
                print(f"  eta_tol reached — stopping AMR early at level {level}")
            break

        marked = dorfer_mark(eta_vals, theta=theta)
        if verbose and comm.rank == 0:
            print(f"  marked {len(marked)}/{n_cells} cells for refinement")

        domain, _ = amr_refine(domain, marked)

    return solver
