"""
Pack SSN solver output into an SSNResult and save to .npz.
"""

from __future__ import annotations

from interface.ssn_result import SSNResult
from .extractor import extract_free_boundary
from .solver import SSNSolver
from .amr_solver import run_amr_ssn


def _pack_and_save(
    solver: SSNSolver,
    output_path: str,
    alpha_scalar: float,
    verbose: bool,
) -> SSNResult:
    coords, y_arr, u_arr, p_arr, chi_A_vals = solver.get_dof_arrays()
    boundary_pts, _ = extract_free_boundary(chi_A_vals, coords)

    result = SSNResult(
        dof_coords=coords,
        y_star=y_arr,
        u_star=u_arr,
        p_star=p_arr,
        chi_A=chi_A_vals,
        boundary_pts=boundary_pts,
        nu=solver.nu,
        alpha=alpha_scalar,
        mesh_n=solver.n,
        ssn_iters=0,  # AMR: iters counted per level; set 0 here as aggregate
    )
    result.save(output_path)
    if verbose:
        print(result.summary())
        print(f"  Saved → {output_path}")
    return result


def run_and_export(
    solver: SSNSolver,
    output_path: str,
    alpha_scalar: float,
    verbose: bool = True,
) -> SSNResult:
    """
    Solve SSN on a pre-configured solver, pack result, save to .npz.

    Parameters
    ----------
    solver       : configured SSNSolver (not yet solved)
    output_path  : path for the .npz file
    alpha_scalar : scalar α for metadata
    """
    info = solver.solve(verbose=verbose)
    coords, y_arr, u_arr, p_arr, chi_A_vals = solver.get_dof_arrays()
    boundary_pts, _ = extract_free_boundary(chi_A_vals, coords)

    result = SSNResult(
        dof_coords=coords,
        y_star=y_arr,
        u_star=u_arr,
        p_star=p_arr,
        chi_A=chi_A_vals,
        boundary_pts=boundary_pts,
        nu=solver.nu,
        alpha=alpha_scalar,
        mesh_n=solver.n,
        ssn_iters=info["n_iter"],
    )
    result.save(output_path)
    if verbose:
        print(result.summary())
        print(f"  Saved → {output_path}")
    return result


def run_amr_and_export(
    n_coarse: int,
    nu: float,
    alpha_expr,
    yd_expr,
    output_path: str,
    alpha_scalar: float,
    max_levels: int = 4,
    theta: float = 0.8,
    eta_tol: float = 5e-3,
    solver_type: str = "mumps",
    dim: int = 2,
    verbose: bool = True,
) -> SSNResult:
    """
    Run AMR-SSN loop, pack the final (finest-mesh) solution, save to .npz.

    Parameters
    ----------
    n_coarse    : initial uniform mesh resolution
    max_levels  : maximum refinement levels (0 = no refinement)
    theta       : Dörfler bulk fraction
    eta_tol     : early-stop threshold on max(jump indicator)
    solver_type : "mumps" (default) or "amg"
    dim         : spatial dimension (2 or 3)
    """
    solver = run_amr_ssn(
        n_coarse=n_coarse,
        nu=nu,
        alpha_expr=alpha_expr,
        yd_expr=yd_expr,
        max_levels=max_levels,
        theta=theta,
        eta_tol=eta_tol,
        solver_type=solver_type,
        dim=dim,
        verbose=verbose,
    )
    return _pack_and_save(solver, output_path, alpha_scalar, verbose)
