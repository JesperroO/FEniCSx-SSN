"""
Entry point: run SSN solver and export result to .npz.

Modes
-----
  2D uniform (default):
    python3 run_ssn.py [--mesh 64] [--output ssn_result.npz]

  2D AMR:
    python3 run_ssn.py --amr [--mesh 32] [--amr-levels 4]

  3D with AMG (recommended):
    python3 run_ssn.py --dim 3 --mesh 24 --solver amg

Usage (inside dolfinx container or custom image):
    docker run --rm -v $(pwd):/workspace -w /workspace ssn-xpinn \\
        python3 run_ssn.py --dim 3 --mesh 24 --solver amg
"""

import argparse
import ufl
from ssn.solver import SSNSolver
from ssn.export import run_and_export, run_amr_and_export

# ── 2D problem: DIAMOND obstacle, two-peak target ─────────────────────────────
def alpha_diamond(x):
    return 0.02 + 0.10 * ufl.max_value(
        1.0 - 4.0 * (abs(x[0] - 0.5) + abs(x[1] - 0.5)), 0.0
    )

def yd_two_peaks(x):
    return (
        5.0 * ufl.exp(-50.0 * ((x[0] - 0.3)**2 + (x[1] - 0.3)**2))
      + 5.0 * ufl.exp(-50.0 * ((x[0] - 0.7)**2 + (x[1] - 0.7)**2))
    )

# ── 3D problem: centre Gaussian target ───────────────────────────────────────
# Uses a broader Gaussian (σ²=1/20 vs 1/160) so that the adjoint p_max
# exceeds alpha.  On the 3D unit cube, λ_min(−Δ+I)≈30, so p_max≈‖yd‖/30.
# With yd_max=8 and width=10: ‖yd‖_L2≈3.4, p_max≈0.11 > alpha=0.01.
def alpha_const(x):
    return ufl.as_ufl(0.05)

def yd_centre_gaussian(x):
    return 8.0 * ufl.exp(
        -10.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + (x[2] - 0.5)**2)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     default="ssn_result.npz")
    parser.add_argument("--mesh",       type=int,   default=64,
                        help="mesh resolution per dim (uniform) or coarse mesh (AMR)")
    parser.add_argument("--nu",         type=float, default=1e-3)
    parser.add_argument("--dim",        type=int,   default=2, choices=[2, 3],
                        help="spatial dimension (default 2)")
    parser.add_argument("--amr",        action="store_true",
                        help="enable AMR (2D only)")
    parser.add_argument("--amr-levels", type=int,   default=4)
    parser.add_argument("--amr-theta",  type=float, default=0.8)
    parser.add_argument("--amr-eta",    type=float, default=5e-3)
    parser.add_argument("--solver",     default="mumps", choices=["mumps", "amg"],
                        help="linear solver: mumps (default) or amg")
    args = parser.parse_args()

    if args.dim == 3:
        alpha_expr, yd_expr, alpha_scalar = alpha_const, yd_centre_gaussian, 0.05  # alpha=0.05, broad Gaussian
    else:
        alpha_expr, yd_expr, alpha_scalar = alpha_diamond, yd_two_peaks, 0.02

    dim_str = f"{args.mesh}³" if args.dim == 3 else f"{args.mesh}×{args.mesh}"
    print("=" * 60)
    if args.amr:
        print(f"  SSN+AMR  coarse={dim_str}  levels={args.amr_levels}"
              f"  solver={args.solver}  ν={args.nu}")
    else:
        print(f"  SSN solver  {args.dim}D  mesh={dim_str}"
              f"  solver={args.solver}  ν={args.nu}")
    print("=" * 60)

    if args.amr:
        run_amr_and_export(
            n_coarse=args.mesh,
            nu=args.nu,
            alpha_expr=alpha_expr,
            yd_expr=yd_expr,
            output_path=args.output,
            alpha_scalar=alpha_scalar,
            max_levels=args.amr_levels,
            theta=args.amr_theta,
            eta_tol=args.amr_eta,
            solver_type=args.solver,
            dim=args.dim,
        )
    else:
        solver = SSNSolver(
            n=args.mesh,
            nu=args.nu,
            alpha_expr=alpha_expr,
            yd_expr=yd_expr,
            solver_type=args.solver,
            dim=args.dim,
        )
        run_and_export(solver, args.output, alpha_scalar=alpha_scalar)


if __name__ == "__main__":
    main()
