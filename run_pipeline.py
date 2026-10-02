"""
One-shot pipeline: SSN (optionally AMR) → save .npz → train XPINNs → evaluate.
Requires the custom Docker image (dolfinx + PyTorch).

Usage:
    python3 run_pipeline.py [--mesh 64] [--device cpu]
    python3 run_pipeline.py --amr --mesh 32 --amr-levels 4
    python3 run_pipeline.py --dim 3 --mesh 24 --solver amg
    python3 run_pipeline.py --dim 3 --amr --mesh 16 --amr-levels 3 --solver amg
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh",       type=int,   default=64)
    parser.add_argument("--nu",         type=float, default=1e-3)
    parser.add_argument("--dim",        type=int,   default=2, choices=[2, 3])
    parser.add_argument("--solver",     default="mumps", choices=["mumps", "amg"])
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--npz",        default="ssn_result.npz")
    parser.add_argument("--amr",        action="store_true")
    parser.add_argument("--amr-levels", type=int,   default=4)
    parser.add_argument("--amr-theta",  type=float, default=0.8)
    parser.add_argument("--amr-eta",    type=float, default=5e-3)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    print("\n[1/2] Running SSN solver...")
    ssn_cmd = [
        sys.executable, os.path.join(base, "run_ssn.py"),
        "--output", args.npz,
        "--mesh",   str(args.mesh),
        "--nu",     str(args.nu),
        "--dim",    str(args.dim),
        "--solver", args.solver,
    ]
    if args.amr:
        ssn_cmd += [
            "--amr",
            "--amr-levels", str(args.amr_levels),
            "--amr-theta",  str(args.amr_theta),
            "--amr-eta",    str(args.amr_eta),
        ]
    subprocess.run(ssn_cmd, check=True)

    print("\n[2/2] Training XPINNs...")
    subprocess.run(
        [sys.executable, os.path.join(base, "run_xpinn.py"),
         "--input",  args.npz,
         "--device", args.device],
        check=True,
    )


if __name__ == "__main__":
    main()
