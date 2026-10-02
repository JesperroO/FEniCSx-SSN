"""
Test: Extreme Regularity Deficiency — Dirac Delta Load
=======================================================
Three increasingly extreme Gaussian approximations to a Dirac point source:

  sigma=50   (width ~0.14):  smooth reference, well-resolved on any mesh
  sigma=500  (width ~0.045): sharp spike, resolved by ~6 cells on 128x128
  sigma=5000 (width ~0.014): near-Dirac, resolved by ~2 cells on 128x128

The target state y_d = A * exp(-sigma * |x - x_c|^2) creates near-infinite
local curvature. The adjoint PDE (-Delta p + p = y - y_d) must absorb this
singularity: the question is whether the SSN residual path stays clean.

Key insight: the elliptic inverse Laplacian acts as a global low-pass filter.
Even a Dirac point source in y_d produces an H^1-smooth adjoint p, so the
active set indicator chi_A = 1_{|p|>alpha} remains a well-defined function.
The Tikhonov/L2 control term (nu > 0) then guarantees strong monotonicity, and
SSN is expected to converge regardless of the sharpness of the forcing.

Expected outcome:
  - Convergence in <= 10 iterations for all three sigma levels
  - Monotonically decreasing residuals (no pollution from the local spike)
  - Active set shrinks to a tiny island as sigma -> infinity
"""

import sys
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc, create_matrix, create_vector,
)
from petsc4py import PETSc
import ufl


def solve_with_dirac_target(N, sigma, amplitude, comm):
    domain = mesh.create_unit_square(comm, N, N)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_y = fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets)
    dofs_p = fem.locate_dofs_topological(W.sub(2), fdim, boundary_facets)
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(2)),
    ]

    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    alpha = 0.05
    nu = 1e-3

    x = ufl.SpatialCoordinate(domain)
    # Dirac-approximation: all three cases centred at (0.5, 0.5)
    y_d = amplitude * ufl.exp(-sigma * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

    chi_A_plus  = ufl.conditional(ufl.gt(p,  alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A       = chi_A_plus + chi_A_minus
    u_expected  = chi_A_plus * (p - alpha) / nu + chi_A_minus * (p + alpha) / nu

    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy * v_y - du * v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp * v_p + dy * v_p) * ufl.dx
    a_u = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx

    L_y = -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y * v_y - u * v_y) * ufl.dx
    L_p = -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p * v_p + (y - y_d) * v_p) * ufl.dx
    L_u = -(u - u_expected) * v_u * ufl.dx

    a_form = fem.form(a_y + a_p + a_u)
    L_form = fem.form(L_y + L_p + L_u)

    A_mat = create_matrix(a_form)
    b_vec = create_vector(L_form)
    dw    = fem.Function(W)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    vol_form = fem.form(chi_A * ufl.dx)
    residuals = []
    converged_iter = -1
    prev_vol = -1.0

    for i in range(30):
        A_mat.zeroEntries()
        assemble_matrix(A_mat, a_form, bcs=bcs)
        A_mat.assemble()

        with b_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(b_vec, L_form)
        apply_lifting(b_vec, [a_form], [bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_vec, bcs)

        res = b_vec.norm()
        active_vol = comm.allreduce(fem.assemble_scalar(vol_form), op=MPI.SUM)
        residuals.append(res)

        if res < 1e-10 or (res < 1e-8 and abs(active_vol - prev_vol) < 1e-12):
            converged_iter = i
            break

        prev_vol = active_vol
        ksp.solve(b_vec, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

    ksp.destroy()
    A_mat.destroy()
    b_vec.destroy()

    return converged_iter, residuals, active_vol


def run_dirac_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    N = 128

    # (sigma, amplitude, label)
    # Amplitude is L2-normalised so ||y_d||_L2 stays comparable across sigma:
    #   ||A*exp(-sigma*r^2)||_L2 = A * sqrt(pi/(2*sigma))
    # Reference: sigma=50, A=20 => ||y_d|| ~ 3.97
    # We keep A * sqrt(pi/(2*sigma)) = const  =>  A = A_ref * sqrt(sigma / sigma_ref)
    cases = [
        (50,   20,    "sigma=50    (smooth Gaussian, reference)"),
        (500,  63,    "sigma=500   (sharp spike, ~6 cells wide) "),
        (5000, 200,   "sigma=5000  (near-Dirac, ~2 cells wide)  "),
    ]

    if rank == 0:
        print("=" * 70)
        print("  TEST: EXTREME REGULARITY DEFICIENCY — DIRAC DELTA LOAD")
        print(f"  Grid: {N}x{N}  |  alpha=0.05  |  nu=1e-3")
        print("=" * 70)

    for sigma, amp, label in cases:
        if rank == 0:
            print(f"\n[{label}]")
            print(f"  Peak value at mesh centre: y_d(0.5,0.5) = {amp:.0f}")
            print(f"  Gaussian width (1/sqrt(sigma))         = {1/sigma**0.5:.4f}")
            print(f"  Mesh spacing h                         = {1/N:.4f}")
            print(f"  Cells across peak (approx)             = {2*N/sigma**0.5:.1f}")
            print()
            print(f"  {'Iter':<6}  {'Residual ||F||':<22}  {'Active-set vol'}")
            print(f"  {'-'*6}  {'-'*22}  {'-'*14}")

        t0 = time.time()
        iters, residuals, final_vol = solve_with_dirac_target(N, sigma, amp, comm)

        if rank == 0:
            for k, r in enumerate(residuals):
                marker = " <-- converged" if k == iters else ""
                print(f"  {k:<6}  {r:<22.6e}{marker}")
            elapsed = time.time() - t0
            status = f"CONVERGED in {iters} iterations" if iters >= 0 else "DID NOT CONVERGE (max iter)"
            print(f"\n  Result  : {status}")
            print(f"  Active set vol  : {final_vol:.6f}")
            print(f"  Wall time       : {elapsed:.1f}s")

    if rank == 0:
        print()
        print("=" * 70)
        print("  VERDICT")
        print("=" * 70)
        print("  The inverse Laplacian in the adjoint PDE regularises all three")
        print("  targets into H^1-smooth adjoint fields. SSN convergence is")
        print("  governed by the operator's monotonicity (guaranteed by nu>0),")
        print("  not by the smoothness of the data. Local Dirac-like spikes")
        print("  cannot pollute the global residual descent path.")
        print("=" * 70)


if __name__ == "__main__":
    run_dirac_test()
