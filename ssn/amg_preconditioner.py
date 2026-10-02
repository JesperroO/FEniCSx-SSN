"""
AMG Block Preconditioner for SSN Saddle-Point System
=====================================================
Replaces the MUMPS direct solver with a GMRES + PCFieldSplit iterative
solver whose diagonal blocks use HYPRE BoomerAMG or Jacobi, achieving
near-O(N) complexity for large-scale 3-D problems.

System structure (mixed P1³ space  W = V_y × V_u × V_p):

    | K_y   -M    0    |   | δy |   | r_y |
    |  0     M   -D/ν  | × | δu | = | r_u |
    |  M     0    K_p  |   | δp |   | r_p |

where K_y = K_p = (-Δ + I)  (SPD Helmholtz, amenable to AMG)
      M                      (mass matrix, SPD)
      D = diag(χ_A)          (diagonal indicator, Jacobi-invertible)

Preconditioner: ADDITIVE block-diagonal (block-Jacobi)
    P = diag(AMG(K_y),  Jacobi(M_uu),  AMG(K_p))

The off-diagonal coupling is handled by GMRES outer iteration.
"""

import numpy as np
from petsc4py import PETSc


def build_amg_ksp(A_mat, W, comm,
                  rtol: float = 1e-8,
                  max_it: int = 300,
                  gmres_restart: int = 60):
    """
    Build a GMRES solver with 3×3 block-Jacobi AMG preconditioner.

    Parameters
    ----------
    A_mat    : assembled PETSc Mat (the full Jacobian of the SSN system)
    W        : dolfinx MixedFunctionSpace (P1 × P1 × P1)
    comm     : MPI communicator
    rtol     : relative tolerance for GMRES inner solver
    max_it   : maximum GMRES iterations before restart
    gmres_restart : Krylov subspace dimension (GMRES restart)

    Returns
    -------
    ksp : PETSc.KSP ready to use; caller is responsible for destroy()
    """
    # ── DOF index sets for each sub-space ─────────────────────────────────────
    # W.sub(i).collapse() -> (SubFunctionSpace, dof_map_list)
    # dof_map contains local indices into the mixed space vector
    _, dofs_y = W.sub(0).collapse()
    _, dofs_u = W.sub(1).collapse()
    _, dofs_p = W.sub(2).collapse()

    is_y = PETSc.IS().createGeneral(np.asarray(dofs_y, dtype=np.int32), comm=comm)
    is_u = PETSc.IS().createGeneral(np.asarray(dofs_u, dtype=np.int32), comm=comm)
    is_p = PETSc.IS().createGeneral(np.asarray(dofs_p, dtype=np.int32), comm=comm)

    # ── KSP: restarted GMRES ──────────────────────────────────────────────────
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_mat)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=max_it)
    ksp.setGMRESRestart(gmres_restart)

    # ── PC: 3×3 ADDITIVE (block-Jacobi) ──────────────────────────────────────
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    pc.setFieldSplitIS(("y", is_y), ("u", is_u), ("p", is_p))
    # ADDITIVE = independent block-diagonal inversion (no Gauss-Seidel sweeps)
    # Cheaper per iteration; off-diagonal coupling handled by GMRES
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # setUp() triggers sub-KSP creation — must precede getFieldSplitSubKSP()
    ksp.setUp()

    sub_ksps = pc.getFieldSplitSubKSP()  # list of 3 KSP objects

    # ── Sub-preconditioner 0: state y  (SPD Helmholtz → BoomerAMG) ───────────
    sub_ksps[0].setType(PETSc.KSP.Type.PREONLY)
    pc_y = sub_ksps[0].getPC()
    pc_y.setType(PETSc.PC.Type.HYPRE)
    pc_y.setHYPREType("boomeramg")

    # ── Sub-preconditioner 1: control u  (diagonal mass → Jacobi) ────────────
    # χ_A mass matrix is diagonal; one Jacobi sweep is exact when χ_A ∈ {0,1}
    sub_ksps[1].setType(PETSc.KSP.Type.PREONLY)
    sub_ksps[1].getPC().setType(PETSc.PC.Type.JACOBI)

    # ── Sub-preconditioner 2: adjoint p  (SPD Helmholtz → BoomerAMG) ─────────
    sub_ksps[2].setType(PETSc.KSP.Type.PREONLY)
    pc_p = sub_ksps[2].getPC()
    pc_p.setType(PETSc.PC.Type.HYPRE)
    pc_p.setHYPREType("boomeramg")

    is_y.destroy()
    is_u.destroy()
    is_p.destroy()

    return ksp


def build_mumps_ksp(A_mat, comm):
    """
    Reference MUMPS direct solver (identical to existing SSN solvers).
    Used as the baseline for performance comparison.
    """
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    return ksp
