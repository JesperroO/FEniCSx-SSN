"""
SSNSolver — complete Semismooth Newton solver for sparse L1 optimal control.

Absorbs:
  - UFL form assembly (previously ssn_core_v2.py)
  - Newton iteration loop (previously inlined in each test file)

Implementation notation:
    ν / nu      = Tikhonov-L2 control regularisation weight
    α / alpha   = L1 sparsity threshold / penalty weight

The implemented optimality system uses the soft-threshold control law:
    u = 0                         if |p| <= α(x)
    u = (p - α(x)) / ν            if p >  α(x)
    u = (p + α(x)) / ν            if p < -α(x)

PDAS / SSN equivalence (Hintermüller et al. 2002):
    Active set  A = {x : |p(x)| > α(x)}
    Free boundary Γ = {x : |p(x)| = α(x)}
"""

from __future__ import annotations
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc,
    create_matrix, create_vector,
)
from petsc4py import PETSc
import ufl

from .amg_preconditioner import build_amg_ksp, build_mumps_ksp


class SSNSolver:
    """
    Parameters
    ----------
    n : int
        Mesh resolution (n×n squares, each bisected → 2n² triangles).
    nu : float
        Tikhonov regularisation weight.
    alpha_expr : callable(x_ufl) → UFL expression
        Spatially varying penalty threshold α(x).  Must be expressible in UFL.
    yd_expr : callable(x_ufl) → UFL expression
        Target state y_d(x).
    max_iter : int
        Maximum SSN iterations.
    tol : float
        Convergence tolerance on ‖residual‖_L2.
    """

    def __init__(
        self,
        n: int = 64,
        nu: float = 1e-3,
        alpha_expr=None,
        yd_expr=None,
        max_iter: int = 40,
        tol: float = 1e-10,
        domain=None,
        solver_type: str = "mumps",
        dim: int = 2,
    ):
        self.n           = n
        self.nu          = nu
        self.max_iter    = max_iter
        self.tol         = tol
        self.solver_type = solver_type
        self.dim         = dim
        self.comm        = MPI.COMM_WORLD

        # Build mesh and function spaces
        if domain is None:
            if dim == 3:
                self.domain = dmesh.create_unit_cube(
                    self.comm, n, n, n, dmesh.CellType.tetrahedron
                )
            else:
                self.domain = dmesh.create_unit_square(
                    self.comm, n, n, dmesh.CellType.triangle
                )
        else:
            self.domain = domain
            self.dim    = domain.geometry.dim
        P1  = ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), 1)
        self.W = fem.FunctionSpace(self.domain, ufl.MixedElement([P1, P1, P1]))

        # Boundary conditions: y = 0 and p = 0 on ∂Ω
        fdim = self.domain.topology.dim - 1
        self.domain.topology.create_entities(fdim)
        self.domain.topology.create_connectivity(fdim, self.domain.topology.dim)
        bfacets = dmesh.exterior_facet_indices(self.domain.topology)
        dofs_y  = fem.locate_dofs_topological(self.W.sub(0), fdim, bfacets)
        dofs_p  = fem.locate_dofs_topological(self.W.sub(2), fdim, bfacets)
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(0), dofs_y, self.W.sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0), dofs_p, self.W.sub(2)),
        ]

        # Problem expressions
        x = ufl.SpatialCoordinate(self.domain)
        if alpha_expr is None:
            raise ValueError("alpha_expr is required")
        if yd_expr is None:
            raise ValueError("yd_expr is required")
        self.alpha_ufl = alpha_expr(x)
        self.yd_ufl    = yd_expr(x)

        # Current iterate (initialised to zero)
        self.w_k = fem.Function(self.W)

        # Build and JIT-compile UFL forms once
        self._build_forms()

        # Pre-allocate PETSc objects for in-loop reuse
        self.A  = create_matrix(self.a_form)
        self.b  = create_vector(self.L_form)
        self.dw = fem.Function(self.W)

        # Pre-compute per-DOF alpha values for chi_A in get_dof_arrays().
        # The TestFunction carries the domain even when alpha_ufl is a constant.
        self._alpha_arr = self._project_alpha()

        # Linear solver (swappable: mumps or amg)
        # AMG's setUp() requires an assembled matrix to extract sub-index sets.
        # Do an initial assembly here so the sparsity pattern is established.
        if solver_type == "amg":
            assemble_matrix(self.A, self.a_form, bcs=self.bcs)
            self.A.assemble()
            self.ksp = build_amg_ksp(self.A, self.W, self.comm)
        else:
            self.ksp = build_mumps_ksp(self.A, self.comm)

    # ------------------------------------------------------------------
    def _project_alpha(self) -> np.ndarray:
        """
        Compute per-DOF alpha values via a lumped L2 average.

        For a TestFunction phi_i on V_sub, the DOF-wise alpha estimate is:
            alpha_i = ∫ alpha_ufl * phi_i dx / ∫ phi_i dx

        Using phi inherits the mesh domain, so this works even when
        alpha_ufl is a UFL constant (which has no attached domain).
        """
        V_sub, _ = self.W.sub(0).collapse()
        phi = ufl.TestFunction(V_sub)

        L_num   = fem.form(self.alpha_ufl * phi * ufl.dx)
        L_denom = fem.form(ufl.as_ufl(1.0) * phi * ufl.dx)

        b_num   = create_vector(L_num)
        b_denom = create_vector(L_denom)

        assemble_vector(b_num,   L_num)
        assemble_vector(b_denom, L_denom)
        b_num.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_denom.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        alpha_arr = b_num.array / np.maximum(b_denom.array, 1e-300)
        return alpha_arr.copy()

    # ------------------------------------------------------------------
    def _build_forms(self):
        """Assemble UFL Jacobian and residual (PDAS formulation)."""
        w_k = self.w_k
        W   = self.W
        nu  = self.nu

        y, u, p  = ufl.split(w_k)
        dy, du, dp = ufl.TrialFunctions(W)
        v_y, v_u, v_p = ufl.TestFunctions(W)

        alpha = self.alpha_ufl
        y_d   = self.yd_ufl

        # Active-set indicators (evaluated at current iterate p)
        chi_A_plus  = ufl.conditional(ufl.gt(p,  alpha), 1.0, 0.0)
        chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
        chi_A = chi_A_plus + chi_A_minus

        # Projected control: u* = χ_A+ (p−α)/ν + χ_A− (p+α)/ν
        u_exp = chi_A_plus * (p - alpha) / nu + chi_A_minus * (p + alpha) / nu

        # Jacobian (bilinear form)
        a = (
            (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy * v_y - du * v_y) * ufl.dx
          + (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp * v_p + dy * v_p) * ufl.dx
          + (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
        )

        # Residual −F  (negative so we solve A·dw = b = −F(w_k))
        L = (
          - (ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y * v_y - u * v_y) * ufl.dx
          - (ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p * v_p + (y - y_d) * v_p) * ufl.dx
          - (u - u_exp) * v_u * ufl.dx
        )

        self.a_form = fem.form(a)
        self.L_form = fem.form(L)

        # Active-set volume monitor
        self.vol_form = fem.form(chi_A * ufl.dx)

    # ------------------------------------------------------------------
    def solve(self, verbose: bool = True) -> dict:
        """
        Run the SSN Newton loop to convergence.

        Returns
        -------
        dict with keys:
            converged   : bool
            n_iter      : int
            final_res   : float
            wall_time   : float
            history     : list of (residual, active_vol) per iteration
        """
        history   = []
        converged = False
        t0        = time.perf_counter()

        if verbose and self.comm.rank == 0:
            print(f"  {'iter':>4}  {'‖res‖':>12}  {'active vol':>12}  {'time':>7}")

        for i in range(self.max_iter):
            t_iter = time.perf_counter()

            # Reassemble at current w_k
            self.A.zeroEntries()
            assemble_matrix(self.A, self.a_form, bcs=self.bcs)
            self.A.assemble()

            with self.b.localForm() as loc:
                loc.set(0)
            assemble_vector(self.b, self.L_form)
            apply_lifting(self.b, [self.a_form], [self.bcs])
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(self.b, self.bcs)

            res_norm   = self.b.norm()
            active_vol = self.comm.allreduce(
                fem.assemble_scalar(self.vol_form), op=MPI.SUM
            )
            history.append((float(res_norm), float(active_vol)))

            if verbose and self.comm.rank == 0:
                print(
                    f"  {i:>4}  {res_norm:>12.4e}  {active_vol:>12.6f}"
                    f"  {time.perf_counter()-t_iter:>6.3f}s"
                )

            if res_norm < self.tol:
                converged = True
                break

            self.ksp.solve(self.b, self.dw.vector)
            self.dw.x.scatter_forward()
            self.w_k.x.array[:] += self.dw.x.array[:]
            self.w_k.x.scatter_forward()

        wall_time = time.perf_counter() - t0

        if verbose and self.comm.rank == 0:
            status = "converged" if converged else "NOT converged"
            print(f"  SSN {status} in {len(history)} iters ({wall_time:.1f}s)")

        return {
            "converged": converged,
            "n_iter":    len(history),
            "final_res": history[-1][0] if history else float("nan"),
            "wall_time": wall_time,
            "history":   history,
        }

    # ------------------------------------------------------------------
    def get_adjoint_func(self):
        """Return the adjoint p as a collapsed fem.Function (for AMR jump indicator)."""
        return self.w_k.sub(2).collapse()

    def get_dof_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (dof_coords, y_star, u_star, p_star, chi_A_vals) as numpy arrays.

        chi_A is computed from the adjoint p and the projected alpha field,
        matching the UFL conditional used in the SSN formulation exactly.
        Using |u|>threshold is unreliable: u→0 near Γ and accumulates
        floating-point noise across Newton iterations.
        """
        V_sub, _ = self.W.sub(0).collapse()

        # DOF coordinates (spatial dims only)
        coords = V_sub.tabulate_dof_coordinates()
        coords = coords[:, : self.domain.geometry.dim]

        y_arr = self.w_k.sub(0).collapse().x.array.copy()
        u_arr = self.w_k.sub(1).collapse().x.array.copy()
        p_arr = self.w_k.sub(2).collapse().x.array.copy()

        # chi_A from adjoint p vs pre-computed alpha (same criterion as UFL SSN)
        chi_A_vals = (
            (p_arr > self._alpha_arr) | (p_arr < -self._alpha_arr)
        ).astype(np.float64)

        return coords, y_arr, u_arr, p_arr, chi_A_vals
