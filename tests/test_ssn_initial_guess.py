import sys
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl
import logging

logging.getLogger("ffcx").setLevel(logging.ERROR)

def solve_with_init(init_type, comm):
    N = 64
    domain = mesh.create_unit_square(comm, N, N)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    bfacets = mesh.exterior_facet_indices(domain.topology)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W.sub(0), fdim, bfacets), W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W.sub(2), fdim, bfacets), W.sub(2))
    bcs = [bc_y, bc_p]

    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    alpha, nu = 0.05, 1e-3
    x = ufl.SpatialCoordinate(domain)
    y_d = 10.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    # Initial Guess Logic
    if init_type == "ZERO":
        w_k.x.array[:] = 0.0
    elif init_type == "NOISE":
        # Intense high-frequency noise
        np.random.seed(42)
        w_k.x.array[:] = np.random.normal(0, 1.5, len(w_k.x.array))
    elif init_type == "OPPOSITE":
        # Physically nonsensical initial guess: High positive state when target is positive,
        # but coupled with a negative adjoint.
        w_k.x.array[:] = 5.0
    
    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu

    a = ((ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx +
         (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx +
         (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx)

    L = -((ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx +
          (ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y - y_d)*v_p) * ufl.dx +
          (u - u_expected) * v_u * ufl.dx)

    a_form, L_form = fem.form(a), fem.form(L)
    A, b = create_matrix(a_form), create_vector(L_form)
    dw_k = fem.Function(W)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.getPC().setFactorSolverType("mumps")

    for i in range(20):
        A.zeroEntries()
        assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        with b.localForm() as loc_b: loc_b.set(0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], [bcs]); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); set_bc(b, bcs)
        
        res = b.norm()
        if res < 1e-9: return i, res
        if res > 1e10 or np.isnan(res): return -1, res
        
        ksp.solve(b, dw_k.vector)
        w_k.x.array[:] += dw_k.x.array[:]
        w_k.x.scatter_forward()
    return -2, res

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(f"{'Init Type':<15} | {'Iterations':<12} | {'Final Res'}")
        print("-" * 45)
    for t in ["ZERO", "NOISE", "OPPOSITE"]:
        iters, res = solve_with_init(t, comm)
        if comm.rank == 0:
            status = str(iters) if iters >= 0 else ("EXPLODED" if iters == -1 else "FAILED")
            print(f"{t:<15} | {status:<12} | {res:.2e}")
