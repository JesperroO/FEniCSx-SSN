import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl
import logging

logging.getLogger("ffcx").setLevel(logging.ERROR)

def run_nonconvex_test(init_val):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 64, 64)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets), W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W.sub(2), fdim, boundary_facets), W.sub(2))
    bcs = [bc_y, bc_p]

    w_k = fem.Function(W)
    # Malicious Non-convex initialization
    w_k.x.array[:] = init_val 
    
    y, u, p = ufl.split(w_k)
    dw = ufl.TrialFunction(W)
    v_y, v_u, v_p = ufl.split(ufl.TestFunction(W))

    # --- THE NON-CONVEX TRAP ---
    # We introduce a non-convex source term in the state equation: sin(5*y)
    # This creates multiple local equilibria (stable and unstable).
    # The energy functional is no longer a bowl; it's a mountain range.
    res_y = (ufl.inner(ufl.grad(y), ufl.grad(v_y)) + (y - u + 5.0 * ufl.sin(5.0 * y)) * v_y) * ufl.dx
    
    # Adjoint linearized correctly: sensitivity of sin(5*y) is 25*cos(5*y)
    res_p = (ufl.inner(ufl.grad(p), ufl.grad(v_p)) + (p + y - 1.0 + 25.0 * ufl.cos(5.0 * y) * p) * v_p) * ufl.dx
    
    alpha, nu = 0.05, 1e-3
    chi_A = ufl.conditional(ufl.gt(abs(p), alpha), 1.0, 0.0)
    u_expected = ufl.conditional(ufl.gt(p, alpha), (p-alpha)/nu, ufl.conditional(ufl.lt(p, -alpha), (p+alpha)/nu, 0.0))
    res_u = (u - u_expected) * v_u * ufl.dx
    
    F = res_y + res_p + res_u
    a = ufl.derivative(F, w_k, dw)
    
    a_form, L_form = fem.form(a), fem.form(-F)
    A, b = create_matrix(a_form), create_vector(L_form)
    dw_k = fem.Function(W)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.getPC().setFactorSolverType("mumps")

    for i in range(15):
        A.zeroEntries(); assemble_matrix(A, a_form, bcs=bcs); A.assemble()
        with b.localForm() as loc_b: loc_b.set(0)
        assemble_vector(b, L_form); apply_lifting(b, [a_form], [bcs]); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); set_bc(b, bcs)
        
        res = b.norm()
        if comm.rank == 0: print(f"Iter {i:2d} | Residual: {res:.4e}")
        if res < 1e-9: return "CONVERGED"
        if res > 1e10 or np.isnan(res): return "EXPLODED"
        
        ksp.solve(b, dw_k.vector)
        w_k.x.array[:] += dw_k.x.array[:]
        w_k.x.scatter_forward()
    return "STALLED/OSCILLATING"

if __name__ == "__main__":
    import os
    rank = MPI.COMM_WORLD.rank
    if rank == 0: print("=== NON-CONVEX INITIAL VALUE DEATH TEST ===")
    
    # Case A: Good initial guess (close to zero)
    if rank == 0: print("\n[Scenario A: Proximity to local minimum (init=0.1)]")
    res_a = run_nonconvex_test(0.1)
    
    # Case B: Bad initial guess (stuck on a non-convex 'ridge')
    if rank == 0: print("\n[Scenario B: The Death Trap (init=2.5)]")
    res_b = run_nonconvex_test(2.5)
    
    if rank == 0:
        print(f"\nFinal Verdict: Scenario A -> {res_a} | Scenario B -> {res_b}")
