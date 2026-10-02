import sys
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl
import logging

logging.getLogger("ffcx").setLevel(logging.ERROR)

def run_plaplace_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("========================================================================")
        print("  TEST: THE NONLINEAR CRUSH (p-LAPLACIAN SSN)")
        print("  Operator: -div(|grad y|^2 grad y) + y = u  (p=4)")
        print("  Constraint: L1 penalty (alpha=0.05, nu=1e-3)")
        print("========================================================================\n")
        
    domain = mesh.create_unit_square(comm, 64, 64)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_y = fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets)
    dofs_p = fem.locate_dofs_topological(W.sub(2), fdim, boundary_facets)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(2))
    bcs = [bc_y, bc_p]

    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    alpha = 0.05
    nu = 1e-3
    x = ufl.SpatialCoordinate(domain)
    y_d = 10.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    # --- 1. p-Laplacian Mathematics (p=4) ---
    # Nonlinear flux: |grad y|^2 * grad y
    # IMPORTANT: We use inner(grad(y), grad(y)) instead of grad_y_sq to avoid UFL arity checks
    flux = ufl.inner(ufl.grad(y), ufl.grad(y)) * ufl.grad(y)
    
    # Linearization (Jacobian) of flux:
    # D(flux)[dy] = |grad y|^2 * grad(dy) + 2 * (grad y . grad dy) * grad y
    d_flux = ufl.inner(ufl.grad(y), ufl.grad(y)) * ufl.grad(dy) + \
             2.0 * ufl.inner(ufl.grad(y), ufl.grad(dy)) * ufl.grad(y)

    # --- 2. SSN / PDAS Logic (L1 penalty) ---
    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu

    # --- 3. Separate the forms to ensure Bilinear / Linear consistency ---
    # Bilinear Form (a)
    a_y = (ufl.inner(d_flux, ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), d_flux) + dp*v_p - dy*v_p) * ufl.dx
    a_u = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
    a = a_y + a_p + a_u

    # Linear Form (L)
    L_y = -(ufl.inner(flux, ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
    # Correcting the adjoint residual: we need the linearization of the objective functional
    # Adjoint is basically testing the sensitivity of (p, flux) 
    L_p = -(ufl.inner(ufl.grad(p), flux) + p*v_p - (y - y_d)*v_p) * ufl.dx
    L_u = -(u - u_expected) * v_u * ufl.dx
    L = L_y + L_p + L_u

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = create_matrix(a_form)
    b = create_vector(L_form)
    dw = fem.Function(W)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    if rank == 0:
        print(f"{'Iter':<6} | {'Residual ||F||':<22} | {'Active Area Vol'}")
        print("-" * 65)

    max_iter = 20
    for i in range(max_iter):
        A.zeroEntries()
        assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        res_norm = b.norm()
        vol_expr = fem.form(chi_A * ufl.dx)
        active_vol = comm.allreduce(fem.assemble_scalar(vol_expr), op=MPI.SUM)
        
        if rank == 0:
            print(f"{i:<6} | {res_norm:<22.6e} | {active_vol:.6f}")

        if res_norm < 1e-8:
            if rank == 0:
                print("\n[SUCCESS] Nonlinear p-Laplacian SSN converged!")
            break
            
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

if __name__ == "__main__":
    run_plaplace_test()
