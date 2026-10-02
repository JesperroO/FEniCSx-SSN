import sys
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
        print("  TEST: THE NONLINEAR CRUSH V2 (p-LAPLACIAN + SSN)")
        print("  Operator: -div(|grad y|^2 grad y) + y = u")
        print("  Mixed Autodiff (PDE) + Manual Slant Derivative (SSN)")
        print("========================================================================\n")
        
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
    y, u, p = ufl.split(w_k)
    dw = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    v_y, v_u, v_p = ufl.split(v)

    alpha = 0.05
    nu = 1e-3
    x = ufl.SpatialCoordinate(domain)
    y_d = 10.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    # --- 1. Define Smooth Residuals (The PDE and Adjoint blocks) ---
    # Nonlinear State Residual
    res_y = (ufl.inner(ufl.inner(ufl.grad(y), ufl.grad(y)) * ufl.grad(y), ufl.grad(v_y)) + (y - u) * v_y) * ufl.dx
    
    # Adjoint Residual (Linearized state operator adjointed)
    # For p-Laplacian, it is symmetric, so we test sensitivity of p against state variations
    res_p = (ufl.inner(ufl.inner(ufl.grad(y), ufl.grad(y)) * ufl.grad(p), ufl.grad(v_p)) + 
             2.0 * ufl.inner(ufl.grad(y), ufl.grad(p)) * ufl.inner(ufl.grad(y), ufl.grad(v_p)) + 
             (p + y - y_d) * v_p) * ufl.dx

    # --- 2. Define Non-Smooth Residual (The Control block) ---
    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu
    res_u = (u - u_expected) * v_u * ufl.dx

    F_total = res_y + res_p + res_u

    # --- 3. Construct Mixed Jacobian ---
    # We use ufl.derivative for the smooth parts of y and p,
    # but the derivative of res_u must be careful with the non-smooth indicators.
    
    # Smooth part Jacobian
    a_smooth = ufl.derivative(res_y + res_p, w_k, dw)
    
    # Manual Slant Derivative for res_u
    # d(res_u)/du = 1, d(res_u)/dp = -chi_A/nu
    dy, du, dp = ufl.split(dw)
    a_nonsmooth = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
    
    a = a_smooth + a_nonsmooth
    L = -F_total

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = create_matrix(a_form)
    b = create_vector(L_form)
    dw_k = fem.Function(W)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    if rank == 0:
        print(f"{'Iter':<6} | {'Residual ||F||':<22} | {'Active Area Vol'}")
        print("-" * 65)

    max_iter = 25
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
        active_vol = comm.allreduce(fem.assemble_scalar(fem.form(chi_A * ufl.dx)), op=MPI.SUM)
        
        if rank == 0:
            print(f"{i:<6} | {res_norm:<22.6e} | {active_vol:.6f}")

        if res_norm < 1e-9:
            if rank == 0:
                print("\n[SUCCESS] Nonlinear Mixed Autodiff SSN converged!")
            break
            
        ksp.solve(b, dw_k.vector)
        dw_k.x.scatter_forward()
        w_k.x.array[:] += dw_k.x.array[:]
        w_k.x.scatter_forward()
    else:
        if rank == 0:
            print("\n[FAILED] Reached max_iter.")

if __name__ == "__main__":
    run_plaplace_test()
