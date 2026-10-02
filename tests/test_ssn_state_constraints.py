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

def run_state_constraint_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("========================================================================")
        print("  TEST: THE DEADLY BARRIER (HARD STATE CONSTRAINTS)")
        print("  Grid: 128x128  |  Target peak = 1.0  |  Constraint y <= 0.2")
        print("========================================================================\n")
        
    domain = mesh.create_unit_square(comm, 128, 128)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1])) # Variables: y, p

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_y = fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets)
    dofs_p = fem.locate_dofs_topological(W.sub(1), fdim, boundary_facets)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(1))
    bcs = [bc_y, bc_p]

    w_k = fem.Function(W)
    y, p = ufl.split(w_k)
    dy, dp = ufl.TrialFunctions(W)
    v_y, v_p = ufl.TestFunctions(W)

    alpha_reg = 1e-4  # Tikhonov regularization for control u
    gamma = 1e3       # Moreau-Yosida penalty for state constraint y <= psi
    
    x = ufl.SpatialCoordinate(domain)
    # Target: A large peak in the center (peak value 1.0)
    y_d = ufl.exp(-25.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
    # Constraint: y(x) <= 0.2 everywhere
    psi = fem.Constant(domain, PETSc.ScalarType(0.2))

    # Regularized multiplier lambda = max(0, gamma * (y - psi))
    lambda_reg = ufl.max_value(0.0, gamma * (y - psi))
    chi_A = ufl.conditional(ufl.gt(gamma * (y - psi), 0.0), 1.0, 0.0)

    # Jacobian
    # F_y: -Delta dy + dy - dp/alpha = 0
    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - (dp/alpha_reg)*v_y) * ufl.dx
    # F_p: -Delta dp + dp - dy + gamma*chi_A*dy = -y_d
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p - dy*v_p + gamma * chi_A * dy * v_p) * ufl.dx
    a = a_y + a_p

    # Residual
    L_y = -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - (p/alpha_reg)*v_y) * ufl.dx
    L_p = -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p - (y - y_d)*v_p + lambda_reg*v_p) * ufl.dx
    L = L_y + L_p

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
        print(f"{'Iter':<6} | {'Residual ||F||':<22} | {'Violation Max'}")
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
        
        y_curr = w_k.sub(0).collapse()
        violation = np.max(y_curr.x.array - 0.2)
        
        if rank == 0:
            print(f"{i:<6} | {res_norm:<22.6e} | {max(0, violation):.6f}")

        if res_norm < 1e-10:
            if rank == 0:
                print("\n  Result: CONVERGED in", i, "iterations")
                print(f"  Final state peak: {np.max(y_curr.x.array):.6f} (Constraint: 0.200000)")
            break
            
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()
        
    # Export for visualization
    y_out = w_k.sub(0).collapse()
    y_out.name = "Constrained_State_Y"
    u_out = fem.Function(fem.FunctionSpace(domain, P1))
    u_out.name = "Resulting_Control_U"
    u_out.x.array[:] = w_k.sub(1).collapse().x.array[:] / alpha_reg
    
    with io.XDMFFile(comm, "output_2d/state_constrained.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(y_out)
        xdmf.write_function(u_out)

if __name__ == "__main__":
    run_state_constraint_test()
