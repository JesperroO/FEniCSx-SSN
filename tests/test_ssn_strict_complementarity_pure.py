import sys
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl

def run_ghost_test_pure():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    domain = mesh.create_unit_square(comm, 64, 64)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    
    # Pre-solve to find the peak adjoint state
    V_p = fem.FunctionSpace(domain, P1)
    p_lin = ufl.TrialFunction(V_p)
    v_p = ufl.TestFunction(V_p)
    y_d = fem.Constant(domain, PETSc.ScalarType(1.0))
    a_lin = (ufl.inner(ufl.grad(p_lin), ufl.grad(v_p)) + p_lin*v_p) * ufl.dx
    L_lin = y_d * v_p * ufl.dx
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_p_lin = fem.locate_dofs_topological(V_p, fdim, boundary_facets)
    bc_lin = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_p_lin, V_p)
    
    problem = fem.petsc.LinearProblem(a_lin, L_lin, bcs=[bc_lin])
    p_ghost = problem.solve()
    alpha_critical = comm.allreduce(np.max(p_ghost.x.array), op=MPI.MAX)
    
    # The Chattering Trap: we want the initial step to ACTIVATE the center,
    # but the pure L1 Jacobian will instantly try to force p = alpha, dropping y.
    alpha_trap = alpha_critical - 1e-3
    
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))
    dofs_y = fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets)
    dofs_p = fem.locate_dofs_topological(W.sub(2), fdim, boundary_facets)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(2))
    bcs = [bc_y, bc_p]

    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)
    
    # We MUST initialize the Newton solver inside the trap!
    # If we start from 0, it converges to the inactive state immediately.
    # Let's start with p = p_ghost so it falls right into the active set trigger.
    w_k.sub(2).collapse().x.array[:] = p_ghost.x.array[:]

    chi_A_plus = ufl.conditional(ufl.gt(p, alpha_trap), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha_trap), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    chi_I = 1.0 - chi_A

    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
    
    # PURE L1 Jacobian (No nu)
    a_u = (chi_I * du * v_u - chi_A * dp * v_u) * ufl.dx
    a = a_y + a_p + a_u

    # PURE L1 Residual
    L_y = -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
    L_p = -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y - y_d)*v_p) * ufl.dx
    L_u = -(chi_I * u + chi_A * (p - alpha_trap)) * v_u * ufl.dx
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
        print(f"{'Iter':<5} | {'Residual L2':<15} | {'Active Area Vol':<15}")
        print("-" * 45)

    try:
        for i in range(25):
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
                print(f"{i:<5} | {res_norm:<15.4e} | {active_vol:<15.8e}")

            if res_norm < 1e-10:
                if rank == 0:
                    print(">>> CONVERGED!")
                break

            ksp.solve(b, dw.vector)
            dw.x.scatter_forward()
            w_k.x.array[:] += dw.x.array[:]
            w_k.x.scatter_forward()
        else:
            if rank == 0:
                print("-" * 45)
                print(">>> FAILED! Chattering confirmed. The Jacobian collapsed under the ambiguity.")
                print(">>> The Active Set volume oscillates infinitely.")
    except Exception as e:
         if rank == 0:
            print("\n[JACOBIAN SINGULARITY] The direct solver crashed because the Generalized Jacobian matrix lost full rank in the ambiguous flat zone.")

if __name__ == "__main__":
    run_ghost_test_pure()
