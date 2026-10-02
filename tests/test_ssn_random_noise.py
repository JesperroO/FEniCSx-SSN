import sys
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl

def run_noise_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # 1. Mesh
    domain = mesh.create_unit_square(comm, 64, 64)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))
    
    # 2. Boundary
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_y = fem.locate_dofs_topological(W.sub(0), fdim, boundary_facets)
    dofs_p = fem.locate_dofs_topological(W.sub(2), fdim, boundary_facets)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(2))
    bcs = [bc_y, bc_p]

    # 3. Variables
    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    alpha = 0.05
    nu = 1e-3
    
    # =========================================================================
    # THE CRUCIAL PART: Generating completely unstructured random target data
    # =========================================================================
    # Instead of analytical Gaussians, we create a Function and fill it with 
    # uniform random noise. This means the target y_d has NO underlying PDE structure,
    # NO smoothness, and NO predictable active sets.
    V_target = fem.FunctionSpace(domain, P1)
    y_d_func = fem.Function(V_target)
    
    # Fill the degrees of freedom with random noise between 0.0 and 2.0
    np.random.seed(42)  # For reproducibility, but it is truly random
    noise = np.random.uniform(0.0, 2.0, len(y_d_func.x.array))
    y_d_func.x.array[:] = noise
    
    # Let's also add a giant, asymmetrical, non-smooth "block" in the middle
    # to really mess with the gradient
    coords = V_target.tabulate_dof_coordinates()
    block_mask = (coords[:, 0] > 0.4) & (coords[:, 0] < 0.8) & (coords[:, 1] > 0.2) & (coords[:, 1] < 0.6)
    y_d_func.x.array[block_mask] += 5.0
    
    if rank == 0:
        print("=== STRESS TEST: SSN WITH RANDOM NOISE & NON-SMOOTH TARGET ===")
        print("Target state y_d is filled with uniform random noise and a sharp discontinuous block.")
    # =========================================================================

    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu

    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
    a_u = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
    a = a_y + a_p + a_u

    L_y = -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
    L_p = -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y - y_d_func)*v_p) * ufl.dx
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

    max_iter = 20
    tol = 1e-10
    prev_vol = -1.0
    vol_expr = fem.form(chi_A * ufl.dx)

    if rank == 0:
        print("-" * 50)
        print(f"{'Iter':<5} | {'Residual L2':<15} | {'Active Area Vol':<15}")
        print("-" * 50)

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
        active_vol = comm.allreduce(fem.assemble_scalar(vol_expr), op=MPI.SUM)
        
        if rank == 0:
            print(f"{i:<5} | {res_norm:<15.4e} | {active_vol:<15.6f}")

        if res_norm < tol or (res_norm < 1e-8 and abs(active_vol - prev_vol) < 1e-12):
            if rank == 0:
                print("-" * 50)
                print(f"[SUCCESS] SSN converged even with pure random noise in {i} steps!")
            break
            
        prev_vol = active_vol
        
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

if __name__ == "__main__":
    run_noise_test()
