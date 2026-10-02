import sys
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl
import gc

def solve_for_N(N, comm):
    domain = mesh.create_unit_square(comm, N, N)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))
    
    dofs_count = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

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
    y_d = 5.0 * ufl.exp(-50.0 * ((x[0]-0.3)**2 + (x[1]-0.3)**2)) + \
          5.0 * ufl.exp(-50.0 * ((x[0]-0.7)**2 + (x[1]-0.7)**2))

    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu

    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
    a_u = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
    a = a_y + a_p + a_u

    L_y = -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
    L_p = -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y - y_d)*v_p) * ufl.dx
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

    max_iter = 25
    tol = 1e-10
    prev_vol = -1.0
    vol_expr = fem.form(chi_A * ufl.dx)
    
    converged_iter = -1

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

        if res_norm < tol or (res_norm < 1e-8 and abs(active_vol - prev_vol) < 1e-12):
            converged_iter = i
            break
            
        prev_vol = active_vol
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()
        
    # Free memory forcefully for the next giant grid
    ksp.destroy()
    A.destroy()
    b.destroy()
    del w_k, dw, A, b, ksp, a_form, L_form, W, domain
    gc.collect()

    return dofs_count, converged_iter

def run_mesh_independence_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=== THE ABYSS OF MESH INDEPENDENCE ===")
        print("Testing SSN convergence across exponentially growing dimensions...")
        print("-" * 55)
        print(f"{'Mesh N x N':<12} | {'Degrees of Freedom':<20} | {'Newton Iterations':<15}")
        print("-" * 55)

    # We test 64, 128, 256, 512
    # 512x512 generates ~790,000 DOFs, which is a massive test for a single-node direct solver
    resolutions = [64, 128, 256, 512]
    
    for N in resolutions:
        if rank == 0:
            print(f"{str(N)+'x'+str(N):<12} | ", end="", flush=True)
            
        try:
            dofs, iters = solve_for_N(N, comm)
            if rank == 0:
                print(f"{dofs:<20} | {iters:<15}")
        except Exception as e:
            if rank == 0:
                print(f"FAILED (Likely OOM or KSP Crash): {e}")

if __name__ == "__main__":
    run_mesh_independence_test()
