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

def run_bang_bang_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("========================================================================")
        print("  TEST: THE ABSOLUTE BANG-BANG RAMPAGE (STRICT BOX CONSTRAINTS)")
        print("  Grid: 128x128  |  u in [-1, 1]  |  y_d = 100 * high-frequency sine")
        print("========================================================================\n")
        
    domain = mesh.create_unit_square(comm, 128, 128)
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

    # Box constraints and regularization
    u_a = -1.0
    u_b = 1.0
    nu = 1e-4  # Push the penalty very low to make the active set boundaries brutally sharp
    
    # The physically impossible target state
    # A massive, high-frequency checkerboard pattern ranging from -100 to 100
    # Since the control u can only supply force between -1 and 1, the optimal
    # control will be forced to purely alternate between -1 and 1 (Bang-Bang),
    # completely abandoning any smooth transition.
    x = ufl.SpatialCoordinate(domain)
    y_d = 100.0 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(10.0 * ufl.pi * x[1])

    # KKT for Box Constraints: u = P_{[-1, 1]}(p / nu)
    # Active Upper: p / nu > 1 => p > nu
    # Active Lower: p / nu < -1 => p < -nu
    # Inactive: -nu <= p <= nu
    chi_upper = ufl.conditional(ufl.gt(p, nu * u_b), 1.0, 0.0)
    chi_lower = ufl.conditional(ufl.lt(p, nu * u_a), 1.0, 0.0)
    chi_I = 1.0 - chi_upper - chi_lower
    
    u_expected = chi_upper * u_b + chi_lower * u_a + chi_I * (p / nu)

    a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
    a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
    
    # Jacobian for Box Constraints
    # Active sets: u is locked to a constant, so du = 0 w.r.t p (dF_u/dp = 0)
    # Inactive set: u = p/nu, so dF_u/dp = -1/nu
    a_u = (du * v_u - (chi_I / nu) * dp * v_u) * ufl.dx
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

    if rank == 0:
        print(f"{'Iter':<6} | {'Residual ||F||':<22} | {'Active Set (Upper / Lower)'}")
        print("-" * 65)

    max_iter = 25
    prev_upper = -1
    prev_lower = -1

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
        
        vol_upper = comm.allreduce(fem.assemble_scalar(fem.form(chi_upper * ufl.dx)), op=MPI.SUM)
        vol_lower = comm.allreduce(fem.assemble_scalar(fem.form(chi_lower * ufl.dx)), op=MPI.SUM)

        if rank == 0:
            print(f"{i:<6} | {res_norm:<22.6e} | {vol_upper:.4f} / {vol_lower:.4f}")

        if res_norm < 1e-10:
            if rank == 0:
                print("\n  Result: CONVERGED in", i, "iterations")
                print("  Total Active Volume:", f"{(vol_upper + vol_lower) * 100:.2f}% of the domain is saturated at absolute limits.")
            break
            
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()
        
    # Export for visualization
    u_out = w_k.sub(1).collapse()
    u_out.name = "Pure_Bang_Bang_Control"
    with io.XDMFFile(comm, "output_2d/bang_bang_control.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_out)

if __name__ == "__main__":
    run_bang_bang_test()
