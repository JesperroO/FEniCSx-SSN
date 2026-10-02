import sys
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl

def run_noise_test_export():
    comm = MPI.COMM_WORLD
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
    
    V_target = fem.FunctionSpace(domain, P1)
    y_d_func = fem.Function(V_target)
    np.random.seed(42)
    noise = np.random.uniform(0.0, 2.0, len(y_d_func.x.array))
    y_d_func.x.array[:] = noise
    
    coords = V_target.tabulate_dof_coordinates()
    block_mask = (coords[:, 0] > 0.4) & (coords[:, 0] < 0.8) & (coords[:, 1] > 0.2) & (coords[:, 1] < 0.6)
    y_d_func.x.array[block_mask] += 5.0

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

    for i in range(15):
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
        if res_norm < 1e-10:
            break
        
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

    # Export
    u_out = w_k.sub(1).collapse()
    u_out.name = "Sparse_Control_U"
    y_d_func.name = "Noisy_Target_Yd"
    
    with io.XDMFFile(comm, "output_2d/noise_control.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_out)
        
    with io.XDMFFile(comm, "output_2d/noise_target.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(y_d_func)

if __name__ == "__main__":
    run_noise_test_export()
