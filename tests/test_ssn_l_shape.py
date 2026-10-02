import sys
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl

try:
    import gmsh
    from dolfinx.io import gmshio
except ImportError:
    print("gmsh module not found.")
    sys.exit(1)

def create_l_shape_mesh(comm):
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("L_shape")
        lc = 0.015 # Characteristic length -> defines mesh resolution
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 0.5, 0, lc)
        p4 = gmsh.model.geo.addPoint(0.5, 0.5, 0, lc) # The re-entrant singularity corner!
        p5 = gmsh.model.geo.addPoint(0.5, 1.0, 0, lc)
        p6 = gmsh.model.geo.addPoint(0.0, 1.0, 0, lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.mesh.generate(2)

    domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return domain

def run_l_shape_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=== THE L-SHAPE SINGULARITY TEST ===")
        print("Constructing L-Shape mesh with a re-entrant corner at (0.5, 0.5)...")
        
    domain = create_l_shape_mesh(comm)
    
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))
    
    if rank == 0:
        print(f"Total Degrees of Freedom: {W.dofmap.index_map.size_global * W.dofmap.index_map_bs}")

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

    alpha = 0.005
    nu = 1e-3
    x = ufl.SpatialCoordinate(domain)
    
    # We place the target exactly crossing the singularity corner
    y_d = 5.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

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

    if rank == 0:
        print("-" * 55)
        print(f"{'Iter':<5} | {'Residual L2':<15} | {'Active Area Vol':<15}")
        print("-" * 55)

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
                print("-" * 55)
                print(f"[SUCCESS] L-Shape Singularity Conquered in {i} steps!")
            break
            
        prev_vol = active_vol
        
        ksp.solve(b, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

    if rank == 0:
        u_out = w_k.sub(1).collapse()
        y_out = w_k.sub(0).collapse()
        u_out.name = "Sparse_Control_U"
        y_out.name = "Optimal_State_Y"
        
        with io.XDMFFile(comm, "output_2d/l_shape_control.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u_out)
        with io.XDMFFile(comm, "output_2d/l_shape_state.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(y_out)
        print("Singularity visualization saved to output_2d/l_shape_control.xdmf")

if __name__ == "__main__":
    run_l_shape_test()
