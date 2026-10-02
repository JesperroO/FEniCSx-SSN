import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl

def run_failure_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=== FAILURE DEMONSTRATION: NON-ELLIPTIC (CONVECTION-DOMINATED) PDE ===")
        print("Attempting to run SSN on an advection-dominated equation where ")
        print("the strong smoothing effect of the Laplacian is destroyed.")
        
    # 1. Mesh
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
    
    # Target state: a simple bump
    x = ufl.SpatialCoordinate(domain)
    y_d = 5.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    # =========================================================================
    # THE POISON PILL: Convection-Dominated Advection-Diffusion
    # We shrink the diffusion coefficient to near zero (1e-4) and introduce 
    # a massive convective wind vector [100.0, 100.0].
    # Equation: -0.0001 * \Delta y + [100, 100] \cdot \nabla y + y = u
    # =========================================================================
    eps = 1e-4  # Tiny diffusion
    wind = ufl.as_vector([100.0, 100.0]) # Massive wind blowing diagonally
    
    # PDAS non-smooth logic (unchanged)
    chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A = chi_A_plus + chi_A_minus
    u_expected = chi_A_plus * (p - alpha)/nu + chi_A_minus * (p + alpha)/nu

    # POISONED Jacobian (Bilinear Form)
    # Notice the huge asymmetrical convective term: inner(wind, grad(dy))*v_y
    a_y = (eps * ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + ufl.inner(wind, ufl.grad(dy))*v_y + dy*v_y - du*v_y) * ufl.dx
    
    # The adjoint operator reverses the wind direction mathematically: -inner(wind, grad(v_p))*dp
    a_p = (eps * ufl.inner(ufl.grad(dp), ufl.grad(v_p)) - ufl.inner(wind, ufl.grad(dp))*v_p + dp*v_p + dy*v_p) * ufl.dx
    
    a_u = (du * v_u - (chi_A / nu) * dp * v_u) * ufl.dx
    a = a_y + a_p + a_u

    # POISONED Residual
    L_y = -(eps * ufl.inner(ufl.grad(y), ufl.grad(v_y)) + ufl.inner(wind, ufl.grad(y))*v_y + y*v_y - u*v_y) * ufl.dx
    L_p = -(eps * ufl.inner(ufl.grad(p), ufl.grad(v_p)) - ufl.inner(wind, ufl.grad(p))*v_p + p*v_p + (y - y_d)*v_p) * ufl.dx
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

    max_iter = 15
    tol = 1e-10

    if rank == 0:
        print("-" * 50)
        print(f"{'Iter':<5} | {'Residual L2':<15} | {'Status'}")
        print("-" * 50)

    try:
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
            
            if np.isnan(res_norm) or res_norm > 1e10:
                if rank == 0:
                    print(f"{i:<5} | {res_norm:<15.4e} | ALGORITHM EXPLODED (NaN or Divergence)")
                break

            if rank == 0:
                print(f"{i:<5} | {res_norm:<15.4e} | Struggling...")

            # The solver will likely crash here because the matrix loses its nice coercive properties
            ksp.solve(b, dw.vector)
            dw.x.scatter_forward()
            w_k.x.array[:] += dw.x.array[:]
            w_k.x.scatter_forward()
            
    except Exception as e:
        if rank == 0:
            print("\n[CATASTROPHIC FAILURE] The direct solver crashed!")
            print(f"Error: {e}")
            print("Reason: The convective wind caused numerical oscillations (Gibbs phenomenon),")
            print("which destroyed the adjoint variable p. The active set indicator chi_A began jumping")
            print("wildly, causing the Semismooth Jacobian to become highly ill-conditioned and singular.")

if __name__ == "__main__":
    run_failure_test()
