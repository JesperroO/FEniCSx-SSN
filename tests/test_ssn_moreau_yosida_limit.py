import sys
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from petsc4py import PETSc
import ufl
import logging

# Suppress FEniCSx C++ logs
logging.getLogger("ffcx").setLevel(logging.ERROR)

def run_vacuum_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=== THE MOREAU-YOSIDA VACUUM TEST ===")
        print("Progressively collapsing the regularization parameter nu to machine epsilon.")
        print("Testing the raw ill-conditioning of the semi-smooth Jacobian.")
        print("-" * 65)
        print(f"{'nu (Penalty)':<15} | {'Iterations':<15} | {'Final Residual':<20}")
        print("-" * 65)
        
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

    alpha = 0.05
    x = ufl.SpatialCoordinate(domain)
    y_d = 5.0 * ufl.exp(-50.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

    # Sweeping nu from 1e-2 down to 1e-16
    nu_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    
    for nu_val in nu_values:
        w_k = fem.Function(W)
        y, u, p = ufl.split(w_k)
        dy, du, dp = ufl.TrialFunctions(W)
        v_y, v_u, v_p = ufl.TestFunctions(W)

        # Standard PDAS Formulation
        chi_A_plus = ufl.conditional(ufl.gt(p, alpha), 1.0, 0.0)
        chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
        chi_A = chi_A_plus + chi_A_minus
        
        # Safe division: mathematically nu_val is non-zero, but numerically it approaches epsilon
        u_expected = chi_A_plus * (p - alpha)/nu_val + chi_A_minus * (p + alpha)/nu_val

        a_y = (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
        a_p = (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
        a_u = (du * v_u - (chi_A / nu_val) * dp * v_u) * ufl.dx
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
        
        converged = False
        iters = 0
        final_res = 0.0
        
        try:
            for i in range(20):
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
                if np.isnan(res_norm):
                    final_res = np.nan
                    break
                    
                if res_norm < 1e-9:
                    converged = True
                    iters = i
                    final_res = res_norm
                    break
                    
                # The critical point: MUMPS will crash here if condition number explodes
                ksp.solve(b, dw.vector)
                dw.x.scatter_forward()
                w_k.x.array[:] += dw.x.array[:]
                w_k.x.scatter_forward()
            
            if not converged:
                iters = "FAIL (No Conv)"
                final_res = res_norm
                
        except Exception as e:
            iters = "CRASH (Singular)"
            final_res = np.nan
            
        if rank == 0:
            res_str = f"{final_res:.4e}" if not np.isnan(final_res) else "NaN"
            print(f"{nu_val:<15.1e} | {str(iters):<15} | {res_str:<20}")
            
        ksp.destroy()
        A.destroy()
        b.destroy()

if __name__ == "__main__":
    run_vacuum_test()
