import sys
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import ufl

def run_dolfinx_smoke_test():
    print("=== FEniCSx (dolfinx v0.7.3) SSN Assembly Smoke Test ===")
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    mixed_el = ufl.MixedElement([P1, P1, P1])
    W = fem.FunctionSpace(domain, mixed_el)
    
    print(f"Total Degrees of Freedom: {W.dofmap.index_map.size_global}")
    
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)
    
    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    w_k.x.array[:] = 0.5 
    
    alpha = 0.05
    nu = 1e-3
    y_d = fem.Constant(domain, PETSc.ScalarType(2.0))
    
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
    
    print("Compiling UFL forms via FFCx...")
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    print("Assembling PETSc Matrix and Vector...")
    A = assemble_matrix(a_form)
    A.assemble()
    
    b = assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    norm_b = b.norm()
    print(f"Matrix A size: {A.getSize()}")
    print(f"Residual b L2-norm: {norm_b:.4e}")
    
    if A.getSize()[0] > 0 and norm_b > 0:
        print("\n[SUCCESS] dolfinx UFL logic parsed and PETSc matrices assembled correctly!")

if __name__ == "__main__":
    run_dolfinx_smoke_test()
