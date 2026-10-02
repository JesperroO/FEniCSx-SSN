import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def setup_1d_pde(N):
    """Sets up the 1D discretized operator A = -Laplacian + I"""
    h = 1.0 / (N + 1)
    diagonals = [
        -1.0/h**2 * np.ones(N-1),
        (2.0/h**2 + 1.0) * np.ones(N),
        -1.0/h**2 * np.ones(N-1)
    ]
    A = sp.diags(diagonals, [-1, 0, 1], format='csr')
    return A, h

def proximal_L1(z, c, alpha):
    return np.maximum(0, z - c*alpha) + np.minimum(0, z + c*alpha)

def run_1d_ssn_test():
    print("=== Running 1D Coupled PDE Semismooth Newton Test ===")
    N = 100  # 100 grid points
    A, h = setup_1d_pde(N)
    
    alpha = 0.1   # L1 penalty parameter (forces sparsity)
    c = 1.0       # Proximal scaling
    
    # Target state: A step function in the middle
    x = np.linspace(0, 1, N+2)[1:-1]
    y_d = np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)
    
    # Initialize variables (y, u, p) to zero
    y_k = np.zeros(N)
    u_k = np.zeros(N)
    p_k = np.zeros(N)
    
    I = sp.eye(N, format='csr')
    O = sp.csr_matrix((N, N))
    
    for i in range(10):
        # 1. Proximal Active Set Evaluation
        z_plus = u_k + c * (p_k - alpha)
        z_minus = u_k + c * (p_k + alpha)
        
        chi_A_plus = (z_plus > 0).astype(float)
        chi_A_minus = (z_minus < 0).astype(float)
        chi_A = chi_A_plus + chi_A_minus
        chi_I = 1.0 - chi_A
        
        # 2. Compute Non-smooth Residuals
        max_term = np.where(z_plus > 0, z_plus, 0.0)
        min_term = np.where(z_minus < 0, z_minus, 0.0)
        
        F_y = A.dot(y_k) - u_k
        F_p = A.dot(p_k) + y_k - y_d
        F_u = u_k - max_term - min_term
        
        F = np.concatenate([F_y, F_p, F_u])
        res_norm = np.linalg.norm(F)
        active_count = int(np.sum(chi_A))
        
        print(f"Iter {i:2d} | Residual L2 Norm: {res_norm:.6e} | Active Nodes: {active_count:3d}/{N}")
        
        if res_norm < 1e-12:
            print("\n[SUCCESS] Global SSN converged to machine precision!")
            break
            
        # 3. Assemble Block Jacobian
        # [ A   -I    0  ] [ dy ]   [ -F_y ]
        # [ I    0    A  ] [ du ] = [ -F_p ]
        # [ 0   X_I -cX_A] [ dp ]   [ -F_u ]
        D_chi_I = sp.diags(chi_I, 0)
        D_chi_A = sp.diags(-c * chi_A, 0)
        
        J = sp.bmat([
            [A, -I, O],
            [I, O, A],
            [O, D_chi_I, D_chi_A]
        ], format='csr')
        
        # 4. Solve the Newton Step
        try:
            dw = spla.spsolve(J, -F)
        except RuntimeError:
            print("[FAILED] Jacobian is singular!")
            break
            
        # 5. Update
        y_k += dw[0:N]
        u_k += dw[N:2*N]
        p_k += dw[2*N:3*N]

if __name__ == "__main__":
    run_1d_ssn_test()
