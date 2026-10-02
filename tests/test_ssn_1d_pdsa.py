import numpy as np

def setup_1d_pde_dense(N):
    h = 1.0 / (N + 1)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2.0/h**2 + 1.0
        if i > 0: A[i, i-1] = -1.0/h**2
        if i < N-1: A[i, i+1] = -1.0/h**2
    return A, h

def run_1d_ssn_test_pdsa():
    print("=== Running 1D Coupled PDE SSN Test (Primal-Dual Active Set / Exact Formulation) ===")
    N = 100 
    A, h = setup_1d_pde_dense(N)
    
    alpha = 0.05
    nu = 1e-3  # L2 Tikhonov regularization is mandatory for L1 control
    
    x = np.linspace(0, 1, N+2)[1:-1]
    y_d = np.where((x > 0.4) & (x < 0.6), 5.0, 0.0) 
    
    y_k = np.zeros(N); u_k = np.zeros(N); p_k = np.zeros(N)
    I = np.eye(N); O = np.zeros((N, N))
    
    for i in range(15):
        # KKT Condition for u with Tikhonov: 
        # nu * u + subdiff(L1) - p = 0
        # -> u = max(0, (p - alpha)/nu) + min(0, (p + alpha)/nu)
        
        # Determine Active Sets directly from adjoint state
        chi_A_plus = (p_k > alpha).astype(float)
        chi_A_minus = (p_k < -alpha).astype(float)
        chi_A = chi_A_plus + chi_A_minus
        chi_I = 1.0 - chi_A
        
        # Calculate expected control based on current p_k
        u_expected = chi_A_plus * (p_k - alpha)/nu + chi_A_minus * (p_k + alpha)/nu
        
        # Residuals
        F_y = A @ y_k - u_k
        F_p = A @ p_k + y_k - y_d
        F_u = nu * u_k - p_k + alpha * chi_A_plus - alpha * chi_A_minus
        F_u = chi_A * F_u + chi_I * u_k  # Force u=0 on inactive set
        
        F = np.concatenate([F_y, F_p, F_u])
        res_norm = np.linalg.norm(F)
        active_count = int(np.sum(chi_A))
        
        print(f"Iter {i:2d} | Residual: {res_norm:.6e} | Active Nodes: {active_count:3d}/{N}")
        
        if res_norm < 1e-12:
            print("\n[SUCCESS] Global SSN converged perfectly in standard PDAS form!")
            break
            
        # Jacobian Assembly
        # Active: nu * du - dp = 0
        # Inactive: du = 0
        D_chi_I = np.diag(chi_I)
        D_chi_A = np.diag(chi_A)
        
        # [ A   -I    0  ] 
        # [ I    0    A  ] 
        # [ 0   X_I+nu*X_A  -X_A ] 
        J = np.block([
            [A, -I, O],
            [I, O, A],
            [O, D_chi_I + nu * D_chi_A, -D_chi_A]
        ])
        
        dw = np.linalg.solve(J, -F)
        y_k += dw[0:N]; u_k += dw[N:2*N]; p_k += dw[2*N:3*N]

if __name__ == "__main__":
    run_1d_ssn_test_pdsa()
