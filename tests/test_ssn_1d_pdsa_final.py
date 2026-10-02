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
    print("=== FINAL: 1D Coupled PDE SSN Test (Primal-Dual Active Set) ===")
    N = 100 
    A, h = setup_1d_pde_dense(N)
    
    alpha = 0.05
    nu = 1e-3  
    
    x = np.linspace(0, 1, N+2)[1:-1]
    y_d = np.where((x > 0.4) & (x < 0.6), 5.0, 0.0) 
    
    y_k = np.zeros(N); u_k = np.zeros(N); p_k = np.zeros(N)
    I = np.eye(N); O = np.zeros((N, N))
    
    prev_active_count = -1
    
    for i in range(15):
        chi_A_plus = (p_k > alpha).astype(float)
        chi_A_minus = (p_k < -alpha).astype(float)
        chi_A = chi_A_plus + chi_A_minus
        chi_I = 1.0 - chi_A
        
        # Calculate the EXACT non-smooth residuals
        F_y = A @ y_k - u_k
        F_p = A @ p_k + y_k - y_d
        
        # u - max(0, (p-alpha)/nu) - min(0, (p+alpha)/nu) = 0
        u_expected = chi_A_plus * (p_k - alpha)/nu + chi_A_minus * (p_k + alpha)/nu
        F_u = u_k - u_expected
        
        F = np.concatenate([F_y, F_p, F_u])
        res_norm = np.linalg.norm(F)
        active_count = int(np.sum(chi_A))
        
        print(f"Iter {i:2d} | Residual: {res_norm:.6e} | Active Nodes: {active_count:3d}/{N}")
        
        # The true stopping criteria for SSN/PDAS is when the active set stops changing
        # and the residual drops to machine precision.
        if res_norm < 1e-12 or (res_norm < 1e-9 and active_count == prev_active_count):
            print("\n[SUCCESS] Global SSN converged with Superlinear precision in {} iterations!".format(i))
            break
            
        prev_active_count = active_count
        
        D_chi_I = np.diag(chi_I)
        D_chi_A = np.diag(chi_A)
        
        # Jacobian Assembly
        # F_u = u - u_expected
        # d(F_u)/du = I
        # d(F_u)/dp = -chi_A / nu
        J = np.block([
            [A, -I, O],
            [I, O, A],
            [O, I, -D_chi_A / nu]
        ])
        
        dw = np.linalg.solve(J, -F)
        y_k += dw[0:N]; u_k += dw[N:2*N]; p_k += dw[2*N:3*N]

if __name__ == "__main__":
    run_1d_ssn_test_pdsa()
