import numpy as np

def setup_1d_pde_dense(N):
    h = 1.0 / (N + 1)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2.0/h**2 + 1.0
        if i > 0: A[i, i-1] = -1.0/h**2
        if i < N-1: A[i, i+1] = -1.0/h**2
    return A, h

def run_1d_ssn_test_dense():
    print("=== Running 1D Coupled PDE SSN Test (Low Penalty) ===")
    N = 100 
    A, h = setup_1d_pde_dense(N)
    
    # We lower the penalty significantly so the control MUST activate
    # to pull the state towards the target.
    alpha = 0.005
    c = 1.0
    
    x = np.linspace(0, 1, N+2)[1:-1]
    y_d = np.where((x > 0.4) & (x < 0.6), 5.0, 0.0) # Sharp, high target
    
    y_k = np.zeros(N); u_k = np.zeros(N); p_k = np.zeros(N)
    I = np.eye(N); O = np.zeros((N, N))
    
    for i in range(10):
        z_plus = u_k + c * (p_k - alpha)
        z_minus = u_k + c * (p_k + alpha)
        
        chi_A = (z_plus > 0).astype(float) + (z_minus < 0).astype(float)
        chi_I = 1.0 - chi_A
        
        max_term = np.where(z_plus > 0, z_plus, 0.0)
        min_term = np.where(z_minus < 0, z_minus, 0.0)
        
        F_y = A @ y_k - u_k
        F_p = A @ p_k + y_k - y_d
        F_u = u_k - max_term - min_term
        F = np.concatenate([F_y, F_p, F_u])
        
        res_norm = np.linalg.norm(F)
        active_count = int(np.sum(chi_A))
        
        print(f"Iter {i:2d} | Residual: {res_norm:.6e} | Active Nodes: {active_count:3d}/{N}")
        
        if res_norm < 1e-12:
            print("\n[SUCCESS] Global SSN converged with active sets triggered!")
            break
            
        D_chi_I = np.diag(chi_I)
        D_chi_A = np.diag(-c * chi_A)
        
        J = np.block([
            [A, -I, O],
            [I, O, A],
            [O, D_chi_I, D_chi_A]
        ])
        
        dw = np.linalg.solve(J, -F)
        y_k += dw[0:N]; u_k += dw[N:2*N]; p_k += dw[2*N:3*N]

if __name__ == "__main__":
    run_1d_ssn_test_dense()
