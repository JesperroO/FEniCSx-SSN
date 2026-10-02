import numpy as np

def proximal_L1(z, c, alpha):
    return np.maximum(0, z - c*alpha) + np.minimum(0, z + c*alpha)

def run_smoke_test():
    print("--- Running Rigorous Local SSN Smoke Test ---")
    alpha = 1.0
    c = 1.0
    
    # Let's test the EXACT linear block system logic at a single grid point
    # Variables: w = [y, u, p]^T
    # True solution we want to reach (say, active set where u > 0)
    # p* = alpha (to trigger active set), u* = 1.5, y* = 1.5
    
    # Current guess
    y_k = 0.0; u_k = 0.0; p_k = 0.0
    y_d = 1.5 # Target state
    
    for i in range(5):
        # 1. Evaluate indicators (Chi) based on current u_k, p_k
        z_plus = u_k + c * (p_k - alpha)
        z_minus = u_k + c * (p_k + alpha)
        
        chi_A_plus = 1.0 if z_plus > 0 else 0.0
        chi_A_minus = 1.0 if z_minus < 0 else 0.0
        chi_A = chi_A_plus + chi_A_minus
        chi_I = 1.0 - chi_A
        
        # Proximal evaluation
        max_term = z_plus if z_plus > 0 else 0.0
        min_term = z_minus if z_minus < 0 else 0.0
        
        # 2. Build local 3x3 Jacobian block (representing the UFL bilinear forms without the Laplacian part)
        # a_y = dy - du
        # a_p = dp + dy
        # a_u = chi_I * du - c * chi_A * dp
        J = np.array([
            [1.0, -1.0,  0.0],
            [1.0,  0.0,  1.0],
            [0.0, chi_I, -c * chi_A]
        ])
        
        # 3. Build local Residual vector (representing L_y, L_p, L_u)
        # L_y = -(y_k - u_k)
        # L_p = -(p_k + y_k - y_d)
        # L_u = -(u_k - max_term - min_term)
        F = np.array([
            (y_k - u_k),
            (p_k + y_k - y_d),
            (u_k - max_term - min_term)
        ])
        
        res_norm = np.linalg.norm(F)
        print(f"Iter {i}: [y={y_k:.4f}, u={u_k:.4f}, p={p_k:.4f}], Residual={res_norm:.4e}, Active={chi_A}")
        
        if res_norm < 1e-10:
            print("\n[SMOKE TEST PASSED] Local SSN step converged exactly!")
            break
            
        # 4. Solve J * dw = -F
        try:
            dw = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("Jacobian is singular! FAILED.")
            break
            
        # Update
        y_k += dw[0]
        u_k += dw[1]
        p_k += dw[2]

if __name__ == "__main__":
    run_smoke_test()
