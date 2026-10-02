import numpy as np
import sys

def calculate_dofs(N, d):
    # Degrees of freedom for a hyper-cube mesh with (N+1) nodes per dimension
    # Assuming P1 elements
    return (N + 1) ** d

def run_dim_test():
    print("=== THE HJB WALL: STATE-SPACE DIMENSION EXPLOSION ===")
    print(f"{'Dim (d)':<10} | {'Res (N)':<10} | {'Unknowns (DOFs)':<25} | {'Memory (est. GB)'}")
    print("-" * 75)
    
    # Common scenarios in Control Theory
    resolutions = [10, 50, 100]
    dimensions = [1, 2, 3, 4, 5, 10, 20, 50]
    
    for d in dimensions:
        for N in resolutions:
            dofs = calculate_dofs(N, d)
            # 8 bytes per float, assuming we store just the solution vector once
            mem_gb = (dofs * 8) / (1024**3)
            
            if dofs > 1e20:
                dofs_str = f"~10^{int(np.log10(dofs))}"
                mem_str = "INFINITY"
            else:
                dofs_str = f"{dofs:,}"
                mem_str = f"{mem_gb:.2e}"
            
            print(f"{d:<10} | {N:<10} | {dofs_str:<25} | {mem_str}")
        print("-" * 75)

    print("\nCONCLUSION: Classical grid-based methods (FEM, SSN) hit the 'Memory Wall' at d=4.")
    print("For d=50 (High-Dim HJB), grid-based optimization is physically impossible.")

if __name__ == "__main__":
    run_dim_test()
