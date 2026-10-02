# 🧪 SSN Edge-Case & Stress Testing Suite

In classical numerical analysis, understanding *where* and *why* an algorithm fails is as important as knowing when it succeeds.

## 1. The Ghost State: Lack of Strict Complementarity
* **Script**: `test_ssn_strict_complementarity_pure.py`
* **Phenomenon**: Jacobian Singularity & Chattering (Infinite Oscillation).
* **Mathematical Background**: In non-smooth optimization, "Strict Complementarity" requires that the adjoint variable $p(x)$ does not perfectly align with the threshold $\alpha$ with a zero gradient ($\nabla p = 0$). We construct a "Ghost State" by pre-solving for the peak of the adjoint field and setting the $L^1$ penalty exactly to this peak, effectively creating a flat plateau on the non-differentiable "kink" of the active set condition.
* **Result**: Without the Tikhonov/L2 control regularization ($\nu=0$), the Clarke generalized Jacobian becomes multi-valued. The algorithm is unable to decide whether these nodes belong to the active or inactive set. The residual explodes exponentially, and the MUMPS solver ultimately crashes with a `NaN` singularity matrix.

```text
Iter  | Residual L2     | Active Area Vol

## 2. The Poison Pill: Convection-Dominated Failure
* **Script**: `test_ssn_failure.py`
* **Phenomenon**: Loss of Matrix Coercivity.
* **Mathematical Background**: SSN relies heavily on the strong smoothing effect (ellipticity) of the underlying PDE operator. We poison the Laplacian by shrinking the diffusion coefficient to `1e-4` and injecting a massive diagonal convective wind field ($\mathbf{w} \cdot \nabla y$). 
* **Result**: The adjoint variable $p$ loses its smoothness due to the Gibbs phenomenon (numerical oscillation). The active set indicator $\chi_{\mathcal{A}}$ chatters wildly between iterations. Convergence halts at $10^{-4}$, and the algorithm is permanently trapped in an infinite loop.

```text
=== FAILURE DEMONSTRATION: NON-ELLIPTIC (CONVECTION-DOMINATED) PDE ===
Attempting to run SSN on an advection-dominated equation where 
the strong smoothing effect of the Laplacian is destroyed.

## 3. The Noise Filter: Pure Random Noise & Discontinuities
* **Script**: `test_ssn_random_noise_export.py`
* **Phenomenon**: Absolute Robustness against ill-posed data.
* **Mathematical Background**: Deep learning models overfit when presented with pure noise. Classical PDE-constrained optimization leverages the inverse Laplacian as a powerful low-pass filter. 
* **Result**: When fed with uniform random noise and massive step discontinuities, the algorithm simply ignores the high-frequency garbage. It extracts the underlying low-frequency macroscopic energy block and converges flawlessly in exactly 5 steps to machine precision ($10^{-14}$).

<p align="center">
  <img src="../assets/noise_visualization.png" alt="Noise Filter Result" width="800">
</p>


```text
=== STRESS TEST: SSN WITH RANDOM NOISE & NON-SMOOTH TARGET ===
Target state y_d is filled with uniform random noise and a sharp discontinuous block.

## 4. The 0D Algebraic Smoke Test
* **Script**: `test_ssn_smoke_fixed.py`
* **Phenomenon**: Superlinear Convergence Verification.
* **Mathematical Background**: Strips away all PDE spatial derivatives (the Laplacian) and simulates the exact 3x3 block algebraic system of the Primal-Dual Active Set KKT conditions at a single mathematical point.
* **Result**: Converges in exactly 2 steps to $0.0000$. This serves as the ultimate proof that our symbolic UFL derivation is algebraically closed and perfectly symmetric.

```text

## 5. Extreme Regularity Deficiency: Dirac Delta Load
* **Script**: `test_ssn_dirac_load.py`
* **Phenomenon**: Convergence under near-singular, non-physical forcing — and a paradoxical speed-up.
* **Mathematical Background**: We inject a pathological target state $y_d = A \cdot e^{-\sigma|x-x_c|^2}$ where $\sigma$ is driven to 5000, producing a near-Dirac point source resolved by only ~3 mesh cells. The local peak stress $y_d(x_c) = 200$ creates extreme curvature in the state equation. The key theoretical question: can local numerical blow-up near the spike contaminate the global residual descent path?
* **Result**: The near-Dirac case converges in only **3 steps** — *faster* than the smooth reference. The Dirac paradox: as the target concentrates to a point source, the $L^1$ penalty makes it too expensive to chase a near-singular spike, so the optimal control retreats to a tiny island. A smaller, crisper active set means a simpler Jacobian structure.

<p align="center">
  <img src="../assets/dirac_convergence.png" alt="Dirac Convergence Acceleration" width="800">
</p>


```text
======================================================================
  TEST: EXTREME REGULARITY DEFICIENCY — DIRAC DELTA LOAD
  Grid: 128x128  |  alpha=0.05  |  nu=1e-3
======================================================================

[sigma=50    (smooth Gaussian, reference)]
  Peak value at mesh centre: y_d(0.5,0.5) = 20
  Gaussian width (1/sqrt(sigma))         = 0.1414
  Mesh spacing h                         = 0.0078
  Cells across peak (approx)             = 36.2

  Iter    Residual ||F||          Active-set vol

## 6. Jagged Obstacle Wall: Non-Smooth Spatially-Varying Control Threshold
* **Script**: `test_ssn_jagged_obstacle.py`
* **Phenomenon**: Complete convergence invariance under a second stacked layer of non-smoothness.
* **Mathematical Background**: The standard SSN threshold $\alpha$ is replaced by a spatially-varying $\alpha(x)$ whose level sets form walls with non-differentiable creases. Three obstacle geometries are tested: flat (constant reference), a diamond tent with 4 sharp folds along the $L^1$-ball boundary, and a waffle grid with 16 intersecting non-smooth ridges.
* **Result**: The **Obstacle Wall Paradox**: SSN is completely blind to the jaggedness of the obstacle. The PDAS update rule absorbs the spatial non-smoothness of $\alpha(x)$ into the active-set Boolean indicator. Once the set is determined, the Jacobian block is structurally identical for all three geometries.

```text
========================================================================
  TEST: JAGGED OBSTACLE WALL — NON-SMOOTH SPATIALLY-VARYING alpha(x)
  Grid: 128x128  |  nu=1e-3  |  y_d = two Gaussian hotspots
========================================================================

[FLAT     (alpha = 0.05 constant)]
  Four-fold symmetric, single smooth boundary.

  Iter    Residual ||F||            Active-set vol

## 7. The L-Shape Singularity (Topological Tearing)
* **Script**: `test_ssn_l_shape.py`
* **Phenomenon**: Absolute immunity to spatial singularities and loss of $H^2$ regularity.
* **Mathematical Background**: By removing a quadrant from the unit square domain, we create a re-entrant corner at $(0.5, 0.5)$. In PDE theory, gradients diverge to infinity at this corner ($r^{-1/3}$ singularity), destroying the classical $H^2$ smoothness of the solution. We force the active set to cross directly over this mathematical tear.
* **Result**: **Converged in 4 Iterations.** The algorithm completely ignores the singularity. Because the PDAS condition relies solely on the value of $p$ ($L^\infty$ bounding) rather than its spatial derivatives ($\nabla p$), the fractured geometry has zero impact on the Jacobian's Newton descent.

```text
=== THE L-SHAPE SINGULARITY TEST ===
Constructing L-Shape mesh with a re-entrant corner at (0.5, 0.5)...
Total Degrees of Freedom: 12357

## 8. The Tikhonov Vacuum (Regularization Limit)
* **Script**: `test_ssn_moreau_yosida_limit.py`
* **Phenomenon**: Exponential Jacobian Ill-Conditioning.
* **Mathematical Background**: The superlinear convergence of SSN in $L^1$ control is theoretically predicated on the strict coercivity provided by the Tikhonov/L2 parameter $\nu$. We progressively collapse $\nu$ toward machine epsilon to observe the transition from a well-posed Hilbert space to an ill-posed measure-space limit.
* **Result**: The residual explodes from $10^{-3}$ up to $10^{17}$. The matrix becomes numerically singular as the non-smooth `(p-alpha)/nu` term approaches infinity, proving that SSN without a mathematical "safety net" is unconditionally unstable.

```text
=== THE TIKHONOV VACUUM TEST ===
Progressively collapsing the regularization parameter nu to machine epsilon.
Testing the raw ill-conditioning of the semi-smooth Jacobian.

## 9. The Absolute Bang-Bang Rampage (Box Constraints)
* **Script**: `test_ssn_bang_bang.py`
* **Phenomenon**: Extreme Box Constraints clipping against an unreachable target.
* **Result**: **Converged in 5 Iterations.** The algorithm does not chatter at the physical discontinuities. A staggering **99.64%** of the physical domain is locked onto the absolute limits.

* **Visualization Note**: The target amplitude (100) is far beyond the control's capacity ([-1, 1]). The algorithm "slams" the control to the absolute limits, turning the smooth sine input into a binary checkerboard output.

<p align="center">
  <img src="../assets/bang_bang_visualization.png" alt="Bang-Bang Control" width="600">
</p>

```text
========================================================================
  TEST: THE ABSOLUTE BANG-BANG RAMPAGE (STRICT BOX CONSTRAINTS)
  Grid: 128x128  |  u in [-1, 1]  |  y_d = 100 * high-frequency sine
========================================================================

Iter   | Residual ||F||         | Active Set (Upper / Lower)

## 10. The Deadly Barrier: Hard State Constraints
* **Script**: `test_ssn_state_constraints.py`
* **Phenomenon**: Active Set Chattering & Lack of Regularity.
* **Result**: **Convergence Failed (Oscillation).** Demonstrates the absolute boundary of simple SSN: without damping or continuation methods, pure SSN cannot resolve the singularity of state-measure coupling.

<p align="center">
  <img src="../assets/state_constraints_visualization.png" alt="State Constraints Struggle" width="600">
</p>

```text
========================================================================
  TEST: THE DEADLY BARRIER (HARD STATE CONSTRAINTS)
  Grid: 128x128  |  Target peak = 1.0  |  Constraint y <= 0.2
========================================================================

Iter   | Residual ||F||         | Violation Max

## 11. The Nonlinear Crush: A Forensic Investigation into p-Laplacian Explosion

* **Script**: `test_ssn_nonlinear_explosion.py`
* **Phenomenon**: Dual-Stress Convergence Failure (Algebraic Overload + Nonlinear Overshoot).
* **Mathematical Background**: We replace the linear Laplacian with the **p-Laplacian ($p=4$)**: $-\nabla \cdot (|\nabla y|^2 \nabla y) + y = u$. This operator is "aggressive"—its stiffness scales cubically with the gradient. This forces the SSN to solve for a system where the "physics" (PDE coefficients) and the "logic" (Active Sets) are both shifting violently in every iteration.

### 🕵️ The Engineering Trial-and-Error Process

This section documents the struggle to even bring this test to life, revealing the hidden complexities of modern FEM symbolic compilers.

#### Phase 1: The Manual Hubris (Arity Conflict)
We initially attempted to manually derive the second-order Fréchet derivative (Hessian) of the p-Laplacian in a mixed function space. 
* **The Goal**: Full control over the Jacobian structure to ensure "pure" SSN compliance.
* **The Failure**: FEniCSx's UFL compiler threw a `ValueError: too many values to unpack (expected 2)`. 
* **The Reason**: UFL's arity checker cannot easily handle the cross-product of trial/test functions when nested inside high-order tensor contractions of a mixed space. The symbolic graph became too deep for the standard assembly pattern.

#### Phase 2: The Strategic Pivot (Mixed Autodiff)
We abandoned the "pure manual" approach and implemented a hybrid strategy:
* **The Fix**: We utilized `ufl.derivative()` for the smooth nonlinear PDE blocks (where Autodiff is mathematically rigorous) while maintaining the manual **Slant Derivative** indicators for the non-smooth $L^1$ control blocks.
* **The Result**: The code finally compiled and the JIT compiler generated the C++ kernels.

#### Phase 3: The Convergence Disaster (Newton's Suicide)
Despite having a "perfect" Jacobian, the physics proved too powerful for the algorithm's base configuration.
* **Observation**: In Iteration 1, the residual was $10^1$. By Iteration 7, it reached $10^{11}$.
* **Conclusion**: This is a classic **Newton Overshoot**. Standard SSN assumes a "full step" ($\Delta w = 1$). For the $p=4$ operator, the linearization's trust region is extremely small. One "logical" flip of an active-set node causes a "physical" ripple that throws the gradient into an unrecoverable divergence.

### 📊 Forensic Evidence

<p align="center">
  <img src="../assets/nonlinear_failure_evolution.png" alt="Nonlinear Divergence Evolution" width="900">
</p>

```text
========================================================================
  CRIMINAL INVESTIGATION: THE NONLINEAR SSN EXPLOSION
  Operator: -div(|grad y|^2 grad y) + y = u  (p=4)
  Goal: Document the collapse of Newton descent under dual-stress.
========================================================================

Iter   | Residual ||F||         | Active Set Vol

## 12. The Abyss of Mesh Independence
* **Script**: `test_ssn_mesh_independence.py`
* **Phenomenon**: Iteration count invariance under extreme dimensional scaling.
* **Result**: The iteration count is pinned at exactly **5** from 12k to 790k DOFs.

```text
=== THE ABYSS OF MESH INDEPENDENCE ===
Testing SSN convergence across exponentially growing dimensions...

---
*Language: [English](README.md) | [中文](README_zh.md)*
