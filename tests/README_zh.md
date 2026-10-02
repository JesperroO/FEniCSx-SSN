# 🧪 SSN 边界情况与极限压力测试套件

本目录归档了一系列极度恶意的、极限的以及代数层面的压力测试。这些测试旨在探测半光滑牛顿法 (SSN) 最深层的数学边界。

在古典数值分析中，理解算法**在何处以及为何失效**，与证明其何时成功同等重要。

## 1. 幽灵状态：严格互补性缺失 (Lack of Strict Complementarity)
* **执行脚本**: `test_ssn_strict_complementarity_pure.py`
* **现象观测**: 雅可比矩阵奇异坍塌与颤振 (Chattering / 无限振荡)。
* **数学原理解析**: 在非光滑优化中，“严格互补性”要求伴随变量 $p(x)$ 不能以零梯度 ($\nabla p = 0$) 恰好停留在激活阈值 $\alpha$ 上。我们构建了一个极其恶毒的“幽灵状态”：预先求解伴随场的绝对最高峰，并将 $L^1$ 惩罚项精确设置为该峰值。这在非可微的“折痕”上制造了一个平顶山。
* **最终结果**: 剥离 Tikhonov/L2 控制正则化 ($\nu=0$) 后，克拉克广义雅可比矩阵在此处变为多值映射。算法在这些网格点上无法判定其归属。残差呈指数级爆炸，最终导致 MUMPS 直接求解器因矩阵输出 `NaN` 而彻底崩溃。
* 

```text
Iter  | Residual L2     | Active Area Vol
---------------------------------------------
0     | 1.5381e-02      | 0.00000000e+00 
1     | 9.9359e-07      | 1.31835938e-02 
2     | 1.9993e-06      | 1.36718750e-02 
3     | 4.0028e-06      | 1.51367188e-02 
4     | 8.0134e-06      | 1.51367188e-02 
5     | 1.6032e-05      | 1.51367188e-02 
6     | 3.2064e-05      | 1.51367188e-02 
7     | 6.4127e-05      | 1.51367188e-02 
8     | 1.2825e-04      | 1.51367188e-02 
9     | 2.1509e-03      | 5.41259766e-01 
10    | nan             | 1.00000000e+00 
11    | nan             | 1.00000000e+00 
12    | nan             | 1.00000000e+00 
13    | nan             | 1.00000000e+00 
14    | nan             | 1.00000000e+00 
15    | nan             | 1.00000000e+00 
16    | nan             | 1.00000000e+00 
17    | nan             | 1.00000000e+00 
18    | nan             | 1.00000000e+00 
19    | nan             | 1.00000000e+00 
20    | nan             | 1.00000000e+00 
21    | nan             | 1.00000000e+00 
22    | nan             | 1.00000000e+00 
23    | nan             | 1.00000000e+00 
24    | nan             | 1.00000000e+00 
---------------------------------------------
>>> FAILED! Chattering confirmed. The Jacobian collapsed under the ambiguity.
>>> The Active Set volume oscillates infinitely.
```

## 2. 毒丸测试：极度对流占优 (Convection-Dominated Failure)
* **执行脚本**: `test_ssn_failure.py`
* **现象观测**: 算子强制性 (Coercivity) 丧失。
* **数学原理解析**: SSN 极其依赖底层 PDE 算子强悍的平滑效应（椭圆性）。我们通过将扩散系数极速压缩至 `1e-4`，并注入一股巨大的对角线对流风场 ($\mathbf{w} \cdot \nabla y$)，给系统喂下了一颗毒丸。由于吉布斯现象 (数值震荡)，伴随变量 $p$ 彻底失去平滑性。活跃集指示器开始在迭代间疯狂横跳。
* **最终结果**: 算法残差卡死在 $10^{-4}$ 级别，陷入永远无法收敛的死循环。

```text
=== FAILURE DEMONSTRATION: NON-ELLIPTIC (CONVECTION-DOMINATED) PDE ===
Attempting to run SSN on an advection-dominated equation where 
the strong smoothing effect of the Laplacian is destroyed.
--------------------------------------------------
Iter  | Residual L2     | Status
--------------------------------------------------
0     | 1.3819e-02      | Struggling...
1     | 1.0719e+00      | Struggling...
2     | 1.2087e-01      | Struggling...
3     | 2.4275e-02      | Struggling...
4     | 5.7267e-03      | Struggling...
5     | 1.6401e-03      | Struggling...
6     | 6.1187e-04      | Struggling...
7     | 6.0035e-04      | Struggling...
8     | 4.6784e-04      | Struggling...
9     | 5.2054e-04      | Struggling...
10    | 4.4837e-04      | Struggling...
11    | 5.0775e-04      | Struggling...
12    | 4.3207e-04      | Struggling...
13    | 4.8772e-04      | Struggling...
14    | 4.2579e-04      | Struggling...
```

## 3. 终极滤波：纯随机噪声与阶跃断层
* **执行脚本**: `test_ssn_random_noise_export.py`
* **现象观测**: 对病态脏数据的绝对鲁棒性。
* **数学原理解析**: 深度学习模型在面对纯噪声时极易过拟合，而古典 PDE 约束优化将逆拉普拉斯算子作为一个极其强大的低通滤波器。它无视了所有高频垃圾信息，精准提取出了宏观能量块。
* **最终结果**: 即使被喂入毫无规律的均匀随机雪花噪声，算法依然在精确的 5 步内收敛至机器精度 ($10^{-14}$)。

<p align="center">
  <img src="../assets/noise_visualization.png" alt="Noise Filter Result" width="800">
</p>

```text
=== STRESS TEST: SSN WITH RANDOM NOISE & NON-SMOOTH TARGET ===
Target state y_d is filled with uniform random noise and a sharp discontinuous block.
--------------------------------------------------
Iter  | Residual L2     | Active Area Vol
--------------------------------------------------
0     | 4.0488e-02      | 0.000000       
1     | 1.0629e+00      | 0.603149       
2     | 9.2425e-02      | 0.374878       
3     | 2.3456e-03      | 0.355591       
4     | 3.0957e-06      | 0.354858       
5     | 1.3624e-14      | 0.354858       
--------------------------------------------------
[SUCCESS] SSN converged even with pure random noise in 5 steps!
```

## 4. 0D 代数冒烟测试
* **执行脚本**: `test_ssn_smoke_fixed.py`
* **现象观测**: 超二次收敛性的纯粹验证。
* **数学原理解析**: 剥离所有 PDE 空间导数（拉普拉斯算子），在单一数学点上直接模拟 Primal-Dual Active Set (PDAS) KKT 条件的精确 3x3 块代数系统。
* **最终结果**: 在精确的 2 步内残差清零 ($0.0000$)。这是对我们基于 UFL 符号推导的广义雅可比矩阵在代数上绝对封闭且对称的最终铁证。

```text
--- Running Rigorous Local SSN Smoke Test ---
Iter 0: [y=0.0000, u=0.0000, p=0.0000], Residual=1.5000e+00, Active=0.0
Iter 1: [y=0.0000, u=0.0000, p=1.5000], Residual=5.0000e-01, Active=1.0
Iter 2: [y=0.5000, u=0.5000, p=1.0000], Residual=0.0000e+00, Active=1.0

[SMOKE TEST PASSED] Local SSN step converged exactly!
```

## 5. 极端正则性缺失：狄拉克冲击载荷 (Dirac Delta Load)
* **执行脚本**: `test_ssn_dirac_load.py`
* **现象观测**: 近奇异载荷下的收敛性——以及一个反直觉的加速现象。
* **数学原理解析**: 我们注入了一个极端病态的目标状态 $y_d = A \cdot e^{-\sigma|x-x_c|^2}$，将 $\sigma$ 推至 5000，形成仅被约 3 个网格单元分辨的近狄拉克点源。
* **最终结果**: 近狄拉克情形仅用 **3 步**收敛——比平滑参考还要快。揭示了“狄拉克悖论”：当目标收缩为点源时，$L^1$ 惩罚让追逐近奇异尖峰的代价极高，最优控制因此退缩到极小的激活岛。更小、更紧凑的激活集意味着更简单的雅可比结构。

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
  ------  ----------------------  --------------
  0       2.768052e-02          
  1       6.246547e-01          
  2       7.527076e-02          
  3       2.473157e-03          
  4       5.513836e-06          
  5       2.872670e-10           <-- converged

  Result  : CONVERGED in 5 iterations
  Active set vol  : 0.261108
  Wall time       : 2.9s

[sigma=500   (sharp spike, ~6 cells wide) ]
  Peak value at mesh centre: y_d(0.5,0.5) = 63
  Gaussian width (1/sqrt(sigma))         = 0.0447
  Mesh spacing h                         = 0.0078
  Cells across peak (approx)             = 11.4

  Iter    Residual ||F||          Active-set vol
  ------  ----------------------  --------------
  0       2.744783e-02          
  1       1.296879e-01          
  2       1.791537e-02          
  3       5.093094e-04          
  4       1.620535e-06          
  5       5.006561e-10           <-- converged

  Result  : CONVERGED in 5 iterations
  Active set vol  : 0.091125
  Wall time       : 2.7s

[sigma=5000  (near-Dirac, ~2 cells wide)  ]
  Peak value at mesh centre: y_d(0.5,0.5) = 200
  Gaussian width (1/sqrt(sigma))         = 0.0141
  Mesh spacing h                         = 0.0078
  Cells across peak (approx)             = 3.6

  Iter    Residual ||F||          Active-set vol
  ------  ----------------------  --------------
  0       2.638675e-02          
  1       6.240880e-03          
  2       1.333188e-05          
  3       8.077380e-09           <-- converged

  Result  : CONVERGED in 3 iterations
  Active set vol  : 0.005066
  Wall time       : 1.9s

======================================================================
  VERDICT
======================================================================
  The inverse Laplacian in the adjoint PDE regularises all three
  targets into H^1-smooth adjoint fields. SSN convergence is
  governed by the operator's monotonicity (guaranteed by nu>0),
  not by the smoothness of the data. Local Dirac-like spikes
  cannot pollute the global residual descent path.
======================================================================
```

## 6. 充滿棱角的障礙物：空間變化控制閾值 (Jagged Obstacle Wall)
* **執行腳本**: `test_ssn_jagged_obstacle.py`
* **現象觀測**: 雙重疊加非光滑層下的完全收斂不變性。
* * **数学原理解析**: 将标准 SSN 的固定阈值 $\alpha$ 替换为空间变化的 $\alpha(x)$，使其等高线形成充满不可导折痕的障碍壁。最优控制必须去"撞击"一面本身就非光滑的约束边界——在 SSN 半光滑算子结构的基础上，再叠加一层非光滑约束。测试三种障碍几何形态：平坦（常数参考基线）、菱形帐篷（沿 $L^1$-球边界的 4 条尖锐折痕）、华夫格（$|\sin(4\pi x_1)|\cdot|\sin(4\pi x_2)|$，16 条交叉非光滑脊线）。
* **最终结果** (Docker 实测，128×128 网格，`dolfinx:v0.7.3`):

  | 障碍形状 | 几何描述 | 牛顿步数 | 激活集体积 |
  |----------|---------|----------|-----------|
  | 平坦 | $\alpha = 0.05$（常数参考） | **5** | 0.1711 |
  | 菱形 | 菱形上的帐篷函数，4 条折痕 | **5** | 0.3253 |
  | 华夫格 | $|\sin|{\times}|\sin|$，16 条非光滑脊线 | **5** | 0.1758 |

  **障碍物悖论**：SSN 对障碍壁的棱角几何完全视而不见。PDAS 更新规则 $\chi_\mathcal{A} = \mathbf{1}_{|p|>\alpha(x)}$ 将 $\alpha(x)$ 的空间非光滑性直接吸收进激活集的布尔指示器中。一旦激活集确定，雅可比块 $(\chi_\mathcal{A}/\nu)\,\delta p\,v_u$ 对三种几何形态在结构上完全相同。16 条非光滑脊线对牛顿迭代步数的贡献为零。

```text
========================================================================
  TEST: JAGGED OBSTACLE WALL — NON-SMOOTH SPATIALLY-VARYING alpha(x)
  Grid: 128x128  |  nu=1e-3  |  y_d = two Gaussian hotspots
========================================================================

[FLAT     (alpha = 0.05 constant)]
  Four-fold symmetric, single smooth boundary.

  Iter    Residual ||F||            Active-set vol
  ------  ------------------------  --------------
  0       9.788064e-03            
  1       5.403914e-02            
  2       6.741045e-03            
  3       1.500612e-04            
  4       3.612382e-07            
  5       2.229697e-11             <-- converged

  Result         : CONVERGED in 5 iterations
  Active-set vol : 0.171143
  Wall time      : 2.7s

[DIAMOND  (tent over rhombus, 4 sharp folds)]
  Active set must trace the L1-ball creases of alpha(x).

  Iter    Residual ||F||            Active-set vol
  ------  ------------------------  --------------
  0       9.788064e-03            
  1       1.514244e-01            
  2       2.175073e-02            
  3       9.823113e-04            
  4       3.707038e-06            
  5       4.739453e-09             <-- converged

  Result         : CONVERGED in 5 iterations
  Active-set vol : 0.325256
  Wall time      : 3.3s

[WAFFLE   (|sin|*|sin|, 16 non-smooth ridges)]
  Maximal crease density: all 16 ridge intersections active simultaneously.

  Iter    Residual ||F||            Active-set vol
  ------  ------------------------  --------------
  0       9.788064e-03            
  1       9.299382e-02            
  2       1.056405e-02            
  3       1.413302e-04            
  4       8.108270e-07            
  5       2.393533e-10             <-- converged

  Result         : CONVERGED in 5 iterations
  Active-set vol : 0.175781
  Wall time      : 3.4s

========================================================================
  VERDICT
========================================================================
  All three cases converge in exactly 5 iterations — the jaggedness
  of the obstacle wall is completely invisible to the SSN.
  chi_A = 1_{|p|>alpha(x)} absorbs alpha(x)'s non-smoothness into
  the active-set indicator. Once the set is determined, the Jacobian
  block (chi_A/nu)*dp*v_u is structurally identical for all three.
  The obstacle wall paradox: SSN does not 'see' the wall's jagged
  geometry — it only sees a Boolean on/off decision at each DOF.
========================================================================
```

## 7. L 型奇異點與拓撲撕裂 (The L-Shape Singularity)
* **執行腳本**: `test_ssn_l_shape.py`
* **現象觀測**: 對空間幾何奇異性與 $H^2$ 正則性喪失的絕對免疫。
* **數學原理解析**: 通過在正方形網格中挖去四分之一，我們在 $(0.5, 0.5)$ 處人為製造了一個向內的直角。在 PDE 理論中，此處的梯度將趨於無窮大（產生 $r^{-1/3}$ 的物理奇異性），徹底摧毀解的 $H^2$ 完美平滑度。我們將目標靶點精確放置在奇異點上，迫使活躍集 (Active Set) 跨越這道數學裂痕。
* **最終結果**: **4 步精確收斂！** SSN 算法對此奇異性完全免疫。因為 PDAS 的判斷邏輯僅依賴於伴隨變量 $p$ 的絕對值大小（$L^\infty$ 有界性），而不依賴其空間梯度 $\nabla p$。

```text
Info    : Meshing 1D...
Info    : [  0%] Meshing curve 1 (Line)
Info    : [ 20%] Meshing curve 2 (Line)
Info    : [ 40%] Meshing curve 3 (Line)
Info    : [ 50%] Meshing curve 4 (Line)
Info    : [ 70%] Meshing curve 5 (Line)
Info    : [ 90%] Meshing curve 6 (Line)
Info    : Done meshing 1D (Wall 0.00199704s, CPU 0.000604s)
Info    : Meshing 2D...
Info    : Meshing surface 1 (Plane, Frontal-Delaunay)
Info    : Done meshing 2D (Wall 0.143473s, CPU 0.138412s)
Info    : 4119 nodes 8242 elements
=== THE L-SHAPE SINGULARITY TEST ===
Constructing L-Shape mesh with a re-entrant corner at (0.5, 0.5)...
Total Degrees of Freedom: 12357
-------------------------------------------------------
Iter  | Residual L2     | Active Area Vol
-------------------------------------------------------
0     | 1.0252e-02      | 0.000000       
1     | 7.2949e-02      | 0.321112       
2     | 3.4214e-03      | 0.247848       
3     | 9.2366e-06      | 0.245855       
4     | 9.5865e-09      | 0.245855       
-------------------------------------------------------
[SUCCESS] L-Shape Singularity Conquered in 4 steps!
Singularity visualization saved to output_2d/l_shape_control.xdmf
```

## 8. 抽真空測試：正則化極限 (The Tikhonov Vacuum)
* **執行腳本**: `test_ssn_moreau_yosida_limit.py`
* **現象觀測**: 雅可比矩陣條件數的指數級爆炸與坍塌。
* **數學原理解析**: SSN 處理 $L^1$ 控制的超二次收斂，底層的強制性來自 Tikhonov/L2 參數 $\nu$。我們將 $\nu$ 從 $10^{-2}$ 逐步抽向機器精度，迫使系統從良態的希爾伯特空間逼近病態的測度空間極限。
* **最終結果**: 隨著 $\nu \le 10^{-8}$，**算法徹底崩塌！** 殘差瘋狂飆升至 $10^{17}$ 級別。這精確定位了該算法的極限武器射程，證明了在沒有正則化保護傘的真空中，牛顿法将瞬间死亡。

```text
=== THE TIKHONOV VACUUM TEST ===
Progressively collapsing the regularization parameter nu to machine epsilon.
Testing the raw ill-conditioning of the semi-smooth Jacobian.
-----------------------------------------------------------------
nu (Penalty)    | Iterations      | Final Residual      
-----------------------------------------------------------------
1.0e-02         | 4               | 2.6896e-12          
1.0e-04         | 7               | 1.0246e-10          
1.0e-06         | 14              | 7.1491e-10          
1.0e-08         | FAIL (No Conv)  | 1.7995e-03          
1.0e-10         | FAIL (No Conv)  | 5.9963e+01          
1.0e-12         | FAIL (No Conv)  | 1.3870e+06          
1.0e-14         | FAIL (No Conv)  | 3.3384e+12          
1.0e-16         | FAIL (No Conv)  | 2.3237e+17
```

## 9. 絕對的 Bang-Bang 狂暴狀態 (嚴格箱式約束)
* **執行腳本**: `test_ssn_bang_bang.py`
* **現象觀測**: 控制變量在物理絕境下的極端邊界飽和與零顫振。
* **數學原理解析**: 取消 $L^1$ 懲罰，轉而施加極端死板的箱式約束 $u \in [-1, 1]$。目標狀態 $y_d$ 被設定為高頻正弦波。由於控制力被死死卡在 $\pm 1$，系統根本不可能達到目標。最優控制被逼入純粹的 "Bang-Bang" 絕境——在 -1 和 1 之間進行瘋狂、毫無過渡的階躍跳變。
* **最終結果**: **5 步精確收斂。** 算法在面對物理斷層時展現了極致的冷血。高達 **99.64%** 的網格區域被死死按在絕對邊界極限上。

* **可視化說明**: 您可能會預期控制量看起來像正弦波，因為目標 $y_d$ 是正弦波。然而，由於目標振幅 (100) 遠超控制能力 ([-1, 1])，算法將控制量直接“拍”在絕對極限上以最小化 $L^2$ 誤差，將平滑正弦輸入變成了棋盤方波。

<p align="center">
  <img src="../assets/bang_bang_visualization.png" alt="Bang-Bang Control" width="600">
</p>

```text
========================================================================
  TEST: THE ABSOLUTE BANG-BANG RAMPAGE (STRICT BOX CONSTRAINTS)
  Grid: 128x128  |  u in [-1, 1]  |  y_d = 100 * high-frequency sine
========================================================================

Iter   | Residual ||F||         | Active Set (Upper / Lower)
-----------------------------------------------------------------
0      | 3.867275e-01           | 0.0000 / 0.0000
1      | 1.918337e+00           | 0.4982 / 0.4982
2      | 5.866997e-06           | 0.4982 / 0.4982
3      | 7.899041e-08           | 0.4982 / 0.4982
4      | 1.008468e-09           | 0.4982 / 0.4982
5      | 1.367577e-11           | 0.4982 / 0.4982

  Result: CONVERGED in 5 iterations
  Total Active Volume: 99.64% of the domain is saturated at absolute limits.
```

## 10. 物理屏障的絕境：硬性狀態約束 (State Constraints)
* **執行腳本**: `test_ssn_state_constraints.py`
* **現象觀測**: 活躍集顫振與正則性缺失導致的收斂失敗。
* **數學原理解析**: 對狀態變量施加硬性上限 $y(x) \le 0.2$，而目標狀態 $y_d$ 峰值為 $1.0$。在狀態約束優化中，拉格朗日乘子不再是普通函數，而是一個 **Radon 測度 (Measure)**。最優化系統變得極其病態。
* **最終結果**: **收斂失敗 (持續振盪)。** 算法進入了永無止境的顫振循環。這種“極限環”現象證明了純粹 SSN 的絕對邊界：在沒有阻尼或延拓法的輔助下，SSN 無法自行處理狀態-測度耦合的奇異性。

<p align="center">
  <img src="../assets/state_constraints_visualization.png" alt="State Constraints Struggle" width="600">
</p>

```text
========================================================================
  TEST: THE DEADLY BARRIER (HARD STATE CONSTRAINTS)
  Grid: 128x128  |  Target peak = 1.0  |  Constraint y <= 0.2
========================================================================

Iter   | Residual ||F||         | Violation Max
-----------------------------------------------------------------
0      | 1.957804e-03           | 0.000000
1      | 1.333978e+02           | 76.609007
2      | 2.476123e-01           | 0.114661
3      | 1.331828e-02           | 0.002990
4      | 1.664907e-02           | 0.003986
5      | 3.014046e-02           | 0.008490
6      | 4.417825e-01           | 0.162331
7      | 5.738437e+00           | 2.814582
8      | 1.731331e-02           | 0.008687
9      | 2.585855e+01           | 13.415403
10     | 9.789785e-03           | 0.006314
11     | 1.942196e-02           | 0.007487
12     | 5.773712e-02           | 0.025918
13     | 1.301051e+01           | 6.630620
14     | 1.644378e-03           | 0.001175
15     | 1.422907e-02           | 0.005121
16     | 3.059667e-02           | 0.012582
17     | 1.117118e-01           | 0.032653
18     | 6.693990e+00           | 3.318078
19     | 1.847308e-03           | 0.001048
20     | 1.551800e-02           | 0.005023
21     | 4.329597e-02           | 0.020779
22     | 2.480889e-01           | 0.069136
23     | 5.688314e+00           | 2.786260
24     | 1.949845e-03           | 0.001000
```

## 11. 非線性物理的重壓：p-Laplacian 爆炸的取證調查
* **執行腳本**: `test_ssn_nonlinear_explosion.py`
* **現象觀測**: 雙重壓力收斂失效 (代數過載 + 非線性超調)。
* **數學原理解析**: 將線性拉普拉斯算子替換為高度非線性的 **p-Laplacian ($p=4$)**。該算子具有高度“侵略性”——其剛度隨梯度呈立方倍率縮放。這迫使 SSN 求解一個“物理”（PDE 係數）與“邏輯”（活躍集）在每次迭代中都在劇烈震盪的系統。

### 🕵️ 工程試錯與調試紀實 (Engineering History)
1. **手動推導的傲慢 (元數衝突)**: 最初試圖手動推導混合空間 Hessian，因 UFL 元數衝突 (`ValueError`) 而失敗。
2. **戰略轉向 (混合自動微分)**: 轉向“混合自動微分”策略——利用 `ufl.derivative` 處理光滑的非線性 PDE 块，同時保留手寫的傾斜導數 (Slant Derivative) 來處理非光滑塊。
3. **最終裁決 (牛頓法的自殺)**: 即便雅可比矩陣在數學上是完美的，物理定律依然勝出。殘差從 $10^{-2}$ 瘋狂飆升至 $10^{11}$。這為 **"Networks" (神經算子)** 部分提供了終極的正當性。

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
-----------------------------------------------------------------
0      | 2.763842e-02           | 0.000000
1      | 2.736429e+01           | 0.334473
2      | 1.507816e+04           | 0.307861
3      | 7.762057e+05           | 0.527100
4      | 3.685367e+08           | 0.849121
5      | 1.158375e+08           | 0.620117
6      | 4.308585e+07           | 0.945312
7      | 6.782532e+11           | 0.923096
8      | 2.009639e+11           | 0.934326
9      | 5.954486e+10           | 0.957764
10     | 1.764292e+10           | 0.938721
11     | 5.228324e+09           | 0.971924
```

## 12. 维度深渊：网格无关性 (Mesh Independence)
* **执行脚本**: `test_ssn_mesh_independence.py`
* **现象观测**: 迭代次数在维度指数级爆炸下的绝对不变性。
* **数学原理解析**: 连续算子理论从数学底层保证：当离散网格 $h \to 0$ 时，收敛至给定容差所需的迭代次数必须趋近于一个有限的常整数。
* **最终结果**: 迭代次数从 12k 到 790k 自由度全程死死钉在 **5 步**，证明了 Banach 空间牛顿法的本质特性。

```text
=== THE ABYSS OF MESH INDEPENDENCE ===
Testing SSN convergence across exponentially growing dimensions...
-------------------------------------------------------
Mesh N x N   | Degrees of Freedom   | Newton Iterations
-------------------------------------------------------
64x64        | 12675                | 5              
128x128      | 49923                | 5              
256x256      | 198147               | 5              
512x512      | 789507               | 5
```

---
*语言切换: [Read in English](README.md) | [阅读中文版](README_zh.md)*
