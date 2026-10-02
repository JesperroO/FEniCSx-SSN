import os

def read_log(n):
    fname = f"tmp_log_{n}.txt"
    if os.path.exists(fname):
        with open(fname, 'r') as f: return f.read().strip()
    return f"[未找到日志文件 logs_{n}]"

logs = {i: read_log(i) for i in range(1, 13)}

content = f"""# 🧪 SSN 边界情况与极限压力测试套件

本目录归档了一系列极度恶意的、极限的以及代数层面的压力测试。这些测试旨在探测半光滑牛顿法 (SSN) 最深层的数学边界。

在古典数值分析中，理解算法**在何处以及为何失效**，与证明其何时成功同等重要。

## 1. 幽灵状态：严格互补性缺失 (Lack of Strict Complementarity)
* **执行脚本**: `test_ssn_strict_complementarity_pure.py`
* **现象观测**: 雅可比矩阵奇异坍塌与颤振 (Chattering / 无限振荡)。
* **数学原理解析**: 在非光滑优化中，“严格互补性”要求伴随变量 $p(x)$ 不能以零梯度 ($\\nabla p = 0$) 恰好停留在激活阈值 $\\alpha$ 上。我们构建了一个极其恶毒的“幽灵状态”：预先求解伴随场的绝对最高峰，并将 $L^1$ 惩罚项精确设置为该峰值。这在非可微的“折痕”上制造了一个平顶山。
* **最终结果**: 剥离 Tikhonov/L2 控制正则化 ($\\nu=0$) 后，克拉克广义雅可比矩阵在此处变为多值映射。算法在这些网格点上无法判定其归属。残差呈指数级爆炸，最终导致 MUMPS 直接求解器因矩阵输出 `NaN` 而彻底崩溃。

```text
{logs[1]}
```

## 2. 毒丸测试：极度对流占优 (Convection-Dominated Failure)
* **执行脚本**: `test_ssn_failure.py`
* **现象观测**: 算子强制性 (Coercivity) 丧失。
* **数学原理解析**: SSN 极其依赖底层 PDE 算子强悍的平滑效应（椭圆性）。我们通过将扩散系数极速压缩至 `1e-4`，并注入一股巨大的对角线对流风场 ($\\mathbf{{w}} \\cdot \\nabla y$)，给系统喂下了一颗毒丸。
* **最终结果**: 由于吉布斯现象 (数值震荡)，伴随变量 $p$ 彻底失去平滑性。激活集指示器 $\\chi_{{\\mathcal{{A}}}}$ 开始在迭代间疯狂横跳。算法残差卡死在 $10^{{-4}}$ 级别，陷入永远无法收敛的死循环。

```text
{logs[2]}
```

## 3. 终极滤波：纯随机噪声与阶跃断层
* **执行脚本**: `test_ssn_random_noise_export.py`
* **现象观测**: 对病态脏数据的绝对鲁棒性。
* **数学原理解析**: 深度学习模型在面对纯噪声时极易过拟合，而古典 PDE 约束优化将逆拉普拉斯算子作为一个极其强大的低通滤波器。
* **最终结果**: 即使被喂入毫无规律的均匀随机雪花噪声和巨大的阶跃断层，算法直接无视了所有高频垃圾信息，精准提取出了宏观能量块，并在精确的 5 步内收敛至机器精度 ($10^{{-14}}$)。

<p align="center">
  <img src="../assets/noise_visualization.png" alt="Noise Filter Result" width="800">
</p>

```text
{logs[3]}
```

## 4. 0D 代数冒烟测试
* **执行脚本**: `test_ssn_smoke_fixed.py`
* **现象观测**: 超二次收敛性的纯粹验证。
* **数学原理解析**: 剥离所有 PDE 空间导数（拉普拉斯算子），在单一数学点上直接模拟 Primal-Dual Active Set (PDAS) KKT 条件的精确 3x3 块代数系统。
* **最终结果**: 在精确的 2 步内残差清零 ($0.0000$)。这是对我们基于 UFL 符号推导的广义雅可比矩阵在代数上绝对封闭且对称的最终铁证。

```text
{logs[4]}
```

## 5. 极端正则性缺失：狄拉克冲击载荷 (Dirac Delta Load)
* **执行脚本**: `test_ssn_dirac_load.py`
* **现象观测**: 近奇异载荷下的收敛性——以及一个反直觉的加速现象。
* **数学原理解析**: 我们注入了一个极端病态的目标状态 $y_d = A \\cdot e^{{-\\sigma|x-x_c|^2}}$，将 $\\sigma$ 推至 5000，形成仅被约 3 个网格单元分辨的近狄拉克点源，局部峰值应力 $y_d(x_c) = 200$。
* **最终结果**: 近狄拉克情形仅用 **3 步**收敛——比平滑参考还要快。揭示了“狄拉克悖论”：越集中的目标反而令 SSN 收敛更快。

```text
{logs[5]}
```

## 6. 充满棱角的障碍物：空间变化控制阈值 (Jagged Obstacle Wall)
* **执行脚本**: `test_ssn_jagged_obstacle.py`
* **现象观测**: 双重叠加非光滑层下的完全收敛不变性。
* **数学原理解析**: 将标准 SSN 的固定阈值 $\\alpha$ 替换为空间变化的 $\\alpha(x)$，使其等高线形成充满不可导折痕的障碍壁。
* **最终结果**: 障碍物悖论：SSN 对障碍壁的棱角几何完全视而不见。三种几何均在 5 步内收敛。

```text
{logs[6]}
```

## 7. L 型奇異點與拓撲撕裂 (The L-Shape Singularity)
* **執行腳本**: `test_ssn_l_shape.py`
* **現象觀測**: 對空間幾何奇異性與 $H^2$ 正則性喪失的絕對免疫。
* **數學原理解析**: 通過在正方形網格中挖去四分之一，我們在 $(0.5, 0.5)$ 處人為製造了一個向內的直角。在 PDE 理論中，此處的梯度將趨於無窮大（產生 $r^{{-1/3}}$ 的物理奇異性），徹底摧毀解的 $H^2$ 完美平滑度。
* **最終結果**: **4 步精確收斂！** SSN 算法對此奇異性完全免疫。因為 PDAS 的判斷邏輯僅依賴於伴隨變量 $p$ 的絕對值大小，而不依賴其空間梯度 $\\nabla p$。无论底层空间如何撕裂，布尔指示函数都能完美切分。

```text
{logs[7]}
```

## 8. 抽真空測試：正則化極限 (The Tikhonov Vacuum)
* **執行腳本**: `test_ssn_moreau_yosida_limit.py`
* **現象觀測**: 雅可比矩陣條件數的指數級爆炸與坍塌。
* **數學原理解析**: SSN 處理 $L^1$ 控制的超二次收敛，底層的強制性來自 Tikhonov/L2 參數 $\\nu$。我們將 $\\nu$ 從 $10^{{-2}}$ 逐步抽向機器精度。
* **最終結果**: 隨著 $\\nu \\le 10^{{-8}}$，算法徹底崩塌。證明了正則化是生存的必需品。

```text
{logs[8]}
```

## 9. 絕對的 Bang-Bang 狂暴狀態 (嚴格箱式約束)
* **執行腳本**: `test_ssn_bang_bang.py`
* **現象觀測**: 控制變量在物理絕境下的極端邊界飽和與零顫振。
* **最終結果**: **5 步精確收斂。** 算法在面對物理斷層時展現了極致的冷血。高達 **99.64%** 的網格區域被死死按在絕對邊界極限上。

* **可視化說明**: 您可能會預期控制量看起來像正弦波，因為目標 $y_d$ 是正弦波。然而，由於目標振幅 (100) 遠超控制能力 ([-1, 1])，算法將控制量直接“拍”在絕對極限上以最小化 $L^2$ 誤差，將平滑正弦輸入變成了棋盘方波輸出。

<p align="center">
  <img src="../assets/bang_bang_visualization.png" alt="Bang-Bang Control" width="600">
</p>

```text
{logs[9]}
```

## 10. 物理屏障的絕境：硬性狀態約束 (State Constraints)
* **執行腳本**: `test_ssn_state_constraints.py`
* **現象觀測**: 活躍集顫振與正則性缺失導致的收斂失敗。
* **數學原理解析**: 對狀態變量施加硬性上限 $y(x) \\le 0.2$，而目標狀態 $y_d$ 峰值為 $1.0$。在狀態约束優化中，乘子不再是普通函数，而是 **Radon 测度**。
* **最終結果**: **收斂失敗 (持續振盪)。** 證明了純粹 SSN 無法自行處理狀態-測度耦合的特異性，展現了算法的絕對極限。

<p align="center">
  <img src="../assets/state_constraints_visualization.png" alt="State Constraints Struggle" width="600">
</p>

```text
{logs[10]}
```

## 11. 非線性物理的重壓：p-Laplacian 爆炸的取證調查
* **執行腳本**: `test_ssn_nonlinear_explosion.py`
* **現象觀測**: 雙重壓力收斂失效 (代數過載 + 非線性超調)。
* **數學原理解析**: 將算子替換為高度“侵略性”的 **p-Laplacian ($p=4$)**。剛度隨梯度呈立方倍率縮放，導致牛頓步長衝出信任域。

### 🕵️ 工程試錯與調試紀實 (Engineering History)
1. **手動推導的傲慢**: 最初試圖手動推導混合空間 Hessian，因 UFL 元數衝突 (`ValueError`) 而失敗。
2. **戰略轉向**: 轉向“混合自動微分”——利用 `ufl.derivative` 處理 PDE，手寫傾斜導數處理控制項。
3. **最終裁決**: **牛頓法的自殺 (Newton's Suicide)**。殘差從 $10^{{-2}}$ 飆升至 $10^{{11}}$，證明對於“暴力”算子，必須依賴全局收斂保障，這也為 **神經算子 (Neural Operators)** 提供了正當性。

<p align="center">
  <img src="../assets/nonlinear_failure_evolution.png" alt="Nonlinear Divergence Evolution" width="900">
</p>

```text
{logs[11]}
```

## 12. 维度深渊：网格无关性 (Mesh Independence)
* **执行脚本**: `test_ssn_mesh_independence.py`
* **现象观测**: 迭代次数在维度指数级爆炸下的绝对不变性。
* **数学原理解析**: 连续算子理论保证当 $h \\to 0$ 时，牛顿步数趋近于常整数。
* **最终结果**: 迭代次数从 12k 到 790k 自由度全程死死钉在 **5 步**。

```text
{logs[12]}
```

---
*语言切换: [English](README.md) | [中文](README_zh.md)*
"""

with open('README_zh.md', 'w') as f:
    f.write(content)
