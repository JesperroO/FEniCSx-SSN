# FEniCSx-SSN

这是一个面向稀疏 PDE 约束最优控制的半光滑 Newton / Primal-Dual Active Set 研究代码发布版本。

核心实现基于 FEniCSx / dolfinx，支持 2D / 3D、直接求解或 AMG、AMR、自定义结果导出，以及一个实验性的 SSN 引导 XPINN 阶段。

## 这个发布分支包含什么

这里只保留可复用的求解器、接口、测试和最小运行说明。内部研究路线、文献全文归档、尚未公开的 novelty / prior-art 分析、生成数据集、实验日志以及 agent 工作文件都被有意排除。

## 目录

- `ssn/`：SSN / PDAS、AMR、AMG、导出与 active-set 提取
- `xpinn/`：实验性的 SSN-guided XPINN 组件
- `interface/`：SSN 结果数据契约
- `tests/`：数值与回归测试
- `figures/showcase/`：精选结果图
- `run_ssn.py`：主要 SSN 入口
- `run_xpinn.py`：XPINN 训练与评估入口
- `run_pipeline.py`：SSN → XPINN 流程
- `Dockerfile`：FEniCSx 0.7.3 + PyTorch CPU 环境

## 快速运行

```bash
docker build -t fenicsx-ssn .

docker run --rm -v "$PWD":/workspace -w /workspace fenicsx-ssn \
  python3 run_ssn.py --mesh 64 --output ssn_result.npz
```

3D + AMG：

```bash
docker run --rm -v "$PWD":/workspace -w /workspace fenicsx-ssn \
  python3 run_ssn.py --dim 3 --mesh 24 --solver amg --output ssn_result.npz
```

完整 SSN → XPINN：

```bash
docker run --rm -v "$PWD":/workspace -w /workspace fenicsx-ssn \
  python3 run_pipeline.py
```

## 方法边界

当前问题设置包含 L1 稀疏项以及小的 L2 / Tikhonov 正则项。active set 从伴随状态计算，控制变量的非光滑关系在 PDAS / SSN 迭代中显式处理，而不是交给通用自动微分系统。

XPINN 部分属于研究性组件，不应被理解为通用 neural PDE solver。

## 发布说明

这个分支是从私有研究仓库整理出的 release candidate，不保留完整研究历史。详细说明见 `PUBLIC_RELEASE_NOTES.md`。

## License

当前 release-candidate 分支尚未最终确定许可证。除适用法律与 GitHub 条款明确授予的权利外，不应推定存在额外的复制、修改或再分发许可。
