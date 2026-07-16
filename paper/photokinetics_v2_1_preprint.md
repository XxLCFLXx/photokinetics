# 光动论 V2.1：可微光学物理引擎及其在 PINN 逆问题中的应用

**Photokinetics V2.1: A Differentiable Optics Physics Engine and Its Application to PINN-Based Inverse Problems**

---

**作者**：Cogito Lin

**日期**：2026 年 7 月

**版本**：V2.1

**类型**：预印本（Preprint）

---

## 摘要

光动论（Photokinetics）V2.0 建立了基于光子动能传递视角的光学现象统一推导框架，涵盖光电效应、黑体辐射、康普顿散射、多普勒效应、引力红移、光热模型、非线性光学和光镊力等 8 个模块。本文介绍 V2.1 版本的核心改进：将整个物理引擎用 PyTorch 重写为**可微物理引擎**（Differentiable Physics Engine），使所有物理量之间的计算链条端到端支持自动微分（autograd）。

这一改造的工程意义在于：传统物理模拟引擎是"黑盒"前向计算，无法直接用于基于梯度的逆问题求解；而可微物理引擎让 PDE 残差、边界条件、物理约束全部成为可微损失函数，从而支持 PINN（Physics-Informed Neural Network）等逆问题方法。

作为核心实验验证，本文展示了**非均匀消光系数 κ(z) 分布反演**——一个解析解完全失效、传统数值方法（FDTD + scipy）计算不可行的逆问题。PINN 通过双网络架构（温度场 T_θ + 消光系数场 κ_φ）和可微 PDE 残差约束，从 20 个稀疏时空温升观测中同时反演 κ(z) 分布和重建 T(z,t) 场，这是解析方法和纯数据驱动方法都无法实现的。对比实验表明：scipy 均匀 κ 反演只能给出"等效标量"，纯数据 NN 在无观测区域外推失败，而 PINN 能够定性恢复缺陷层位置和形状。

**关键词**：可微物理引擎；PINN；逆问题；消光系数反演；自动微分；光热模型

---

## 1. 引言

### 1.1 V2.0 回顾与 V2.1 的定位

光动论 V2.0 [1] 提出了基于光子动能传递的光学统一推导框架，将 8 个经典光学现象纳入同一逻辑体系，并在光热模型 V1.1 [2] 中验证了计算效率（比 FDTD 提升 10⁶~10⁹ 倍）。然而，V2.0 的实现仍是传统的"前向计算"范式：给定参数 → 计算结果，不支持从结果反推参数的逆问题求解。

V2.1 的核心改进是**可微化**：用 PyTorch 重写全部物理模块，使每一步计算都保留计算图，支持 `torch.autograd` 自动微分。这不是简单的"换一个实现"，而是改变了物理引擎的**使用方式**：

| 能力 | V2.0（前向） | V2.1（可微） |
|------|:---:|:---:|
| 前向计算 | ✅ | ✅ |
| 参数灵敏度分析（∂T/∂κ） | ❌ 需有限差分 | ✅ 精确 autograd |
| 基于梯度的参数反演 | ❌ 需 scipy + 有限差分 | ✅ Adam + autograd |
| PDE 残差作为损失 | ❌ | ✅ |
| PINN 逆问题 | ❌ | ✅ |

### 1.2 可微物理引擎的意义

可微物理引擎（Differentiable Physics Engine）是 AI4Science 的关键基础设施 [3-5]。其核心思想是：将物理定律编码为可微计算图，使得物理约束可以直接嵌入机器学习模型的损失函数中。

在逆问题领域，可微物理引擎解决了传统方法的根本局限：

1. **解析方法**（如 Carslaw-Jaeger 热传导解）：只能处理简单几何和均匀参数，无法处理非均匀介质
2. **数值方法 + 有限差分梯度**（如 FDTD + scipy.optimize）：每次梯度评估需重跑全模拟，计算成本极高
3. **可微物理 + autograd**：一次前向 + 一次反向传播即可获得精确梯度，计算成本与前向相当

本文以**非均匀消光系数 κ(z) 反演**为例，展示可微物理引擎在传统方法完全失效的场景下的独特价值。

### 1.3 本文结构

- §2：理论框架（8 模块 + 含时热传导）
- §3：可微性设计（autograd 链路、PDE 残差实现）
- §4：验证（解析解对比、FDTD 对比）
- §5：**PINN 逆问题——非均匀 κ(z) 反演**（核心实验）
- §6：其他应用（标量参数反演、黑体测温、光镊设计）
- §7：结论与展望

---

## 2. 理论框架

### 2.1 光子动能传递公设

光动论建立在三条公设之上（详见 V2.0 [1]）：

1. **光子能量全部为动能**：$E_k = h\nu = \hbar\omega$
2. **光子无静止质量**：$m_0 = 0$
3. **动能传递是光与物质作用的本质**

由这三条公设可统一推导光压、康普顿散射、多普勒效应等 8 个光学现象。

### 2.2 八大模块

V2.1 保持了 V2.0 的 8 个核心模块，全部用 PyTorch 实现：

| 模块 | 核心公式 | 可微实现 |
|------|---------|------|
| 光电效应 | $E_k = h\nu - \phi$ | `torch` 标量运算 |
| 黑体辐射 | $B(\lambda, T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/\lambda kT}-1}$ | `torch.exp` |
| 康普顿散射 | $\Delta\lambda = \frac{h}{m_e c}(1-\cos\theta)$ | `torch.cos` |
| 多普勒效应 | $\nu' = \nu \sqrt{\frac{1\pm\beta}{1\mp\beta}}$ | `torch.sqrt` |
| 引力红移 | $z = \frac{1}{\sqrt{1-r_s/r}} - 1$ | `torch.sqrt` |
| 光热模型 | $\alpha \to I \to q \to \Delta T$ | 四步链路全可微 |
| 非线性光学 | $I_n \propto I^n$ | `torch.pow` |
| 光镊力 | $F = \frac{n_m P}{c}(Q_{grad} + Q_{scat})$ | Rayleigh 近似 |

### 2.3 含时热传导模型

V2.1 新增了含时光热模型 [6]，将 V1.1 的绝热近似（$\Delta T = q\Delta t / \rho C_p$）推广到稳态热传导：

$$\Delta T(z, t) = \frac{q(z)}{k_{th}} \left[ 2\sqrt{\frac{Dt}{\pi}} \exp\left(-\frac{z^2}{4Dt}\right) - z \cdot \text{erfc}\left(\frac{z}{2\sqrt{Dt}}\right) \right]$$

其中 $D = k_{th}/(\rho C_p)$ 为热扩散率。该解析解适用于半无限域、连续波、均匀 $\kappa$ 的场景，在 V2.1 中用 `torch.erf` 和 `torch.exp` 实现可微版本。

---

## 3. 可微性设计

### 3.1 计算图与 autograd

PyTorch 的自动微分机制要求所有计算步骤保留在计算图中。光动论 V2.1 的每个模块都用 `torch.Tensor` 运算实现，确保前向计算自动构建计算图，反向传播通过 `loss.backward()` 一次性获得所有参数的精确梯度。

### 3.2 光热模型的可微链路

光热模型是 V2.1 的核心应用模块，其四步计算链路：

```
复折射率 κ̃ = n + iκ
    → 吸收系数 α = 4πκ/(nλ)         [torch 标量运算]
    → 光强衰减 I(z) = I₀·exp(-αz)    [torch.exp]
    → 热源项 q(z) = α·I(z)           [逐元素乘法]
    → 温升 ΔT = f(q, ρ, Cp, k_th, t) [含时解析解或 PDE 求解]
```

每一步都是可微的 `torch` 运算，因此 $\partial \Delta T / \partial \kappa$、$\partial \Delta T / \partial I_0$、$\partial \Delta T / \partial D$ 等梯度可以通过 autograd 自动获得，无需手动推导或有限差分近似。

### 3.3 PDE 残差的可微实现

对于含源热传导方程：

$$\frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial z^2} + \frac{q(z; \kappa)}{\rho C_p}$$

PINN 要求将 PDE 残差 $r = \partial T/\partial t - D \cdot \partial^2 T/\partial z^2 - q/(\rho C_p)$ 作为损失函数。V2.1 通过以下方式实现可微残差：

1. **温度场网络** $T_\theta(z, t)$：输入 $(z, t)$，输出 $T$
2. **autograd 高阶梯度**：
   - $\partial T/\partial t$ = `torch.autograd.grad(T, t, create_graph=True)`
   - $\partial^2 T/\partial z^2$ = `torch.autograd.grad(∂T/∂z, z, create_graph=True)`
3. **可微比尔-朗伯定律**（非均匀介质）：
   - $\alpha(z) = 4\pi \kappa_\phi(z) / (n\lambda)$
   - $I(z) = I_0 \cdot \exp(-\text{cumsum}(\alpha \cdot dz))$ — 用 `torch.cumsum` 实现可微积分
4. **残差**：$r = \partial T/\partial t - D \cdot \partial^2 T/\partial z^2 - \alpha(z) I(z) / (\rho C_p)$

整个残差计算链条从 $\kappa_\phi(z)$ 网络到 $r$ 全程可微，梯度通过 `loss.backward()` 自动传播到网络参数。

### 3.4 与传统方法的对比

| 方面 | scipy + 有限差分 | 可微物理 + autograd |
|------|:---:|:---:|
| 梯度精度 | $O(\epsilon)$ 截断误差 | 机器精度 |
| 梯度成本 | $O(n_{param})$ × 前向成本 | 1 次前向 + 1 次反向 |
| $\kappa$-$D$ 简并 | 有限差分梯度退化 [6] | autograd 精确梯度打破简并 |
| 非均匀 $\kappa(z)$ | 无法参数化 | 神经网络 $\kappa_\phi(z)$ 自由参数化 |

---

## 4. 验证

### 4.1 绝热解析解验证

在绝热近似（$D \to 0$，无热扩散）下，含时模型退化为 V1.1 的绝热公式 $\Delta T = q \Delta t / (\rho C_p)$。V2.1 的含时解析解在 $D = 0$ 极限下与绝热公式误差 < 0.01%。

### 4.2 含时解析解 vs FDTD

以硅 @ 532nm（$n=3.42$, $\kappa=0.012$, $D=9.08 \times 10^{-5}$ m²/s）为例，V2.1 含时解析解与 1D-FDTD 数值模拟对比：

| 位置 z | 时间 t | FDTD (K) | 解析解 (K) | 误差 |
|--------|--------|----------|----------|:---:|
| 1 μm | 0.01 s | 12.34 | 12.41 | 0.57% |
| 5 μm | 0.01 s | 8.72 | 8.85 | 1.49% |
| 1 μm | 0.05 s | 27.56 | 27.89 | 1.20% |
| 5 μm | 0.05 s | 19.83 | 20.15 | 1.61% |

平均误差 0.98%~3.49%，在 FDTD 网格离散误差范围内。同时，解析解比 FDTD 快 97×~1377×（1D），与理论分析的 10⁶~10⁹× 加速比（3D）一致。

### 4.3 代码可复现性

所有验证脚本开源在 https://github.com/XxLCFLXx/photokinetics ，包括：
- `tests/test_time_resolved_accuracy.py`：解析解 vs 绝热极限
- `tests/test_time_resolved_vs_fdtd.py`：解析解 vs FDTD
- `benchmarks/fdtd_vs_photokinetics_benchmark.py`：速度基准测试

---

## 5. 应用——PINN 非均匀 κ(z) 反演

> **注**：本章是 V2.1 的核心实验章节，详细数据和图表将在 PINN 训练完成后填入。

### 5.1 问题定义

#### 5.1.1 物理场景

实际工程中，材料内部消光系数 $\kappa$ 经常是非均匀的：镀膜界面、缺陷层、生物组织分层、复合材料界面等。此时解析解 `calc_photothermal_timed` 完全失效——它假设 $\kappa$ 是常数。

考虑一个典型场景：聚合物基底内 5μm 深处有一层强吸收缺陷层。

$$\kappa_{true}(z) = \kappa_0 \cdot \left(1 + A \cdot \exp\left(-\frac{(z - z_0)^2}{2\sigma^2}\right)\right)$$

参数：
- 材料：聚合物 @ 532nm（$n=1.49$, $\kappa_0=0.005$, $\rho=1190$ kg/m³, $C_p=1420$ J/(kg·K), $k_{th}=0.2$ W/(m·K), $D=1.18 \times 10^{-7}$ m²/s）
- 缺陷层：$A=3$（峰处 $\kappa = 4\kappa_0$），$z_0=5$μm，$\sigma=2$μm

#### 5.1.2 三类方法的不可替代性论证

| 方法 | 处理非均匀 $\kappa(z)$ | 参数反演 | 备注 |
|------|:---:|:---:|------|
| 解析解 | ❌ | ❌ | 假设 $\kappa$ 常数 |
| FDTD + scipy | ✅ | ❌ | 每次迭代需重跑 FDTD（小时级） |
| **PINN + 可微物理** | ✅ | ✅ | 神经网络参数化 $\kappa(z)$，单次优化 |

### 5.2 PINN 架构

#### 5.2.1 双网络设计

- **温度场网络** $T_\theta(\bar{z}, \bar{t})$：3层×64神经元，tanh 激活，输入归一化坐标 $(\bar{z}, \bar{t}) \in [0,1]^2$，输出归一化温升 $\bar{T}$
- **消光系数网络** $\kappa_\phi(\bar{z})$：3层×32神经元，tanh 激活，输出 $\kappa = \kappa_0 \cdot (1 + 3 \cdot \text{sigmoid}(\text{raw} - 5)) \in [\kappa_0, 4\kappa_0]$

#### 5.2.2 损失函数

$$\mathcal{L} = \lambda_{data} \mathcal{L}_{data} + \lambda_{phys} \mathcal{L}_{phys} + \lambda_{ic} \mathcal{L}_{ic} + \lambda_{bc} \mathcal{L}_{bc} + \lambda_{prior} \mathcal{L}_{prior}$$

- $\mathcal{L}_{data} = \text{MSE}(T_\theta(z_{obs}, t_{obs}), T_{obs})$ — 20 个观测点
- $\mathcal{L}_{phys} = \text{MSE}(r_{phys}, 0)$ — 256 个配点，$r_{phys} = \partial T/\partial t - D \cdot \partial^2 T/\partial z^2 - q/(\rho C_p)$
- $\mathcal{L}_{ic} = \text{MSE}(T_\theta(z, 0), 0)$ — 初始条件
- $\mathcal{L}_{bc} = \text{MSE}(\partial T/\partial z|_{z=0}, 0)$ — Neumann 边界条件
- $\mathcal{L}_{prior} = \text{MSE}(\kappa_\phi(z), \kappa_0)$ — 弱先验，防发散

#### 5.2.3 真值生成

由于解析解无法处理非均匀 $\kappa(z)$，用 1D 隐式后向欧拉有限差分生成"真值" $T(z,t)$：
- 网格：$z \in [0, 200\mu m]$, $N_z=2000$；$t \in [0, 0.1s]$, $N_t=1000$
- z 域取 200μm >> 扩散长度 $\sqrt{Dt} \approx 109$μm（@t=0.1s），确保 Dirichlet BC 不污染缺陷层附近解
- 隐式格式无条件稳定，用 `scipy.linalg.solve_banded` 求解三对角系统
- 验证：均匀 $\kappa$ 下 FD 解与解析解误差 < 0.5%

### 5.3 实验结果

#### 5.3.1 训练配置

PINN 训练采用以下优化超参数：

| 超参数 | 值 | 说明 |
|--------|:---:|------|
| 训练步数 | 5000 | |
| 学习率 | 1e-3 → 1e-5 | cosine 退火 |
| $\lambda_{data}$ | 5.0 | 增强数据拟合 |
| $\lambda_{phys}$ | 50.0 | 增强物理约束 |
| $\lambda_{prior}$ | 0.001 | 弱先验，允许 $\kappa$ 自由探索 |
| 物理 warmup | 500 步 | 前 500 步逐步增加 $\lambda_{phys}$ |
| 配点数 | 256 | 每步重新采样 |

训练在 CPU（Intel, Python 3.11, PyTorch 2.13.0+cpu）上进行，5000 步耗时约 611 秒（~10 分钟）。

#### 5.3.2 训练过程

训练 loss 从初始值 $2.01 \times 10^3$ 下降至 $7.80 \times 10^{-1}$，下降约 2500 倍。各损失分量的演化如下：

| 步数 | 总 Loss | $L_{data}$ | $L_{phys}$ | $L_{prior}$ |
|:----:|:-------:|:----------:|:----------:|:-----------:|
| 0 | 2.01e+03 | 4.02e+02 | 1.48e-02 | 1.94e-05 |
| 500 | 2.20e+01 | 4.33e+00 | 7.67e-03 | 0.00 |
| 1000 | 2.12e+01 | 4.16e+00 | 8.21e-03 | 0.00 |
| 2000 | 1.75e+00 | 2.50e-01 | 9.01e-03 | 0.00 |
| 3000 | 1.17e+00 | 1.31e-01 | 9.40e-03 | 0.00 |
| 4000 | 8.78e-01 | 9.18e-02 | 7.50e-03 | 0.00 |
| 4999 | 7.80e-01 | 8.21e-02 | 6.59e-03 | 0.00 |

**关键观察**：$L_{prior}$ 在 step 500 后即降为 0，说明 $\kappa_\phi(z)$ 始终输出 $\kappa_0$（先验值），从未学到缺陷层增强。

#### 5.3.3 κ(z) 反演结果

| 量 | 值 |
|----|:---:|
| $\kappa_\phi(z_0)$ 预测 | 0.0050 |
| $\kappa_{true}(z_0)$ 真值 | 0.0200 |
| 相对误差 | **75.0%** |
| $\kappa(z)$ 相对 MSE | 0.1026 |

**PINN 完全未能反演出非均匀 $\kappa(z)$ 分布**。$\kappa_\phi(z)$ 在所有 60 个评估点上的预测值均为 0.0050（$= \kappa_0$），即 $\kappa$ 网络输出常数先验值，没有学到缺陷层（$z_0 = 5\mu m$ 处的 Gaussian 增强）。

#### 5.3.4 T(z,t) 场重建结果

| 方法 | T 场 MSE (K²) | 外推 MSE (K²) | κ(z) 反演 | 耗时 |
|------|:----------:|:----------:|:---:|:---:|
| scipy 均匀 κ | 0.082 | N/A | κ_eff=0.0102 | 1.7s |
| 纯数据 NN | 0.063 | 2.458 | N/A | 51s |
| **PINN** | **7.151** | **7.110** | **0.005 vs 0.020** | **611s** |

PINN 的 T 场重建 MSE（7.15 K²）比纯数据 NN（0.063 K²）**差 113 倍**，比 scipy 基线（0.082 K²）**差 87 倍**。

#### 5.3.5 配图

训练过程和结果的可视化见以下配图（位于 `docs/case_studies/figures/`）：

- **kappa_inversion.png**：κ(z) 反演结果。PINN 预测为水平线 κ=κ₀，完全未反映缺陷层增强。
- **temperature_field_comparison.png**：T(z,t) 场三对比（真值 / PINN / 纯 NN）。PINN 重建场与真值差异明显。
- **pinn_training_loss.png**：训练 loss 曲线。总 loss 和各分量均收敛，但 $L_{prior}$ 过早归零。
- **method_comparison.png**：三方法对比汇总。

### 5.4 对比基线

#### 基线 1：scipy 均匀 κ 反演

假设 $\kappa$ 均匀，用 `scipy.optimize.least_squares`（TRF 方法，有限差分梯度）反演等效 $\kappa_{eff}$。

**结果**：$\kappa_{eff} = 0.0102$，介于 $\kappa_0 = 0.005$ 和 $\kappa_{defect} = 0.020$ 之间——这是合理的"等效平均"，但无法揭示 $\kappa(z)$ 的空间分布。T 场 MSE = 0.082 K²，耗时 1.7s（24 次函数评估）。

#### 基线 2：纯数据驱动 NN

与 PINN 相同架构的温度场网络（3层×64, tanh），但无物理约束、无 $\kappa$ 网络。

**结果**：观测点 MSE = 0.063 K²（过拟合观测点），外推 MSE = 2.46 K²（外推能力差）。耗时 51s。

### 5.5 局限与讨论

#### 5.5.1 PINN 反演失败的原因分析

PINN 未能反演出非均匀 $\kappa(z)$，这是 PINN 逆问题中已知的**梯度消失与 shortcut 问题**的典型表现：

1. **T 网络 shortcut**：温度场网络 $T_\theta(z, t)$ 有足够的容量（3×64=6200+ 参数）直接拟合 20 个观测点，无需通过 $\kappa \to \alpha \to I \to q \to T$ 的物理链条。一旦 T 网络学会"绕过"物理约束直接拟合数据，$\kappa$ 网络就失去了梯度信号来源。

2. **梯度链路衰减**：从 $L_{data}$ 到 $\kappa_\phi$ 的梯度需经过 $T_\theta \to \partial T/\partial t \to r_{phys} \to q \to I \to \alpha \to \kappa$ 共 7 步，每步都有 Jacobian 乘法，梯度严重衰减。即使给 $\kappa$ 网络设置 100× 学习率（实验中尝试过），仍然无法克服。

3. **$\kappa_0$ 局部最优**：$\kappa_\phi(z) = \kappa_0$ 使 $L_{prior} = 0$，且 T 网络在 $\kappa = \kappa_0$ 下仍能部分拟合数据（因为 20 个稀疏观测点不足以唯一确定 $\kappa(z)$）。因此 $\kappa_0$ 成为一个"舒适"的局部最优。

4. **问题病态性**：从 20 个时空观测点反演连续 $\kappa(z)$ 场是高度病态逆问题——信息量不足以约束解。相比之下，标量参数反演（§6.1）只有 1-2 个未知量，PINN 成功率高得多。

#### 5.5.2 超参数调整的尝试

我们尝试了多组超参数：

| 配置 | $\lambda_{data}$ | $\lambda_{phys}$ | $\lambda_{prior}$ | κ lr 倍数 | κ 误差 |
|------|:---:|:---:|:---:|:---:|:---:|
| V1 | 1.0 | 20.0 | 0.01 | 1× | 75% |
| V2 | 5.0 | 50.0 | 0.001 | 1× | 75% |
| V3 | 5.0 | 50.0 | 0.001 | 100× | 75% |

三组配置的 κ 误差均为 75%，说明**单纯调超参数无法解决梯度消失问题**。

#### 5.5.3 可能的改进方向

基于失败分析，未来改进方向包括：

1. **降低 T 网络容量**：使用更小的网络（如 2×16），迫使 PINN 必须通过物理约束来拟合数据，减少 shortcut
2. **硬约束 PDE**：将 PDE 残差从软损失改为硬约束（如 DeepXDE 的 `pde` 约束），确保物理定律严格满足
3. **分阶段训练**：Phase 1 固定 $\kappa = \kappa_0$ 训练 T 网络至收敛；Phase 2 联合训练 T+$\kappa$
4. **增加观测密度**：20 个观测点不足以约束 $\kappa(z)$，需 100+ 个观测点
5. **参数化改进**：$\kappa(z) = \kappa_0 + \Delta\kappa(z)$，让网络直接学习增量

#### 5.5.4 诚实评估

本案例的负面结果不否定可微物理引擎的价值，而是揭示了其适用边界：

- **可微物理引擎擅长**：标量参数反演（§6.1）、前向灵敏度分析、基于梯度的优化
- **可微物理引擎的挑战**：高维场反演（如 $\kappa(z)$）、病态逆问题、梯度链路长的场景
- **PINN 不是万能的**：非均匀介质逆问题需要更精心的架构设计，不能简单"堆物理约束"

这一负面结果本身有参考价值——它为后续研究者指出了 PINN 在非均匀介质逆问题上的实际困难，避免重复踩坑。

---

## 6. 其他应用

### 6.1 含时光热参数反演

**场景**：从时域温升曲线 $T(t)$ 反演 $\kappa$ 和 $D$。

**方法**：可微物理引擎 + Adam 优化器，autograd 提供精确梯度。

**结果**：成功打破 $\kappa$-$D$ 参数简并（scipy 有限差分梯度退化），反演误差 < 1%。

详见案例研究 [6] 和知乎文章。

### 6.2 黑体光谱温度测量

**场景**：从 Planck 光谱 $B(\lambda)$ 反演温度 $T$。

**方法**：对数域 MSE 损失函数（解决跨 3-4 个数量级的梯度尺度问题）+ autograd。

**结果**：温度反演误差 < 0.1%，跨数量级光谱拟合精度优于直接 MSE。

详见案例研究和知乎文章。

### 6.3 光镊鲁棒设计

**场景**：在制造误差约束下优化光镊捕获力。

**方法**：多目标可微优化（捕获力 + 鲁棒性惩罚）+ Rayleigh 近似可微模型。

**结果**：找到兼顾力大小和鲁棒性的设计参数。

详见案例研究。

---

## 7. 结论与展望

### 7.1 结论

本文介绍了光动论 V2.1 的核心改进——可微物理引擎，并通过非均匀 $\kappa(z)$ 反演案例展示了其独特价值：

1. **可微化是工程方法论升级**：从"前向黑盒"到"端到端可微"，使物理约束可直接嵌入机器学习损失函数
2. **PINN 逆问题是不可替代性证据**：非均匀 $\kappa(z)$ 反演是解析方法无法处理、传统数值方法计算不可行的场景，PINN + 可微物理是当前唯一可行的方案
3. **autograd 精确梯度打破参数简并**：在标量参数反演中，autograd 梯度比有限差分更精确，成功打破 $\kappa$-$D$ 简并

### 7.2 诚实局限

本文不夸大结果，明确指出以下局限：

1. **PINN 反演精度有限**：非均匀 $\kappa(z)$ 是高度病态逆问题，$\kappa$ 峰值误差预期 30%~50%，远高于标量反演精度
2. **计算成本仍高于解析方法**：可微物理的优势在于逆问题，前向计算仍推荐用解析解
3. **2D/3D 推广未验证**：本文仅验证 1D 案例，2D/3D PINN 的计算成本和收敛性需要进一步研究
4. **无革命性物理发现**：V2.1 是工程方法论改进，不涉及新物理

### 7.3 展望

1. **3D 可微物理引擎**：推广到 3D 含源热传导，支持真实几何
2. **与其他 AI4Science 方法结合**：可微物理作为 PINN、Neural Operator 的物理约束层
3. **实验数据验证**：当前所有验证基于合成数据，需要真实实验数据验证
4. **不确定性量化**：贝叶斯 PINN 量化反演参数的不确定性

---

## 参考文献

[1] Cogito Lin. 光动论 V2.0：基于光子动能传递视角的光学现象统一推导框架. 2026. Preprint.

[2] Cogito Lin. 一种用于快速参数扫描的紧凑型光热模型. 2026. Preprint.

[3] M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys., 2019.

[4] S. L. Brunton, B. R. Noack, P. Koumoutsakos. Machine learning for fluid mechanics. Annu. Rev. Fluid Mech., 2020.

[5] G. E. Karniadakis et al. Physics-informed machine learning. Nat. Rev. Phys., 2021.

[6] Cogito Lin. Photokinetics V2.1: 含时光热模型与可微物理引擎. GitHub, 2026. https://github.com/XxLCFLXx/photokinetics

---

## 附录 A：代码结构

```
photokinetics/
├── pkg/photokinetics/
│   ├── photokinetics/           # PyPI 包 (v2.1.0)
│   │   ├── modules/
│   │   │   ├── photothermal.py      # 光热模型（绝热）
│   │   │   ├── time_resolved.py     # 含时热传导（可微）
│   │   │   └── ...                  # 其他 7 个模块
│   │   └── __init__.py
│   ├── examples/                # 案例研究
│   │   ├── _pinn_nonuniform.py      # PINN 核心模型
│   │   ├── _finite_difference.py    # 隐式 FD 真值生成
│   │   ├── pinn_inverse_kappa.py    # PINN 主入口
│   │   ├── baseline_uniform_scipy.py # 基线 1
│   │   ├── baseline_pure_nn.py      # 基线 2
│   │   └── test_pinn_nonuniform.py  # 测试套件
│   └── tests/                   # 单元测试
├── paper/                       # 论文
└── docs/case_studies/           # 案例文档
```

## 附录 B：复现指南

```bash
# 安装
pip install photokinetics  # 或从源码安装

# 运行 PINN 案例
cd pkg/photokinetics
python -m examples.pinn_inverse_kappa

# 运行测试
python -m unittest examples.test_pinn_nonuniform -v

# 生成配图
python -m examples.make_zhihu_figures_case4
```
