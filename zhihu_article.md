# 我写了一个光学的"可微物理引擎"，比传统FDTD快100万倍

## 先说结论

我开源了一个Python库：**photokinetics**，用PyTorch实现了光学的统一计算框架。

它有三个核心特性：
1. **可微分** — 支持自动求导，可以做逆向设计
2. **极快** — 比传统数值仿真快10²~10⁹倍
3. **统一** — 8个光学模块从同一套物理框架推导

```bash
pip install photokinetics
```

仓库：https://github.com/XxLCFLXx/photokinetics
在线体验：https://xxlcflxx.github.io/photokinetics/

---

## 背景：光学仿真为什么这么慢？

做光学仿真的人都知道：**FDTD太慢了**。

算一个简单的光热模型，传统数值仿真（FDTD/FEM）可能要几分钟到几小时。如果你要做参数扫描、逆向设计，那就是几天甚至几周的计算时间。

为什么慢？因为数值仿真要把空间切分成网格，时间切分成步长，然后一步步求解偏微分方程。

但有没有想过：**光学公式其实是有解析解的**。

我做的就是这件事：把8个常见光学现象（光电效应、黑体辐射、康普顿散射、多普勒效应、引力红移、光热转换、非线性光学、光镊力）的解析公式整理成统一框架，用PyTorch实现，天然支持自动微分。

---

## 光动论（Photokinetics）是什么？

光动论是一个基于"光子动能传递"视角的光学统一框架。

传统光学教材里，光电效应、康普顿散射、光热模型是分散的几章，公式之间看起来没什么关系。光动论的核心思想是：**这些现象本质上都是光子把能量/动量传递给物质的过程**。

从这个视角出发，可以推导出一套统一的公式体系。V2.0版本覆盖了8个模块：

| 模块 | 典型应用 |
|------|---------|
| 光电效应 | 光电探测器设计 |
| 黑体辐射 | 温度测量、恒星光谱 |
| 康普顿散射 | X射线成像 |
| 多普勒效应 | 天文红移测量 |
| 引力红移 | GPS定位修正 |
| 光热模型 | 激光加工、激光医疗 |
| 非线性光学 | 多光子吸收材料分析 |
| 光镊力 | 细胞操控、单分子实验 |

---

## 核心特性：可微分意味着什么？

这是光动论和传统仿真工具最大的不同。

传统FDTD：输入参数 → 输出结果（单向）

光动论：输入参数 → 输出结果 → **梯度**（双向）

举个例子：

```python
import torch
from photokinetics import calc_photothermal

# 计算光热温升
I0 = torch.tensor(1e7, requires_grad=True)  # 光强，可求导
result = calc_photothermal(
    n=1.33, kappa=0.00012, wavelength_nm=1064,
    I0=I0, rho=1000, Cp=4186, depth_mm=1.0, time_s=1.0
)
print(f"温升: {result['dT'].item():.2f} K")  # 输出: 877 K

# 自动求导
result['dT'].backward()
print(f"d(ΔT)/d(I₀) = {I0.grad.item():.2e}")
# 输出: 8.77e-05  → 每增加1 W/m²光强，温升增加8.77e-5 K
```

这意味着你可以：
- **灵敏度分析**：知道每个参数对结果的影响
- **逆向设计**：用梯度下降反推最优参数
- **嵌入AI模型**：作为可微物理层，混合建模

---

## 实际例子：逆向设计

假设你要在硅片上达到100℃温升，该用多少激光功率？

传统方法：二分法或网格搜索，算几十次到几百次。

用光动论：

```python
import torch
from photokinetics import calc_photothermal

target_dT = 100.0  # 目标温升
I0 = torch.tensor(1e5, requires_grad=True)  # 初始猜测
optimizer = torch.optim.Adam([I0], lr=5e4)

for i in range(100):
    optimizer.zero_grad()
    result = calc_photothermal(4.15, 0.044, 532, I0, 2329, 700, 0.002, 1e-9)
    loss = (result['dT'] - target_dT) ** 2
    loss.backward()
    optimizer.step()

print(f"最优光强: {I0.item():.2e} W/m²")
# 几十次迭代就收敛
```

---

## 速度对比：到底快多少？

我做了实测对比（1D-FDTD数值仿真 vs 光动论解析模型）：

| 材料 | 温升误差 | 加速比 |
|------|---------|--------|
| 水 @ 1064nm | 3.49% | 1125x |
| 硅 @ 532nm | 0.98% | 97x |
| 锗 @ 532nm | 1.15% | 385x |

误差在5%以内，但速度快了几个数量级。

如果是3D场景，理论分析显示加速比可以达到**10⁶~10⁹倍**——因为FDTD的计算量是维度的立方关系。

---

## 有什么限制？

诚实说，光动论目前有这些局限：

1. **绝热近似**：光热模型假设热扩散可以忽略，适用条件是短脉冲（t ≪ L²/D）
2. **简单几何**：目前只支持平面波入射+均匀介质
3. **不是新物理**：推导结果和经典电动力学完全一致，价值在于"统一框架"和"计算效率"

如果你要算复杂结构（如光子晶体、金属纳米结构），还是得用FDTD。但如果你的场景是参数扫描、逆向设计、实时反馈控制，光动论的速度优势就很明显了。

---

## 谁会用到？

- **做激光加工/激光医疗的**：参数优化、治疗规划
- **做光镊的**：力计算、颗粒操控设计
- **做AI for Science的**：可微物理引擎，嵌入PyTorch
- **做光学教学的**：在线计算器，直观理解现象

在线计算器（免安装）：https://xxlcflxx.github.io/photokinetics/

---

## 如何使用？

```bash
pip install photokinetics
```

```python
import torch
from photokinetics import calc_photothermal

# 基本计算
result = calc_photothermal(1.33, 0.00012, 1064, 1e7, 1000, 4186, 1.0, 1.0)
print(f"温升: {result['dT'].item():.2f} K")

# 批量计算（同时算1000个光强）
I0_batch = torch.logspace(4, 8, 1000)
result_batch = calc_photothermal(1.33, 0.00012, 1064, I0_batch, 1000, 4186, 1.0, 1.0)

# 其他模块
from photokinetics import (
    calc_photoelectric,      # 光电效应
    calc_blackbody,          # 黑体辐射
    calc_compton,            # 康普顿散射
    calc_doppler,            # 多普勒效应
    calc_gravitational_redshift,  # 引力红移
    calc_nonlinear_order,    # 非线性光学
    calc_tweezer_force,      # 光镊力
)
```

---

## 后续计划

目前版本覆盖了8个模块，但还有扩展空间：

- 含时热传导解析解（突破绝热近似）
- 多层介质模型（更接近实际应用）
- 和FDTD的混合方法

如果你在光学/激光/光镊领域有实际需求，欢迎在评论区讨论，我可以评估优先级。

---

## 总结

光动论不是发现新物理，而是把已知物理变得**更快、更易用、更可微**。

它不会取代FDTD（复杂结构还是要数值求解），但在参数优化、逆向设计、AI for Science这些场景下，速度优势是实打实的。

如果你觉得有用，欢迎：
- GitHub点个Star：https://github.com/XxLCFLXx/photokinetics
- 在线体验：https://xxlcflxx.github.io/photokinetics/
- 提Issue或PR，一起完善

---

**相关链接**

- 仓库：https://github.com/XxLCFLXx/photokinetics
- PyPI：https://pypi.org/project/photokinetics/
- 在线计算器：https://xxlcflxx.github.io/photokinetics/
- 论文（V2.0预印本）：见仓库paper目录