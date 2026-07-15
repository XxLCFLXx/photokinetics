# Photokinetics V2.0

**可微光学物理引擎 (Differentiable Optics Engine in PyTorch)**

基于光子动能传递的光学统一计算框架，使用 PyTorch 实现，天然支持自动微分、批量计算和逆向设计。

## 核心特性

- **可微分** — 所有公式返回 `torch.Tensor`，支持 `requires_grad=True` 自动求导
- **批量化** — 参数可以是张量，自动向量化（如批量光强扫描）
- **纯解析** — 没有数值积分，没有训练数据，直接公式计算
- **极速** — 比 1D-FDTD 快 10²~10³x，比 3D-FDTD 理论快 10⁶~10⁹x
- **统一框架** — 8 个光学模块从同一套光子动能传递公设推导

## 安装

```bash
pip install photokinetics
```

## 快速开始

### 基本计算

```python
from photokinetics import calc_photothermal

# 计算水在 1064nm 激光下的温升
result = calc_photothermal(
    n=1.33, kappa=0.00012, wavelength_nm=1064,
    I0=1e7, rho=1000, Cp=4186, depth_mm=1.0, time_s=1.0
)
print(f"温升: {result['dT'].item():.2f} K")
# 输出: 温升: 877.02 K
```

### 自动微分

```python
import torch
from photokinetics import calc_photothermal

I0 = torch.tensor(1e7, requires_grad=True)
result = calc_photothermal(1.33, 0.00012, 1064, I0, 1000, 4186, 1.0, 1.0)
result['dT'].backward()

print(f"d(ΔT)/d(I₀) = {I0.grad.item():.2e}")
# 输出: 8.77e-05
```

### 逆向设计（梯度下降求光强）

```python
import torch
from photokinetics import calc_photothermal

target_dT = 100.0
I0 = torch.tensor(1e5, requires_grad=True)
optimizer = torch.optim.Adam([I0], lr=5e4)

for i in range(100):
    optimizer.zero_grad()
    result = calc_photothermal(1.33, 0.00012, 1064, I0, 1000, 4186, 1.0, 1.0)
    loss = (result['dT'] - target_dT) ** 2
    loss.backward()
    optimizer.step()

print(f"I₀ = {I0.item():.2e}, ΔT = {result['dT'].item():.2f} K")
# 输出: I₀ = 1.15e+06, ΔT = 100.52 K
```

### 批量计算

```python
import torch
from photokinetics import calc_photothermal

I0_batch = torch.logspace(4, 8, 1000)  # 1000 个光强
result = calc_photothermal(1.33, 0.00012, 1064, I0_batch, 1000, 4186, 1.0, 1.0)
print(result['dT'].shape)  # torch.Size([1000])
```

## 实战案例

三个可微物理逆问题实战案例，展示如何用 photokinetics 解决真实工程问题：

| 案例 | 场景 | 反演目标 | 脚本 |
|------|------|---------|------|
| 1 | 含时光热参数反演 | 消光系数 κ + 热扩散率 D | `examples/fit_transient_photothermal.py` |
| 2 | 黑体光谱测温 | 温度 T + 仪器增益 | `examples/fit_blackbody_temperature.py` |
| 3 | 光镊鲁棒设计 | 光强梯度（多目标优化） | `examples/design_optical_tweezer.py` |

运行案例：

```bash
python -m examples.fit_transient_photothermal
python -m examples.fit_blackbody_temperature
python -m examples.design_optical_tweezer
```

每个案例都包含：
- 合成观测数据生成（固定种子，可复现）
- Adam 优化循环（autograd 梯度反传）
- 结果验收（参数恢复误差 + loss 下降）

## 8 个模块速查表

| 模块 | 函数 | 核心参数 |
|------|------|---------|
| 光电效应 | `calc_photoelectric(phi_ev, lambda_nm)` | 逸出功, 波长 |
| 黑体辐射 | `calc_blackbody(T, lambda_nm=None)` | 温度, 波长 |
| 康普顿散射 | `calc_compton(E0_keV, theta_deg)` | 入射能量, 散射角 |
| 多普勒效应 | `calc_doppler(nu0, v_km_s, receding)` | 频率, 速度 |
| 引力红移 | `calc_gravitational_redshift(M, r1, r2)` | 质量, 半径 |
| **光热模型** | `calc_photothermal(n, kappa, λ, I0, ρ, Cp, z, t)` | 8 个参数 |
| 非线性光学 | `calc_nonlinear_order(Eg_eV, lambda_nm)` | 禁带, 波长 |
| 光镊力 | `calc_tweezer_force(a, n_p, n_m, λ, grad_I)` | 半径, 折射率 |

## 物理常数

所有常数使用 CODATA 2018 推荐值，定义为 `torch.tensor`。

## 性能对比

与 1D-FDTD 数值仿真对比（绝热近似适用域内）：

| 材料 | ΔT 误差 | 加速比 |
|------|---------|--------|
| 水 @ 1064nm | 3.49% | 1125x |
| 硅 @ 532nm | 0.98% | 97x |
| 锗 @ 532nm | 1.15% | 385x |

详见 [benchmarks/](https://github.com/XxLCFLXx/photokinetics/tree/main/benchmarks)。

## 应用场景

- **AI for Science** — 嵌入 PyTorch 做物理驱动机器学习
- **逆向设计** — 已知目标温升，梯度下降反推光强/波长
- **灵敏度分析** — 计算各参数对输出的梯度
- **参数扫描** — 批量计算数千组参数
- **实时仿真** — 解析公式微秒级计算
- **教学/科普** — 在线计算器 [在线体验](https://xxlcflxx.github.io/photokinetics/calculator/photokinetics_calculator.html)

## 限制

- 光热模型采用绝热近似，适用条件 `t ≪ L²/D`
- 目前支持平面波入射 + 均匀介质
- 光镊力使用瑞利近似（小颗粒，a ≪ λ）

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{photokinetics2026,
  title={Photokinetics V2.0: A Differentiable Optics Engine},
  author={Cogito Lin},
  year={2026},
  url={https://github.com/XxLCFLXx/photokinetics}
}
```

## License

MIT
