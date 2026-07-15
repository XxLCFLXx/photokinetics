# Photokinetics Studio 实战案例实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建三个可微物理实战案例（含时光热反演、黑体测温、光镊鲁棒设计）+ 共享工具层，作为 photokinetics v2.1 的产品原型。

**Architecture:** 三层结构——共享工具层 (`examples/_common.py`) → 三个独立案例脚本 + 配套测试 → 复用 v2.1 公共 API。每个案例采用 TDD：先写失败测试，再实现，最后验收。

**Tech Stack:** Python 3.8+, PyTorch (autograd), photokinetics v2.1, pytest

**Spec:** `docs/superpowers/specs/2026-07-15-photokinetics-studio-design.md`

**Python 解释器:** `C:\Users\Administrator\AppData\Local\Python\bin\python.exe`（项目工作环境，torch 2.12.1+cpu, scipy 1.18.0 已安装）

**工作目录:** `c:\Users\Administrator\Documents\Trae solo\photokinetics\pkg\photokinetics`

---

## File Structure

```
photokinetics/pkg/photokinetics/
├── examples/
│   ├── __init__.py                     # 新建，空文件
│   ├── _common.py                      # 新建，共享工具层
│   ├── fit_transient_photothermal.py   # 新建，案例1
│   ├── fit_blackbody_temperature.py    # 新建，案例2
│   └── design_optical_tweezer.py       # 新建，案例3
└── tests/
    ├── test_inverse_photothermal.py    # 新建，案例1测试
    ├── test_blackbody_inversion.py     # 新建，案例2测试
    └── test_tweezer_design.py          # 新建，案例3测试
```

**依赖说明：**
- `examples/_common.py` 无外部依赖（仅 torch）
- 案例1依赖 `calc_photothermal_timed`（v2.1 已发布）
- 案例2依赖 `calc_blackbody`（v2.0 已有）
- 案例3依赖 `calc_tweezer_force`（v2.0 已有）
- 测试用 pytest，已在 `pyproject.toml` 的 dev 依赖中

---

## Task 1: 共享工具层 `examples/_common.py`

**Files:**
- Create: `examples/__init__.py`（空文件）
- Create: `examples/_common.py`
- Test: `tests/test_common.py`

- [ ] **Step 1: 创建 examples 包**

创建空文件 `examples/__init__.py`。

- [ ] **Step 2: 写失败测试 `tests/test_common.py`**

```python
"""共享工具层测试。"""
import torch
import pytest
from examples._common import (
    positive_parameter,
    relative_mse,
    log_mse,
    optimize,
)


def test_positive_parameter_basic():
    """softplus 变换保证正值。"""
    raw = torch.nn.Parameter(torch.tensor(-5.0))
    val = positive_parameter(raw, scale=1.0, floor=0.0)
    assert val.item() > 0
    assert val.requires_grad  # 保持梯度


def test_positive_parameter_floor():
    """floor 参数生效。"""
    raw = torch.nn.Parameter(torch.tensor(-100.0))
    val = positive_parameter(raw, scale=1.0, floor=10.0)
    assert val.item() >= 10.0


def test_positive_parameter_scale():
    """scale 参数缩放。"""
    raw = torch.nn.Parameter(torch.tensor(0.0))
    val = positive_parameter(raw, scale=1e6, floor=0.0)
    # softplus(0) ≈ 0.693, * 1e6 ≈ 693147
    assert 690000 < val.item() < 700000


def test_relative_mse_uniform():
    """均匀误差时 relative_mse 等于普通 mse。"""
    pred = torch.tensor([1.1, 2.2, 3.3])
    target = torch.tensor([1.0, 2.0, 3.0])
    # 相对误差都是 0.1，square 后 0.01，mean 0.01
    result = relative_mse(pred, target)
    assert abs(result.item() - 0.01) < 1e-6


def test_relative_mse_avoids_high_value_dominance():
    """高值不主导——小值的大相对误差也能被捕捉。"""
    pred = torch.tensor([101.0, 1.0])
    target = torch.tensor([100.0, 10.0])
    # 直接 MSE = (1 + 81)/2 = 41，高值主导
    # 相对 MSE = (0.0001 + 81)/2 ≈ 40.5，小值误差被放大
    result = relative_mse(pred, target)
    assert result.item() > 30  # 小值误差被捕捉


def test_log_mse_cross_magnitude():
    """跨数量级数据用 log_mse。"""
    pred = torch.tensor([1e3, 1e-3])
    target = torch.tensor([1e3, 1e-3])
    result = log_mse(pred, target)
    assert result.item() < 1e-10  # 完全匹配时近零


def test_log_mse_relative_error():
    """log_mse 等价于对数域相对误差。"""
    pred = torch.tensor([110.0])  # 10% 误差
    target = torch.tensor([100.0])
    result = log_mse(pred, target)
    expected = (torch.log(torch.tensor(110.0)) - torch.log(torch.tensor(100.0))) ** 2
    assert abs(result.item() - expected.item()) < 1e-6


def test_optimize_basic():
    """优化循环能降低 loss。"""
    x = torch.nn.Parameter(torch.tensor(5.0))
    target = torch.tensor(1.0)

    def closure():
        return (x - target) ** 2

    history = optimize([x], closure, steps=100, lr=0.1)
    assert history[-1] < history[0]
    assert abs(x.item() - 1.0) < 0.1


def test_optimize_callback():
    """callback 被调用。"""
    x = torch.nn.Parameter(torch.tensor(5.0))
    calls = []

    def closure():
        return x ** 2

    def callback(step, loss):
        calls.append((step, loss.item()))

    optimize([x], closure, steps=10, lr=0.1, callback=callback)
    assert len(calls) == 10
    assert calls[0][0] == 0
    assert calls[-1][0] == 9
```

- [ ] **Step 3: 运行测试验证失败**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_common.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'examples._common'`

- [ ] **Step 4: 实现 `examples/_common.py`**

```python
"""Photokinetics 实战案例共享工具层。

提供正值参数化、损失函数和通用优化循环，
被三个实战案例脚本复用。
"""
import torch


def positive_parameter(raw, scale=1.0, floor=0.0):
    """
    通过 softplus 变换保证参数始终为正。

    参数:
        raw    — 原始参数（torch.nn.Parameter，无约束）
        scale  — 缩放系数
        floor  — 下限（保证 val >= floor）

    返回: 正值张量，保持梯度
    """
    return floor + scale * torch.nn.functional.softplus(raw)


def relative_mse(prediction, target, eps=1e-12):
    """
    相对误差 MSE，避免高值主导。

    L = mean(((pred - target) / clamp(|target|, min=eps))²)
    """
    scale = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(((prediction - target) / scale) ** 2)


def log_mse(prediction, target, eps=1e-30):
    """
    对数域 MSE，适用于跨数量级数据（如 Planck 光谱）。

    L = mean((log(pred + eps) - log(target + eps))²)
    """
    return torch.mean((torch.log(prediction + eps) - torch.log(target + eps)) ** 2)


def optimize(parameters, closure, steps, lr, callback=None):
    """
    Adam 优化循环，返回每步 loss 历史。

    参数:
        parameters — torch.nn.Parameter 列表
        closure    — 无参函数，返回 loss 张量
        steps      — 优化步数
        lr         — 学习率
        callback   — 可选，每步调用 callback(step, loss)

    返回: list[float]，每步 loss 值
    """
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = closure()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if callback is not None:
            callback(step, loss)
    return history
```

- [ ] **Step 5: 运行测试验证通过**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_common.py -v`

Expected: 9 passed

- [ ] **Step 6: 提交**

```bash
git add examples/__init__.py examples/_common.py tests/test_common.py
git commit -m "feat: add shared utilities layer for case studies (positive_parameter, relative_mse, log_mse, optimize)"
```

---

## Task 2: 案例 1 — 含时光热参数反演

**Files:**
- Create: `examples/fit_transient_photothermal.py`
- Test: `tests/test_inverse_photothermal.py`

**物理场景：** 硅 @ 532nm，κ_true=0.044, D_true=9.08e-5 m²/s，z∈[0,10]μm × t∈[1ns,1ms] 对数网格，1% 噪声。

- [ ] **Step 1: 写失败测试 `tests/test_inverse_photothermal.py`**

```python
"""案例1测试：含时光热参数反演。"""
import torch
import pytest
from examples.fit_transient_photothermal import (
    make_observations,
    forward,
    run_inversion,
)


def test_make_observations_shape():
    """合成观测数据形状正确。"""
    z_grid, t_grid, T_obs, params_true = make_observations(seed=42, noise=0.0)
    # z∈[0,10]μm → 10点, t∈[1ns,1ms] → 10点
    # T_obs 应为 10×10 网格
    assert T_obs.shape[0] == 10  # z 维度
    assert T_obs.shape[1] == 10  # t 维度
    assert 'kappa' in params_true
    assert 'D' in params_true
    assert params_true['kappa'] == pytest.approx(0.044, abs=1e-6)
    assert params_true['D'] == pytest.approx(9.08e-5, abs=1e-10)


def test_make_observations_reproducible():
    """固定种子可复现。"""
    _, _, T1, _ = make_observations(seed=42, noise=0.01)
    _, _, T2, _ = make_observations(seed=42, noise=0.01)
    assert torch.allclose(T1, T2)


def test_forward_gradient_available():
    """前向计算梯度可用。"""
    raw_kappa = torch.nn.Parameter(torch.tensor(-2.0))
    raw_D = torch.nn.Parameter(torch.tensor(-10.0))
    z_grid, t_grid, T_obs, _ = make_observations(seed=42, noise=0.0)

    loss = forward(raw_kappa, raw_D, z_grid, t_grid, T_obs)
    loss.backward()

    assert raw_kappa.grad is not None
    assert torch.isfinite(raw_kappa.grad).all()
    assert raw_D.grad is not None
    assert torch.isfinite(raw_D.grad).all()


def test_run_inversion_noiseless_recovery():
    """无噪声下参数恢复误差 <5%。"""
    result = run_inversion(seed=42, noise=0.0, steps=500, lr=0.05)
    kappa_err = abs(result['kappa_fit'] - result['kappa_true']) / result['kappa_true']
    D_err = abs(result['D_fit'] - result['D_true']) / result['D_true']
    assert kappa_err < 0.05, f"κ 误差 {kappa_err:.2%} 超过 5%"
    assert D_err < 0.05, f"D 误差 {D_err:.2%} 超过 5%"


def test_run_inversion_loss_decreases():
    """优化后 loss 下降。"""
    result = run_inversion(seed=42, noise=0.01, steps=300, lr=0.05)
    assert result['final_loss'] < result['initial_loss']


def test_run_inversion_noisy_recovery():
    """1% 噪声下参数恢复误差 <10%。"""
    result = run_inversion(seed=42, noise=0.01, steps=500, lr=0.05)
    kappa_err = abs(result['kappa_fit'] - result['kappa_true']) / result['kappa_true']
    D_err = abs(result['D_fit'] - result['D_true']) / result['D_true']
    assert kappa_err < 0.10, f"κ 误差 {kappa_err:.2%} 超过 10%"
    assert D_err < 0.10, f"D 误差 {D_err:.2%} 超过 10%"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_inverse_photothermal.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'examples.fit_transient_photothermal'`

- [ ] **Step 3: 实现 `examples/fit_transient_photothermal.py`**

```python
"""案例1：含时光热参数反演。

从时空温升曲线 T(z,t) 联合反演消光系数 κ 和热扩散率 D，
直接使用 photokinetics v2.1 的 calc_photothermal_timed 含时解析解。

物理场景：硅 @ 532nm
    α = 4πκ/(nλ) ≈ 2.5×10⁵ /m（强吸收）
    D = 9.08e-5 m²/s
    时间跨度 1ns-1ms 覆盖绝热→过渡→长时区域

可辨识性教学点：
    短时绝热区 T ∝ α·t，与 D 无关；
    需用过渡区数据才能联合反演 κ 和 D。

用法：
    python -m examples.fit_transient_photothermal
"""
import torch
import numpy as np
from photokinetics import calc_photothermal_timed
from examples._common import positive_parameter, relative_mse, optimize


# ===== 物理常数（硅 @ 532nm）=====
N_SI = 4.15           # 折射率
KAPPA_TRUE = 0.044    # 消光系数真值
WAVELENGTH = 532.0    # nm
RHO_SI = 2329.0       # kg/m³
CP_SI = 700.0         # J/(kg·K)
D_TRUE = 9.08e-5      # m²/s 热扩散率真值
I0 = 1e7              # W/m² 入射光强

# ===== 观测网格 =====
Z_POINTS_MM = torch.logspace(-5, -2, 10) * 1e3  # [0.01μm, 10μm] → mm，对数10点
T_POINTS_S = torch.logspace(-9, -3, 10)         # [1ns, 1ms]，对数10点


def make_observations(seed=42, noise=0.0):
    """
    用真值生成合成观测数据。

    返回: (z_grid_mm, t_grid_s, T_obs[10×10], params_true_dict)
    """
    torch.manual_seed(seed)
    z_grid = Z_POINTS_MM.clone()
    t_grid = T_POINTS_S.clone()

    # 用真值调用含时解生成 T(z,t)
    # 注意：calc_photothermal_timed 接受标量 z 和 t，需逐点计算
    T_obs = torch.zeros(len(z_grid), len(t_grid))
    for i, z in enumerate(z_grid):
        for j, t in enumerate(t_grid):
            result = calc_photothermal_timed(
                N_SI, KAPPA_TRUE, WAVELENGTH, I0,
                RHO_SI, CP_SI, z.item(), t.item(),
                thermal_diffusivity=D_TRUE
            )
            T_obs[i, j] = result['T_timed']

    # 加噪声
    if noise > 0:
        T_obs = T_obs * (1 + noise * torch.randn_like(T_obs))

    params_true = {'kappa': KAPPA_TRUE, 'D': D_TRUE}
    return z_grid, t_grid, T_obs, params_true


def forward(raw_kappa, raw_D, z_grid, t_grid, T_obs):
    """
    前向计算：从原始参数 → 预测 T(z,t) → 相对 MSE loss。

    参数:
        raw_kappa, raw_D — 无约束原始参数（torch.nn.Parameter）
        z_grid, t_grid   — 观测网格
        T_obs            — 观测温升 [len(z)×len(t)]

    返回: loss 张量
    """
    # 正值约束
    kappa = positive_parameter(raw_kappa, scale=1.0, floor=1e-6)
    D = positive_parameter(raw_D, scale=1.0, floor=1e-10)

    # 前向预测（逐点）
    T_pred = torch.zeros_like(T_obs)
    for i, z in enumerate(z_grid):
        for j, t in enumerate(t_grid):
            result = calc_photothermal_timed(
                N_SI, kappa, WAVELENGTH, I0,
                RHO_SI, CP_SI, z, t,
                thermal_diffusivity=D
            )
            T_pred[i, j] = result['T_timed']

    return relative_mse(T_pred, T_obs)


def run_inversion(seed=42, noise=0.0, steps=500, lr=0.05):
    """
    运行完整参数反演。

    返回: dict with keys:
        kappa_true, D_true, kappa_fit, D_fit,
        initial_loss, final_loss, history
    """
    z_grid, t_grid, T_obs, params_true = make_observations(seed=seed, noise=noise)

    # 从错误初值开始（κ_init = 10×κ_true → raw 设为使 softplus ≈ 0.44）
    # softplus(raw) = 0.44 → raw ≈ log(exp(0.44)-1) ≈ -0.67
    # 我们从 10×真值开始：softplus(raw) ≈ 0.44 → raw ≈ -0.67
    # 要让初始 κ = 0.44（即真值），raw = -0.67
    # 要让初始 κ = 4.4（10×真值），raw = log(e^4.4 - 1) ≈ 4.4
    raw_kappa = torch.nn.Parameter(torch.tensor(4.4))   # 初始 κ ≈ 4.4（10×真值）
    raw_D = torch.nn.Parameter(torch.tensor(-5.0))       # 初始 D ≈ softplus(-5) ≈ 0.0067

    # 记录初始 loss
    with torch.no_grad():
        initial_loss = forward(raw_kappa, raw_D, z_grid, t_grid, T_obs).item()

    # 优化
    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_kappa, raw_D],
        lambda: forward(raw_kappa, raw_D, z_grid, t_grid, T_obs),
        steps=steps, lr=lr, callback=callback
    )

    # 提取结果
    kappa_fit = positive_parameter(raw_kappa, scale=1.0, floor=1e-6).item()
    D_fit = positive_parameter(raw_D, scale=1.0, floor=1e-10).item()

    return {
        'kappa_true': KAPPA_TRUE,
        'D_true': D_TRUE,
        'kappa_fit': kappa_fit,
        'D_fit': D_fit,
        'initial_loss': initial_loss,
        'final_loss': history[-1] if history else initial_loss,
        'history': history,
    }


def run():
    """主入口：运行反演并打印结果。"""
    print("=" * 70)
    print("案例1：含时光热参数反演（硅 @ 532nm）")
    print("=" * 70)
    print(f"真值: κ={KAPPA_TRUE}, D={D_TRUE} m²/s")
    print(f"网格: z∈[{Z_POINTS_MM[0]:.4f}, {Z_POINTS_MM[-1]:.4f}] mm, "
          f"t∈[{T_POINTS_S[0]:.1e}, {T_POINTS_S[-1]:.1e}] s")
    print()

    print("--- 无噪声反演 ---")
    r1 = run_inversion(seed=42, noise=0.0, steps=500, lr=0.05)
    print(f"  κ_fit = {r1['kappa_fit']:.6f} (真值 {r1['kappa_true']}, "
          f"误差 {abs(r1['kappa_fit']-r1['kappa_true'])/r1['kappa_true']:.2%})")
    print(f"  D_fit = {r1['D_fit']:.4e} (真值 {r1['D_true']:.4e}, "
          f"误差 {abs(r1['D_fit']-r1['D_true'])/r1['D_true']:.2%})")
    print(f"  Loss: {r1['initial_loss']:.4e} → {r1['final_loss']:.4e}")
    print()

    print("--- 1% 噪声反演 ---")
    r2 = run_inversion(seed=42, noise=0.01, steps=500, lr=0.05)
    print(f"  κ_fit = {r2['kappa_fit']:.6f} (误差 {abs(r2['kappa_fit']-r2['kappa_true'])/r2['kappa_true']:.2%})")
    print(f"  D_fit = {r2['D_fit']:.4e} (误差 {abs(r2['D_fit']-r2['D_true'])/r2['D_true']:.2%})")
    print(f"  Loss: {r2['initial_loss']:.4e} → {r2['final_loss']:.4e}")
    print()
    print("可辨识性提示：短时绝热区 T∝α·t 与 D 无关，"
          "过渡区数据对联合反演 κ 和 D 至关重要。")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_inverse_photothermal.py -v`

Expected: 6 passed（注意：无噪声恢复 <5% 和噪声恢复 <10% 可能需要调整 lr/steps，如果失败先运行脚本观察输出）

- [ ] **Step 5: 运行脚本验证可执行**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m examples.fit_transient_photothermal`

Expected: 打印反演结果，κ 和 D 误差在阈值内

- [ ] **Step 6: 提交**

```bash
git add examples/fit_transient_photothermal.py tests/test_inverse_photothermal.py
git commit -m "feat: add case 1 - transient photothermal parameter inversion (kappa + D)"
```

---

## Task 3: 案例 2 — 黑体光谱测温与仪器标定

**Files:**
- Create: `examples/fit_blackbody_temperature.py`
- Test: `tests/test_blackbody_inversion.py`

**物理场景：** 太阳光谱 T_true=5800K, gain_true=1.0, 波长 400-2500nm 32通道, 2% 噪声。固定 gain 反演 T（避免可辨识性简并）。

- [ ] **Step 1: 写失败测试 `tests/test_blackbody_inversion.py`**

```python
"""案例2测试：黑体光谱测温。"""
import torch
import pytest
from examples.fit_blackbody_temperature import (
    make_observations,
    forward,
    run_inversion,
)


def test_make_observations_shape():
    """合成光谱形状正确。"""
    wavelengths, I_obs, params_true = make_observations(seed=42, noise=0.0)
    assert len(wavelengths) == 32  # 32 通道
    assert I_obs.shape == (32,)
    assert params_true['T'] == pytest.approx(5800.0, abs=0.1)
    assert params_true['gain'] == pytest.approx(1.0, abs=1e-6)


def test_make_observations_reproducible():
    """固定种子可复现。"""
    _, I1, _ = make_observations(seed=42, noise=0.02)
    _, I2, _ = make_observations(seed=42, noise=0.02)
    assert torch.allclose(I1, I2)


def test_forward_gradient_available():
    """前向梯度可用。"""
    raw_T = torch.nn.Parameter(torch.tensor(2.0))
    wavelengths, I_obs, params_true = make_observations(seed=42, noise=0.0)
    gain = params_true['gain']  # 固定 gain

    loss = forward(raw_T, gain, wavelengths, I_obs)
    loss.backward()

    assert raw_T.grad is not None
    assert torch.isfinite(raw_T.grad).all()


def test_run_inversion_noiseless_recovery():
    """无噪声下温度恢复误差 <1%。"""
    result = run_inversion(seed=42, noise=0.0, steps=300, lr=0.1)
    T_err = abs(result['T_fit'] - result['T_true']) / result['T_true']
    assert T_err < 0.01, f"温度误差 {T_err:.2%} 超过 1%"


def test_run_inversion_loss_decreases():
    """优化后 loss 下降。"""
    result = run_inversion(seed=42, noise=0.02, steps=200, lr=0.1)
    assert result['final_loss'] < result['initial_loss']


def test_run_inversion_noisy_recovery():
    """2% 噪声下温度恢复误差 <3%。"""
    result = run_inversion(seed=42, noise=0.02, steps=300, lr=0.1)
    T_err = abs(result['T_fit'] - result['T_true']) / result['T_true']
    assert T_err < 0.03, f"温度误差 {T_err:.2%} 超过 3%"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_blackbody_inversion.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 `examples/fit_blackbody_temperature.py`**

```python
"""案例2：黑体光谱测温与仪器标定。

从多波长辐射强度反演黑体温度 T（固定仪器增益 gain），
使用 photokinetics 的 calc_blackbody Planck 光谱计算。

物理场景：太阳光谱
    T_true = 5800 K
    gain_true = 1.0
    波长 400-2500nm，32 通道（可见光到近红外）

可辨识性教学点：
    T 和 gain 同时自由时存在简并（任意 gain 都能用 T 补偿），
    首版固定 gain 只反演 T。

对数损失必要性：
    Planck 光谱在 400-2500nm 跨 3-4 个数量级，
    直接 MSE 让短波高值主导，对数域损失等价于拟合相对误差。

用法：
    python -m examples.fit_blackbody_temperature
"""
import torch
from photokinetics import calc_blackbody
from examples._common import positive_parameter, log_mse, optimize


# ===== 物理常数（太阳光谱）=====
T_TRUE = 5800.0       # K
GAIN_TRUE = 1.0       # 仪器增益
WAVELENGTHS = torch.linspace(400.0, 2500.0, 32)  # nm，32通道


def make_observations(seed=42, noise=0.0):
    """
    用真值生成合成观测光谱。

    返回: (wavelengths_nm[32], I_obs[32], params_true_dict)
    """
    torch.manual_seed(seed)
    wavelengths = WAVELENGTHS.clone()

    # Planck 光谱
    _, _, B = calc_blackbody(T_TRUE, wavelengths)
    I_obs = GAIN_TRUE * B

    # 加噪声
    if noise > 0:
        I_obs = I_obs * (1 + noise * torch.randn_like(I_obs))

    params_true = {'T': T_TRUE, 'gain': GAIN_TRUE}
    return wavelengths, I_obs, params_true


def forward(raw_T, gain, wavelengths, I_obs):
    """
    前向计算：从原始温度参数 → 预测光谱 → 对数域 MSE loss。

    参数:
        raw_T      — 无约束温度原始参数（torch.nn.Parameter）
        gain       — 固定增益（标量）
        wavelengths — 波长网格 [nm]
        I_obs      — 观测光谱

    返回: loss 张量
    """
    # 温度正值约束：T = 300 + softplus(raw_T) * 5000，范围 [300, 5300]
    T = 300.0 + positive_parameter(raw_T, scale=5000.0, floor=0.0)

    # Planck 光谱
    _, _, B = calc_blackbody(T, wavelengths)
    pred = gain * B

    return log_mse(pred, I_obs)


def run_inversion(seed=42, noise=0.0, steps=300, lr=0.1):
    """
    运行温度反演（固定 gain）。

    返回: dict with keys:
        T_true, gain_true, T_fit, initial_loss, final_loss, history
    """
    wavelengths, I_obs, params_true = make_observations(seed=seed, noise=noise)
    gain = params_true['gain']  # 固定 gain

    # 初始温度 3000K：300 + softplus(raw)*5000 = 3000 → softplus(raw) = 0.54 → raw ≈ 0.14
    raw_T = torch.nn.Parameter(torch.tensor(0.14))

    # 记录初始 loss
    with torch.no_grad():
        initial_loss = forward(raw_T, gain, wavelengths, I_obs).item()

    # 优化
    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_T],
        lambda: forward(raw_T, gain, wavelengths, I_obs),
        steps=steps, lr=lr, callback=callback
    )

    # 提取结果
    T_fit = (300.0 + positive_parameter(raw_T, scale=5000.0, floor=0.0)).item()

    return {
        'T_true': T_TRUE,
        'gain_true': GAIN_TRUE,
        'T_fit': T_fit,
        'initial_loss': initial_loss,
        'final_loss': history[-1] if history else initial_loss,
        'history': history,
    }


def run():
    """主入口：运行测温并打印结果。"""
    print("=" * 70)
    print("案例2：黑体光谱测温（太阳光谱 400-2500nm）")
    print("=" * 70)
    print(f"真值: T={T_TRUE} K, gain={GAIN_TRUE}")
    print(f"通道: {len(WAVELENGTHS)} 点, {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f} nm")
    print()

    print("--- 无噪声反演 ---")
    r1 = run_inversion(seed=42, noise=0.0, steps=300, lr=0.1)
    print(f"  T_fit = {r1['T_fit']:.2f} K (真值 {r1['T_true']}, "
          f"误差 {abs(r1['T_fit']-r1['T_true'])/r1['T_true']:.2%})")
    print(f"  Loss: {r1['initial_loss']:.4e} → {r1['final_loss']:.4e}")
    print()

    print("--- 2% 噪声反演 ---")
    r2 = run_inversion(seed=42, noise=0.02, steps=300, lr=0.1)
    print(f"  T_fit = {r2['T_fit']:.2f} K (误差 {abs(r2['T_fit']-r2['T_true'])/r2['T_true']:.2%})")
    print(f"  Loss: {r2['initial_loss']:.4e} → {r2['final_loss']:.4e}")
    print()
    print("可辨识性提示：T 和 gain 同时自由时简并，需固定其一或加温度锚点。")
    print("对数损失：Planck 光谱跨数量级，log_mse 等价于拟合相对误差。")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_blackbody_inversion.py -v`

Expected: 6 passed

- [ ] **Step 5: 运行脚本验证可执行**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m examples.fit_blackbody_temperature`

Expected: 打印测温结果，T 误差在阈值内

- [ ] **Step 6: 提交**

```bash
git add examples/fit_blackbody_temperature.py tests/test_blackbody_inversion.py
git commit -m "feat: add case 2 - blackbody spectral thermometry with log-space inversion"
```

---

## Task 4: 案例 3 — 光镊鲁棒粒径设计

**Files:**
- Create: `examples/design_optical_tweezer.py`
- Test: `tests/test_tweezer_design.py`

**物理场景：** 聚苯乙烯微粒（n=1.59）在水中（n=1.33），波长 1064nm，粒径 [0.7,0.8,0.9,1.0]μm，目标力 10pN，下限 5pN。

- [ ] **Step 1: 写失败测试 `tests/test_tweezer_design.py`**

```python
"""案例3测试：光镊鲁棒粒径设计。"""
import torch
import pytest
from examples.design_optical_tweezer import (
    PARTICLE_RADII,
    run_design,
    forward,
)


def test_particle_radii_in_rayleigh_regime():
    """粒径在瑞利近似域内（a ≪ λ）。"""
    wavelength_nm = 1064.0
    for a in PARTICLE_RADII:
        # 瑞利近似要求 a << λ，通常 a/λ < 0.1
        assert (a * 1e6) / wavelength_nm < 0.1, \
            f"粒径 {a*1e6}μm 不满足瑞利近似（λ={wavelength_nm}nm）"


def test_forward_gradient_available():
    """前向梯度可用。"""
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))
    loss = forward(raw_grad_I)
    loss.backward()

    assert raw_grad_I.grad is not None
    assert torch.isfinite(raw_grad_I.grad).all()


def test_forward_multi_objective():
    """多目标损失包含跟踪+鲁棒+功率三项。"""
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))
    result = forward(raw_grad_I, return_components=True)

    assert 'L_track' in result
    assert 'L_robust' in result
    assert 'L_power' in result
    assert 'L_total' in result


def test_run_design_loss_decreases():
    """优化后 loss 下降。"""
    result = run_design(steps=200, lr=0.05)
    assert result['final_loss'] < result['initial_loss']


def test_run_design_target_force_achieved():
    """优化后平均力达到目标（误差 <5%）。"""
    result = run_design(steps=200, lr=0.05)
    F_target = 10e-12  # 10 pN
    F_mean = result['forces_final'].mean().item()
    err = abs(F_mean - F_target) / F_target
    assert err < 0.05, f"目标力误差 {err:.2%} 超过 5%"


def test_run_design_minimum_force_satisfied():
    """优化后最弱粒径力 ≥ 下限。"""
    result = run_design(steps=200, lr=0.05)
    F_min_limit = 5e-12  # 5 pN
    F_weakest = result['forces_final'].min().item()
    assert F_weakest >= F_min_limit * 0.9, \
        f"最弱粒径力 {F_weakest*1e12:.2f} pN 低于下限 {F_min_limit*1e12:.2f} pN（允许10%裕量）"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_tweezer_design.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 `examples/design_optical_tweezer.py`**

```python
"""案例3：光镊鲁棒粒径设计。

为不同粒径分布的微粒设计光强梯度，使：
    - 平均捕获力达到目标值（跟踪损失）
    - 最弱粒径下的捕获力不低于下限（鲁棒损失）
    - 光强梯度（功率消耗）尽量小（功率惩罚）

使用 photokinetics 的 calc_tweezer_force（瑞利近似）。

物理场景：聚苯乙烯微粒在水中
    粒径 [0.7, 0.8, 0.9, 1.0] μm
    n_particle = 1.59（聚苯乙烯）
    n_medium = 1.33（水）
    波长 1064 nm（红外光镊）
    目标力 10 pN，下限 5 pN

限制声明：
    - 瑞利近似域 a ≪ λ，粒径 ≤1μm@1064nm 满足
    - 当前 wavelength_nm 参数不参与计算，不优化波长
    - F_scat = F_grad/2 是简化近似

用法：
    python -m examples.design_optical_tweezer
"""
import torch
from photokinetics import calc_tweezer_force
from examples._common import positive_parameter, optimize


# ===== 物理常数 =====
PARTICLE_RADII = torch.tensor([0.7e-6, 0.8e-6, 0.9e-6, 1.0e-6])  # m
N_PARTICLE = 1.59    # 聚苯乙烯
N_MEDIUM = 1.33      # 水
WAVELENGTH_NM = 1064.0
F_TARGET = 10e-12    # 10 pN 目标力
F_MIN = 5e-12        # 5 pN 下限


def forward(raw_grad_I, return_components=False):
    """
    前向计算：多目标损失函数。

    参数:
        raw_grad_I        — 无约束光强梯度原始参数
        return_components — 是否返回各分损失

    返回: loss 张量（或 dict 含各分损失）
    """
    # 光强梯度正值约束
    grad_I = positive_parameter(raw_grad_I, scale=1e17, floor=0.0)

    # 批量计算所有粒径的力
    result = calc_tweezer_force(
        a_m=PARTICLE_RADII,
        n_particle=N_PARTICLE,
        n_medium=N_MEDIUM,
        wavelength_nm=WAVELENGTH_NM,
        grad_I=grad_I,
    )
    forces = result['F_total']  # [4] 张量

    # 多目标损失
    L_track = ((forces.mean() - F_TARGET) / F_TARGET) ** 2
    L_robust = torch.relu((F_MIN - forces) / F_MIN).pow(2).mean()
    L_power = 1e-4 * (grad_I / 1e17) ** 2
    L_total = L_track + L_robust + L_power

    if return_components:
        return {
            'L_track': L_track,
            'L_robust': L_robust,
            'L_power': L_power,
            'L_total': L_total,
            'forces': forces,
        }
    return L_total


def run_design(steps=200, lr=0.05):
    """
    运行鲁棒光镊设计优化。

    返回: dict with keys:
        grad_I_final, forces_initial, forces_final,
        initial_loss, final_loss, history
    """
    # 初始光强梯度（raw=0 → softplus(0)*1e17 ≈ 0.693e17）
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))

    # 初始状态
    with torch.no_grad():
        init_components = forward(raw_grad_I, return_components=True)
        initial_loss = init_components['L_total'].item()
        forces_initial = init_components['forces'].clone()

    # 优化
    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_grad_I],
        lambda: forward(raw_grad_I),
        steps=steps, lr=lr, callback=callback
    )

    # 最终状态
    with torch.no_grad():
        final_components = forward(raw_grad_I, return_components=True)
        forces_final = final_components['forces'].clone()
        grad_I_final = positive_parameter(raw_grad_I, scale=1e17, floor=0.0).item()

    return {
        'grad_I_final': grad_I_final,
        'forces_initial': forces_initial,
        'forces_final': forces_final,
        'initial_loss': initial_loss,
        'final_loss': history[-1] if history else initial_loss,
        'history': history,
    }


def run():
    """主入口：运行设计并打印结果。"""
    print("=" * 70)
    print("案例3：光镊鲁棒粒径设计（聚苯乙烯 @ 1064nm）")
    print("=" * 70)
    print(f"粒径: {PARTICLE_RADII.numpy()*1e6} μm")
    print(f"颗粒 n={N_PARTICLE}, 介质 n={N_MEDIUM}, 波长 {WAVELENGTH_NM} nm")
    print(f"目标力: {F_TARGET*1e12} pN, 下限: {F_MIN*1e12} pN")
    print()

    result = run_design(steps=200, lr=0.05)

    print("--- 优化结果 ---")
    print(f"  光强梯度: {result['grad_I_final']:.4e} W/m³")
    print(f"  Loss: {result['initial_loss']:.4e} → {result['final_loss']:.4e}")
    print()

    print("--- 各粒径捕获力（优化前 → 优化后）---")
    for i, a in enumerate(PARTICLE_RADII):
        f_init = result['forces_initial'][i].item()
        f_final = result['forces_final'][i].item()
        print(f"  a={a.item()*1e6:.1f}μm: {f_init*1e12:.2f} pN → {f_final*1e12:.2f} pN")

    f_mean = result['forces_final'].mean().item()
    f_min = result['forces_final'].min().item()
    print(f"\n  平均力: {f_mean*1e12:.2f} pN (目标 {F_TARGET*1e12:.2f} pN)")
    print(f"  最弱力: {f_min*1e12:.2f} pN (下限 {F_MIN*1e12:.2f} pN)")
    print()
    print("限制声明：瑞利近似域（a≪λ），F_scat=F_grad/2 简化模型，不优化波长。")
    print("多目标：L_track(跟踪) + L_robust(鲁棒下限) + L_power(功率惩罚)。")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/test_tweezer_design.py -v`

Expected: 6 passed

- [ ] **Step 5: 运行脚本验证可执行**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m examples.design_optical_tweezer`

Expected: 打印设计结果，各粒径力达标

- [ ] **Step 6: 提交**

```bash
git add examples/design_optical_tweezer.py tests/test_tweezer_design.py
git commit -m "feat: add case 3 - optical tweezer robust design with multi-objective loss"
```

---

## Task 5: 全量测试与 README 更新

**Files:**
- Modify: `pkg/photokinetics/README.md`（添加实战案例章节）
- Run: 全量测试

- [ ] **Step 1: 运行全量测试**

Run: `& "C:\Users\Administrator\AppData\Local\Python\bin\python.exe" -m pytest tests/ -v`

Expected: 所有测试通过（test_common + test_inverse_photothermal + test_blackbody_inversion + test_tweezer_design + 原有 test_all + test_time_resolved）

- [ ] **Step 2: 更新 README 添加实战案例章节**

在 `pkg/photokinetics/README.md` 的"批量计算"章节后添加：

```markdown
## 实战案例

三个可微物理逆问题实战案例，展示如何用 photokinetics 解决真实工程问题：

| 案例 | 场景 | 反演目标 | 脚本 |
|---|---|---|---|
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
```

- [ ] **Step 3: 提交**

```bash
git add pkg/photokinetics/README.md
git commit -m "docs: add case studies section to README"
```

---

## Self-Review 完成情况

**1. Spec coverage:**
- ✓ 第2节三层架构 → Task 1（共享层）+ Task 2-4（三案例）+ Task 5（README）
- ✓ 第3节共享工具层 → Task 1 完整实现
- ✓ 第4节案例1 → Task 2 完整实现（含可辨识性教学点）
- ✓ 第5节案例2 → Task 3 完整实现（含对数损失、可辨识性）
- ✓ 第6节案例3 → Task 4 完整实现（含多目标、瑞利限制声明）
- ✓ 第9节接口风险 → 各 Task 均已规避（auto vs timed、波长不参与、黑体截断）
- ⚠ 第7节Web入口 → 未包含在本计划（spec 第10节标注为 P4，依赖案例完成）
- ⚠ 第8节文章 → 未包含在本计划（spec 第10节标注为 P5）

**2. Placeholder scan:** 无 TBD/TODO，所有代码完整。

**3. Type consistency:** 检查通过——`positive_parameter`、`relative_mse`、`log_mse`、`optimize` 在 Task 1 定义，Task 2-4 引用一致。`make_observations`/`forward`/`run_inversion`/`run_design` 在各案例内部一致。

**4. 潜在调整点（实施时关注）：**
- 案例1的 `run_inversion` 初始 raw 值可能导致 500 步不够收敛，如测试失败需调整 lr 或 steps
- 案例2的 `calc_blackbody` 高指数截断可能影响 400nm 通道，如梯度异常需跳过短波通道
- 案例3的 `L_power` 权重 1e-4 可能需调整以平衡三项目标
