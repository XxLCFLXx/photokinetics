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
    """用真值生成合成观测光谱。"""
    torch.manual_seed(seed)
    wavelengths = WAVELENGTHS.clone()

    _, _, B = calc_blackbody(T_TRUE, wavelengths)
    I_obs = GAIN_TRUE * B

    if noise > 0:
        I_obs = I_obs * (1 + noise * torch.randn_like(I_obs))

    params_true = {'T': T_TRUE, 'gain': GAIN_TRUE}
    return wavelengths, I_obs, params_true


def forward(raw_T, gain, wavelengths, I_obs):
    """前向计算：从原始温度参数 → 预测光谱 → 对数域 MSE loss。"""
    T = 300.0 + positive_parameter(raw_T, scale=5000.0, floor=0.0)

    _, _, B = calc_blackbody(T, wavelengths)
    pred = gain * B

    return log_mse(pred, I_obs)


def run_inversion(seed=42, noise=0.0, steps=300, lr=0.1):
    """运行温度反演（固定 gain）。"""
    wavelengths, I_obs, params_true = make_observations(seed=seed, noise=noise)
    gain = params_true['gain']

    raw_T = torch.nn.Parameter(torch.tensor(0.14))

    with torch.no_grad():
        initial_loss = forward(raw_T, gain, wavelengths, I_obs).item()

    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_T],
        lambda: forward(raw_T, gain, wavelengths, I_obs),
        steps=steps, lr=lr, callback=callback
    )

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
