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
# z ∈ [1e-5, 1e-2] mm = [0.01μm, 10μm]，覆盖 0.004~4 个吸收长度（1/α≈4μm）
Z_POINTS_MM = torch.logspace(-5, -2, 10)
T_POINTS_S = torch.logspace(-9, -3, 10)         # [1ns, 1ms]，对数10点


def make_observations(seed=42, noise=0.0):
    """用真值生成合成观测数据。"""
    torch.manual_seed(seed)
    z_grid = Z_POINTS_MM.clone()
    t_grid = T_POINTS_S.clone()

    T_obs = torch.zeros(len(z_grid), len(t_grid))
    for i, z in enumerate(z_grid):
        for j, t in enumerate(t_grid):
            result = calc_photothermal_timed(
                N_SI, KAPPA_TRUE, WAVELENGTH, I0,
                RHO_SI, CP_SI, z.item(), t.item(),
                thermal_diffusivity=D_TRUE
            )
            T_obs[i, j] = result['T_timed']

    if noise > 0:
        T_obs = T_obs * (1 + noise * torch.randn_like(T_obs))

    params_true = {'kappa': KAPPA_TRUE, 'D': D_TRUE}
    return z_grid, t_grid, T_obs, params_true


def forward(raw_kappa, raw_D, z_grid, t_grid, T_obs):
    """前向计算：从原始参数 → 预测 T(z,t) → 相对 MSE loss。

    对热扩散未到达的 (z,t) 对（u = z/(2√(D·τ_min)) > 3），
    在 no_grad 下计算以避免库函数 erfcx 在大负参数下溢出导致的 NaN 梯度。
    这些点处于绝热区 T∝α·t，对 D 不敏感，不影响 D 的反演。
    """
    kappa = positive_parameter(raw_kappa, scale=1.0, floor=1e-6)
    D = positive_parameter(raw_D, scale=1.0, floor=1e-10)

    T_pred = torch.zeros_like(T_obs)
    for i, z in enumerate(z_grid):
        z_m = z * 1e-3  # mm → m
        for j, t in enumerate(t_grid):
            # 安全性判定：最小 GL 节点 τ_min ≈ 0.02·t 下 u = z/(2√(D·τ_min))
            with torch.no_grad():
                u_est = z_m / (2.0 * torch.sqrt(D * t * 0.02 + 1e-30))
                safe = bool(u_est < 3.0)
            if safe:
                result = calc_photothermal_timed(
                    N_SI, kappa, WAVELENGTH, I0,
                    RHO_SI, CP_SI, z, t,
                    thermal_diffusivity=D
                )
                T_pred[i, j] = result['T_timed']
            else:
                with torch.no_grad():
                    result = calc_photothermal_timed(
                        N_SI, kappa, WAVELENGTH, I0,
                        RHO_SI, CP_SI, z, t,
                        thermal_diffusivity=D
                    )
                    T_pred[i, j] = result['T_timed'].detach()

    return relative_mse(T_pred, T_obs)


def run_inversion(seed=42, noise=0.0, steps=500, lr=0.05):
    """运行完整参数反演。"""
    z_grid, t_grid, T_obs, params_true = make_observations(seed=seed, noise=noise)

    raw_kappa = torch.nn.Parameter(torch.tensor(-1.0))
    raw_D = torch.nn.Parameter(torch.tensor(-9.0))

    with torch.no_grad():
        initial_loss = forward(raw_kappa, raw_D, z_grid, t_grid, T_obs).item()

    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_kappa, raw_D],
        lambda: forward(raw_kappa, raw_D, z_grid, t_grid, T_obs),
        steps=steps, lr=lr, callback=callback
    )

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
