"""含时热传导解析解模块（突破绝热近似）。

基于 Carslaw & Jaeger (1959) 的半无限域含源热传导解析解。
使用 erfcx 函数和渐近展开保证数值稳定性。

物理模型:
    1D 含源热传导方程:
        ∂T/∂t = D·∂²T/∂z² + q(z)/(ρCp)
    其中:
        D = κ_th/(ρCp)             热扩散率 (m²/s)
        q(z) = α·I₀·exp(-αz)      光热源 (W/m³)

    解析解 (Carslaw & Jaeger, §2.9):
        T(z,t) = (I₀·α)/(ρCp) · ∫₀ᵗ G(z,τ) dτ

        其中 G(z,τ) = 0.5·exp(-u²)·[erfcx(w1) + erfcx(w2)]
        u = z/(2√(Dτ)),  v = α√(Dτ)
        w1 = v - u,  w2 = v + u

    数值稳定:
        当 u > 5 (z >> √(Dτ), 热扩散未到达) 时，
        用渐近展开: exp(-u²)·erfcx(v-u) ≈ exp(-u² + (v-u)²)·2 ≈ exp(v²-2uv)·2

参考:
    Carslaw, H.S. & Jaeger, J.C. (1959)
    "Conduction of Heat in Solids", 2nd ed., Oxford Univ. Press
"""
import torch
from photokinetics.constants import C


def _erfcx_torch(x):
    """数值稳定的 erfcx(x) = exp(x²)·erfc(x)。"""
    try:
        return torch.special.erfcx(x)
    except AttributeError:
        # Fallback for older PyTorch
        return torch.exp(x * x) * torch.erfc(x)


def _green_integrand_safe(z, tau, alpha, D):
    """
    数值稳定的 Green 函数积分核。
    G(z,τ) = 0.5·exp(-u²)·[erfcx(w1) + erfcx(w2)]

    当 u > 5 时用 log-space 计算避免 0×inf。
    """
    sqrt_Dtau = torch.sqrt(D * tau + 1e-30)
    u = z / (2.0 * sqrt_Dtau + 1e-30)
    v = alpha * sqrt_Dtau

    w1 = v - u  # 可能很负
    w2 = v + u  # 总是正

    # exp(-u²) 项
    exp_neg_u2 = torch.exp(-u * u)

    # ===== 分支1: u 较小（热扩散已到达），正常计算 =====
    # erfcx(w1) 和 erfcx(w2) 都是有限值
    erfcx_w1_normal = _erfcx_torch(w1)
    erfcx_w2 = _erfcx_torch(w2)
    green_normal = 0.5 * exp_neg_u2 * (erfcx_w1_normal + erfcx_w2)

    # ===== 分支2: u 很大（热扩散未到达），用渐近展开 =====
    # 当 u >> v 时:
    #   w1 = v - u → 大负数
    #   erfcx(w1) = exp(w1²)·erfc(w1) ≈ exp(w1²)·2  (erfc(-x)→2 for x→∞)
    #   exp(-u²)·erfcx(w1) = exp(-u² + w1²)·2 = exp(-u² + (v-u)²)·2
    #                      = exp(v² - 2uv)·2
    #   exp(-u²)·erfcx(w2) = exp(-u²)·erfcx(v+u) ≈ exp(-u²)/(w2·√π) (很小)
    # 所以 G ≈ 0.5·exp(v² - 2uv)·2 = exp(v² - 2uv)
    exponent = v * v - 2.0 * u * v  # v² - 2uv，注意 u > v 时为负
    # 限制指数避免下溢
    exponent_clamped = torch.clamp(exponent, max=50.0)
    green_asymp = torch.exp(exponent_clamped)

    # 选择分支: u > 3 时用渐近展开
    use_asymp = u > 3.0
    green = torch.where(use_asymp, green_asymp, green_normal)

    # 清理 nan/inf
    green = torch.where(
        torch.isnan(green) | torch.isinf(green),
        torch.zeros_like(green),
        green
    )

    return green


def calc_photothermal_timed(
    n, kappa, wavelength_nm, I0, rho, Cp, depth_mm, time_s,
    thermal_diffusivity=None, k_thermal=None
):
    """
    含时光热解析解（突破绝热近似）。

    使用 8 点 Gauss-Legendre 积分计算 Carslaw-Jaeger 公式，
    数值稳定地处理热扩散未到达深度的极限情况。

    参数:
        n                    — 折射率
        kappa                — 消光系数
        wavelength_nm        — 波长 (nm)
        I0                   — 入射光强 (W/m²)
        rho                  — 密度 (kg/m³)
        Cp                   — 比热 (J/(kg·K))
        depth_mm             — 深度 (mm)
        time_s               — 照射时间 (s)
        thermal_diffusivity  — 热扩散率 D (m²/s)，可选
        k_thermal            — 热导率 (W/(m·K))，可选

    返回: dict with keys:
        alpha         — 吸收系数 (1/m)
        D             — 热扩散率 (m²/s)
        T_adiabatic   — 绝热近似温升 (K)
        T_timed       — 含时温升 (K)，本模块主结果
        T_steady      — 稳态温升 (K)，参考值
        regime        — 物理regime: 'adiabatic' / 'transient' / 'steady'
    """
    # ===== 转换为张量 =====
    n = torch.as_tensor(n, dtype=torch.float32)
    kappa = torch.as_tensor(kappa, dtype=torch.float32)
    lam = torch.as_tensor(wavelength_nm, dtype=torch.float32) * 1e-9
    I0 = torch.as_tensor(I0, dtype=torch.float32)
    rho = torch.as_tensor(rho, dtype=torch.float32)
    Cp = torch.as_tensor(Cp, dtype=torch.float32)
    z = torch.as_tensor(depth_mm, dtype=torch.float32) * 1e-3
    t = torch.as_tensor(time_s, dtype=torch.float32)

    # ===== Step 1: 光学吸收 =====
    alpha = 4.0 * torch.pi * kappa / (n * lam)

    # ===== Step 2: 热扩散率 =====
    if thermal_diffusivity is not None:
        D = torch.as_tensor(thermal_diffusivity, dtype=torch.float32)
    elif k_thermal is not None:
        k_th = torch.as_tensor(k_thermal, dtype=torch.float32)
        D = k_th / (rho * Cp)
    else:
        raise ValueError("必须提供 thermal_diffusivity 或 k_thermal 之一")

    # ===== Step 3: 绝热近似温升 =====
    q_z = alpha * I0 * torch.exp(-alpha * z)
    T_adiabatic = q_z * t / (rho * Cp)

    # ===== Step 4: 含时解析解 =====
    # T(z,t) = (I₀·α)/(ρCp) · ∫₀ᵗ G(z,τ) dτ
    # 用变量替换 s = τ/t, 积分区间变为 [0,1]
    # T(z,t) = (I₀·α·t)/(ρCp) · ∫₀¹ G(z, s·t) ds

    t_safe = torch.where(t > 0, t, torch.tensor(1e-30))

    # 8 点 Gauss-Legendre 节点和权重 (更高精度)
    gl_nodes = torch.tensor([
        0.0198550717512319,
        0.1016667612931866,
        0.2372337950418355,
        0.4082826787521752,
        0.5917173212478248,
        0.7627662049581645,
        0.8983332387068134,
        0.9801449282487681,
    ])
    gl_weights = torch.tensor([
        0.0506142681451881,
        0.1111905172266872,
        0.1568533229389436,
        0.1813418916891810,
        0.1813418916891810,
        0.1568533229389436,
        0.1111905172266872,
        0.0506142681451881,
    ])

    # 积分累加
    integral = torch.zeros_like(z)
    for node, weight in zip(gl_nodes, gl_weights):
        tau = node * t_safe
        g = _green_integrand_safe(z, tau, alpha, D)
        integral = integral + weight * g

    T_timed = (I0 * alpha * t_safe) / (rho * Cp) * integral

    # 处理 t=0 情况
    T_timed = torch.where(t > 0, T_timed, torch.zeros_like(T_timed))

    # ===== 稳态参考值（仅用于对照，非半无限域真实极限）=====
    # 注意: 半无限域含源热传导无真正稳态，长时间温度按 √t 持续增长
    #       (Carslaw & Jaeger 渐近: T(z,t) ~ 2·I₀·√(t/π) / √(ρCp·κ_th))
    # 此处 T_steady 是"有限厚度一维平板"公式，仅作数量级参考，
    # 不应作为长时间极限使用。
    k_thermal_eff = D * rho * Cp
    T_steady = I0 / (k_thermal_eff * alpha) * torch.exp(-alpha * z)

    # ===== 物理regime判定 =====
    dimensionless_time = t * D * alpha * alpha
    regime_code = torch.where(
        dimensionless_time < 0.01,
        torch.tensor(0),
        torch.where(
            dimensionless_time > 10.0,
            torch.tensor(2),
            torch.tensor(1)
        )
    )
    regime_map = {0: 'adiabatic', 1: 'transient', 2: 'steady'}
    regime = regime_map[regime_code.item()] if regime_code.dim() == 0 else regime_code

    return {
        'alpha': alpha,
        'D': D,
        'T_adiabatic': T_adiabatic,
        'T_timed': T_timed,
        'T_steady': T_steady,
        'regime': regime,
        'dimensionless_time': dimensionless_time,
    }


def calc_photothermal_auto(n, kappa, wavelength_nm, I0, rho, Cp, depth_mm, time_s,
                           thermal_diffusivity=None, k_thermal=None):
    """
    自动选择绝热近似或含时解（根据物理regime）。
    """
    result = calc_photothermal_timed(
        n, kappa, wavelength_nm, I0, rho, Cp, depth_mm, time_s,
        thermal_diffusivity=thermal_diffusivity, k_thermal=k_thermal
    )

    dt = result['dimensionless_time']
    if dt < 0.01:
        result['T'] = result['T_adiabatic']
        result['method'] = 'adiabatic'
    else:
        result['T'] = result['T_timed']
        result['method'] = 'timed'

    return result
