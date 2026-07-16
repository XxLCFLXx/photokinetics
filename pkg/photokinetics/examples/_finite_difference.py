"""隐式后向欧拉有限差分求解器（1D 含源热传导）。

用于生成 PINN 案例的"真值" T(z,t)，支持非均匀 κ(z) 分布。

物理模型:
    ∂T/∂t = D · ∂²T/∂z² + q(z; κ) / (ρCp)
    q(z) = α(z) · I(z),   α(z) = 4πκ(z)/(nλ)
    I(z) = I₀ · exp(-∫₀ᶻ α(z') dz')   (非均匀比尔-朗伯)

边界条件:
    z=0:  ∂T/∂z = 0   (绝热表面, Neumann)
    z=L:  T = 0       (远场, Dirichlet)

数值方法:
    隐式后向欧拉 + 三对角求解器 (scipy.linalg.solve_banded)
    无条件稳定，可任意细化 Δz

用法:
    from examples._finite_difference import solve_implicit_fd, kappa_true_func
    z_array, t_array, T_field = solve_implicit_fd(...)
"""
import numpy as np
from scipy.linalg import solve_banded


# ===== 真值 κ(z) 设计 =====

# 聚合物 @ 532nm 参数
N_POLY = 1.49
KAPPA0 = 0.005          # 基底消光系数
RHO_POLY = 1190.0       # kg/m³
CP_POLY = 1420.0        # J/(kg·K)
K_TH_POLY = 0.2         # W/(m·K)
WAVELENGTH_NM = 532.0   # nm
LAMBDA_M = WAVELENGTH_NM * 1e-9

# 缺陷层参数
DEFECT_AMP = 3.0        # A: 峰值放大倍数（峰处 κ = 4κ₀）
DEFECT_Z0 = 5e-6        # m, 缺陷层中心深度 5μm
DEFECT_SIGMA = 2e-6     # m, 缺陷层宽度 2μm

# 热扩散率（派生量）
D_POLY = K_TH_POLY / (RHO_POLY * CP_POLY)   # ≈ 1.18e-7 m²/s


def kappa_true_func(z):
    """真值消光系数分布 κ(z) = κ₀·(1 + A·exp(-(z-z₀)²/(2σ²)))。

    z 单位：m，可以是标量或 numpy 数组。
    返回：κ(z)（与 z 同形状）
    """
    z = np.asarray(z, dtype=np.float64)
    defect = DEFECT_AMP * np.exp(-(z - DEFECT_Z0) ** 2 / (2.0 * DEFECT_SIGMA ** 2))
    return KAPPA0 * (1.0 + defect)


def compute_alpha_array(z_array, kappa_array):
    """α(z) = 4π κ(z) / (n λ)。"""
    return 4.0 * np.pi * kappa_array / (N_POLY * LAMBDA_M)


def compute_intensity_array(z_array, kappa_array, I0):
    """非均匀比尔-朗伯光强: I(z) = I₀·exp(-cumtrapz(α, z))。

    用梯形积分计算累积光学深度。
    """
    alpha = compute_alpha_array(z_array, kappa_array)
    # 累积梯形积分: optical_depth[i] = ∫₀^{z_i} α(z') dz'
    optical_depth = np.zeros_like(alpha)
    optical_depth[1:] = np.cumsum(0.5 * (alpha[:-1] + alpha[1:]) * np.diff(z_array))
    return I0 * np.exp(-optical_depth)


def compute_source_array(z_array, kappa_array, I0):
    """热源 q(z) = α(z)·I(z)。"""
    alpha = compute_alpha_array(z_array, kappa_array)
    I = compute_intensity_array(z_array, kappa_array, I0)
    return alpha * I


# ===== 隐式后向欧拉求解器 =====

def solve_implicit_fd(kappa_array, z_array, t_array, I0, D, rho, Cp):
    """隐式后向欧拉求解 1D 含源热传导。

    方程:
        (T^{n+1}_i - T^n_i)/Δt = D·(T^{n+1}_{i+1} - 2T^{n+1}_i + T^{n+1}_{i-1})/Δz² + q_i/(ρCp)

    整理:
        (I - D·Δt/Δz²·A)·T^{n+1} = T^n + Δt·q/(ρCp)

    其中 A 是三对角拉普拉斯算子 (1, -2, 1)。
    BC: Neumann at z=0 (T_{-1} = T_1), Dirichlet at z=L (T_L = 0)

    参数:
        kappa_array — κ(z) 在 z_array 上的取值, shape [Nz]
        z_array     — 空间网格, shape [Nz], 单位 m
        t_array     — 时间网格, shape [Nt], 单位 s
        I0          — 入射光强, W/m²
        D           — 热扩散率, m²/s
        rho         — 密度, kg/m³
        Cp          — 比热, J/(kg·K)

    返回:
        z_array, t_array, T_field — T_field shape [Nz, Nt]
    """
    kappa_array = np.asarray(kappa_array, dtype=np.float64)
    z_array = np.asarray(z_array, dtype=np.float64)
    t_array = np.asarray(t_array, dtype=np.float64)

    Nz = len(z_array)
    Nt = len(t_array)
    dz = z_array[1] - z_array[0]
    dt = t_array[1] - t_array[0]

    # CFL 数（仅信息用，隐式格式不需要满足）
    cfl = D * dt / (dz * dz)

    # 热源 q(z) / (ρCp)
    q_array = compute_source_array(z_array, kappa_array, I0)
    source_term = q_array / (rho * Cp)

    # 构建三对角算子 A 的 banded 表示
    # solve_banded 期望 ab[l+r-i+j, i+j] = a_{i,j}, 这里 l=u=1
    # ab shape: [3, Nz], 三行分别是上对角、主对角、下对角
    ab = np.zeros((3, Nz))
    # 主对角: -2 - 1/diffusion_factor，配合 -1/df 来抵消 (I - df·A)
    # 等价于: 主对角 = 1 + 2·D·dt/dz²，上下对角 = -D·dt/dz²
    diag_main = 1.0 + 2.0 * cfl * np.ones(Nz)
    diag_lower = -cfl * np.ones(Nz)
    diag_upper = -cfl * np.ones(Nz)

    # Neumann BC at z=0: T_{-1} = T_1 → 主对角第一行 = 1 (无变化), 第二行 = -2cfl (合并 -1-1)
    # 改写第一行方程: T0^{n+1} = T0^n + dt·q0/(ρCp) + 2·D·dt/dz²·(T1^{n+1} - T0^{n+1})
    # → -2cfl·T1 + (1+2cfl)·T0 = T0^n + dt·q0/(ρCp)
    # 这个形式已经和内部点一致（Neumann BC 仅影响"虚拟点" T_{-1}=T_1，二阶差分变成 T1 - 2T0 + T_{-1} = 2(T1-T0)）
    # 所以第一行主对角改为 1 + 2·cfl - cfl·0 ... 重新推导
    # 标准内部点: (T_{i-1} - 2T_i + T_{i+1})/dz²
    # Neumann at i=0 用 ghost cell T_{-1}=T_1: (T_1 - 2T_0 + T_1)/dz² = 2(T_1-T_0)/dz²
    # 所以第一行: 上对角(连到 T1)系数 = -2·cfl (而非 -cfl)
    diag_upper[0] = -2.0 * cfl   # 连到 T_1 的系数加倍

    # Dirichlet BC at z=L (i=Nz-1): T_L = 0
    # 最后一行直接 T_{Nz-1}^{n+1} = 0，但更干净的做法是让 last index 取固定值
    # 这里采用: 最后一行主对角=1, 上下对角=0, 右端=0
    diag_main[-1] = 1.0
    diag_upper[-1] = 0.0      # 连到 ghost (不存在)，置 0
    diag_lower[-1] = 0.0      # 连到 T_{Nz-2}，置 0 (因为 T_{Nz-1} 固定)

    # solve_banded 的 ab 矩阵: ab[0] = 上对角 (索引 0..Nz-2), ab[1] = 主对角, ab[2] = 下对角 (索引 1..Nz-1)
    ab[0, 1:] = diag_upper[:-1]   # 上对角 ab[0, j] = a_{j-1, j}
    ab[1, :] = diag_main
    ab[2, :-1] = diag_lower[1:]   # 下对角 ab[2, j] = a_{j+1, j}

    # 时间推进
    T_field = np.zeros((Nz, Nt))
    T = np.zeros(Nz)  # 初始 T(z, 0) = 0

    rhs_factor = dt  # source_term 已经是 q/(ρCp)，再乘 dt

    for n in range(Nt - 1):
        # 右端: T^n + dt · q/(ρCp)
        rhs = T + rhs_factor * source_term
        # Dirichlet BC: T_{Nz-1}^{n+1} = 0
        rhs[-1] = 0.0

        T = solve_banded((1, 1), ab, rhs)
        T_field[:, n + 1] = T

    return z_array, t_array, T_field


# ===== 默认网格（与 spec 一致）=====

def make_default_grid():
    """默认网格: z ∈ [0, 200μm], Nz=2000; t ∈ [0, 0.1s], Nt=1000。

    z 域取 200μm >> 扩散长度 @t=0.1s (≈109μm)，确保 Dirichlet BC @z=L
    不污染缺陷层附近 (z<15μm) 的解。
    """
    z_array = np.linspace(0.0, 200e-6, 2000)  # 0~200μm, Δz = 0.1μm
    t_array = np.linspace(0.0, 0.1, 1000)     # 0~0.1s, Δt = 1e-4s
    return z_array, t_array


def generate_ground_truth(I0=1e5, seed=None):
    """生成 PINN 案例的真值 T(z,t) 场。

    参数:
        I0   — 入射光强 (W/m²)，默认 1e5 (10 W/cm²，典型激光功率密度)
        seed — 随机种子（用于观测采样，这里不噪声）

    返回:
        dict with:
            z_array, t_array, T_field  — 真值场
            kappa_array                — 真值 κ(z)
            I_array                    — 真值光强 I(z)
            q_array                    — 真值热源 q(z)
            params                     — 参数字典
    """
    z_array, t_array = make_default_grid()
    kappa_array = kappa_true_func(z_array)
    I_array = compute_intensity_array(z_array, kappa_array, I0)
    q_array = compute_source_array(z_array, kappa_array, I0)

    _, _, T_field = solve_implicit_fd(
        kappa_array, z_array, t_array, I0,
        D=D_POLY, rho=RHO_POLY, Cp=CP_POLY,
    )

    return {
        'z_array': z_array,
        't_array': t_array,
        'T_field': T_field,
        'kappa_array': kappa_array,
        'I_array': I_array,
        'q_array': q_array,
        'params': {
            'n': N_POLY, 'kappa0': KAPPA0, 'wavelength_nm': WAVELENGTH_NM,
            'rho': RHO_POLY, 'Cp': CP_POLY, 'k_th': K_TH_POLY, 'D': D_POLY,
            'I0': I0,
            'defect_amp': DEFECT_AMP, 'defect_z0': DEFECT_Z0, 'defect_sigma': DEFECT_SIGMA,
        }
    }


def sample_observations(gt, z_obs_um, t_obs_s, noise=0.0, seed=42):
    """从真值场采样稀疏观测。

    参数:
        gt          — generate_ground_truth 返回的 dict
        z_obs_um    — 观测点 z 坐标 (μm), array
        t_obs_s     — 观测点 t 坐标 (s), array
        noise       — 乘性高斯噪声标准差（如 0.01 = 1%）
        seed        — 随机种子

    返回:
        z_obs, t_obs, T_obs — 都是 1D arrays, 长度 = len(z_obs_um) × len(t_obs_s)
        (z_obs, t_obs 是笛卡尔积展开后对应的坐标)
    """
    rng = np.random.default_rng(seed)
    z_array = gt['z_array']
    t_array = gt['t_array']
    T_field = gt['T_field']

    z_obs_m = np.asarray(z_obs_um) * 1e-6  # μm → m
    t_obs = np.asarray(t_obs_s)

    # 在网格上插值 (双线性)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((z_array, t_array), T_field, bounds_error=False, fill_value=0.0)

    Z, T = np.meshgrid(z_obs_m, t_obs, indexing='ij')
    pts = np.stack([Z.ravel(), T.ravel()], axis=-1)
    T_obs = interp(pts)

    if noise > 0:
        T_obs = T_obs * (1.0 + noise * rng.standard_normal(T_obs.shape))

    # 返回笛卡尔积展开后的 1D 坐标（保持一致性）
    return Z.ravel(), T.ravel(), T_obs
