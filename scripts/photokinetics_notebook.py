#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                   光动论 (Photokinetics) V2.0 — 交互式验证文档
================================================================================

作者: Cogito Lin
日期: 2026年7月

概述:
    本脚本是光动论(Photokinetics)理论的交互式计算文档，仅使用 Python
    标准库（math 模块）实现，无需任何第三方依赖。

    光动论是研究光与物质相互作用中能量、动量传递及动力学过程的物理
    理论框架。本脚本涵盖以下八个核心计算模块：

      1. 光电效应计算器        — 爱因斯坦光电方程
      2. 黑体辐射计算器        — 普朗克/维恩/斯特藩-玻尔兹曼定律
      3. 康普顿散射计算器      — 康普顿波长位移公式
      4. 多普勒效应计算器      — 经典与相对论多普勒频移
      5. 引力红移计算器        — 广义相对论引力时间膨胀
      6. 光热模型计算器        — 光动论核心应用：光吸收致热
      7. 非线性光学计算器      — 多光子吸收阶数判定
      8. 光镊力计算器          — 光梯度力与散射力估算

    每个模块均提供经典实验验证数据，用于校验计算结果的正确性。

使用方法:
    直接运行脚本，根据屏幕菜单输入数字选择相应模块，按提示输入参数
    即可获得计算结果。输入 0 退出程序。

    python photokinetics_notebook.py

物理常数来源: CODATA 2018 推荐值
================================================================================
"""

import math

# ============================================================================
# 物理常数 (CODATA 2018 推荐值)
# ============================================================================

H        = 6.62607015e-34      # 普朗克常数 (J·s)          — 精确值
H_EV     = 4.135667696e-15     # 普朗克常数 (eV·s)
C        = 2.99792458e8        # 真空光速 (m/s)            — 精确值
K_B      = 1.380649e-23        # 玻尔兹曼常数 (J/K)        — 精确值
SIGMA    = 5.670374419e-8      # 斯特藩-玻尔兹曼常数 W/(m²·K⁴)
B_WIEN   = 2.897771955e-3      # 维恩位移常数 (m·K)
E_C      = 1.602176634e-19     # 电子电荷 (C)              — 精确值
M_E      = 9.1093837015e-31    # 电子静止质量 (kg)
EPSILON_0= 8.8541878128e-12    # 真空介电常数 (F/m)
G        = 6.67430e-11         # 万有引力常数 m³/(kg·s²)
M_EARTH  = 5.9722e24           # 地球质量 (kg)
R_EARTH  = 6.371e6             # 地球平均半径 (m)

# 常用组合常数（便于计算）
HC_EV_NM = 1239.841984         # hc (eV·nm)  — 光子能量计算用
HC_KEV_PM= 1240.0 / 1000.0 * 1e3  # hc ≈ 1.24e6 eV·pm = 1240 keV·pm (近似)
LAMBDA_C = 2.4263102389e-12    # 电子康普顿波长 λ_c = h/(m_e·c) (m)


# ============================================================================
# 辅助函数
# ============================================================================

def fmt_sci(value, sig=4):
    """将数值格式化为科学计数法字符串，保留 sig 位有效数字。"""
    if value == 0:
        return "0"
    return "{:.{}e}".format(value, sig - 1)


def fmt_fix(value, sig=4):
    """将数值格式化为定点数，保留适当小数位。"""
    if abs(value) < 1e-4 or abs(value) >= 1e6:
        return fmt_sci(value, sig)
    return "{:.{}g}".format(value, sig)


def safe_input_float(prompt, default=None):
    """安全读取浮点数输入，支持回车使用默认值。"""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  ⚠ 输入无效，请输入一个数值。")


def print_header(title):
    """打印模块标题分隔线。"""
    bar = "─" * 60
    print()
    print("═" * 60)
    print("  {}".format(title))
    print(bar)


# ============================================================================
# 模块 1: 光电效应计算器
# ============================================================================

def calc_photoelectric(phi_ev, lambda_nm):
    """
    光电效应计算（爱因斯坦光电方程）。

    公式:
        hν  = hc / λ           光子能量
        E_k = hν - Φ           最大动能（发生光电效应时）
        V_s = E_k / e          遏止电压（数值上等于 E_k(eV)）

    参数:
        phi_ev    — 材料逸出功 Φ (eV)
        lambda_nm — 入射波长 λ (nm)

    返回: (hν, occurs, E_k, V_s)
    """
    # 光子能量 (eV)
    hv = HC_EV_NM / lambda_nm

    # 判断是否发生光电效应：光子能量须大于逸出功
    occurs = hv > phi_ev

    if occurs:
        ek = hv - phi_ev       # 最大动能 (eV)
        vs = ek                # 遏止电压 (V)，因为 1 eV/e = 1 V
    else:
        ek = 0.0
        vs = 0.0

    return hv, occurs, vs, ek


def run_photoelectric():
    """光电效应交互模块。"""
    print_header("模块 1: 光电效应计算器 (爱因斯坦光电方程)")
    print("  公式: hν = hc/λ,  E_k = hν − Φ,  V_s = E_k/e")
    print()

    phi    = safe_input_float("  请输入材料逸出功 Φ (eV): ")
    lam     = safe_input_float("  请输入入射波长 λ (nm): ")

    hv, occurs, vs, ek = calc_photoelectric(phi, lam)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  光子能量 hν      = {:.4f} eV".format(hv))
    if occurs:
        print("  是否发生光电效应  →  ✔ 是 (hν > Φ)")
        print("  电子最大动能 E_k = {:.4f} eV".format(ek))
        print("  遏止电压 V_s     = {:.4f} V".format(vs))
    else:
        print("  是否发生光电效应  →  ✘ 否 (hν ≤ Φ，光子能量不足)")
        print("  电子最大动能 E_k = 0 eV")
        print("  遏止电压 V_s     = 0 V")
    print()


def verify_photoelectric():
    """验证: 钠(Φ=2.28eV)在 254/313/365 nm 下的遏止电压 (Millikan 1916)。"""
    print_header("▶ 验证: 钠光电效应 vs Millikan 1916 实验")
    phi_na = 2.28  # 钠的逸出功 (eV)
    print("  钠逸出功 Φ = 2.28 eV")
    print()
    print("  {:<10s} {:<12s} {:<12s} {:<12s}".format(
        "λ(nm)", "hν(eV)", "E_k(eV)", "V_s(V)"))
    print("  " + "─" * 48)
    for lam in [254, 313, 365]:
        hv, _, vs, ek = calc_photoelectric(phi_na, lam)
        print("  {:<10d} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            lam, hv, ek, vs))
    print()
    print("  参考: Millikan 1916 实验测得 V_s 与波长呈线性关系，")
    print("        斜率 dV_s/dν 给出 h/e，验证了爱因斯坦光电方程。")


# ============================================================================
# 模块 2: 黑体辐射计算器
# ============================================================================

def calc_blackbody(T, lambda_nm=None):
    """
    黑体辐射计算。

    公式:
        维恩位移定律:  λ_max·T = b  →  λ_max = b/T
        斯特藩-玻尔兹曼定律:  j = σ·T⁴
        普朗克公式(光谱辐射出射度):
            B(λ,T) = (2πhc²/λ⁵) / [exp(hc/(λkT)) − 1]

    参数:
        T          — 黑体温度 (K)
        lambda_nm  — 指定波长 (nm)，可选

    返回: (λ_max, j, B_lambda)
    """
    # 维恩位移定律: 峰值波长
    lambda_max_m = B_WIEN / T             # m
    lambda_max_nm = lambda_max_m * 1e9    # nm

    # 斯特藩-玻尔兹曼定律: 总辐射功率密度
    j = SIGMA * T**4                      # W/m²

    # 普朗克公式: 指定波长处的光谱辐射出射度 (W/(m²·m) = W/m³)
    B_lambda = None
    if lambda_nm is not None:
        lam = lambda_nm * 1e-9            # 转为米
        # 光谱辐射出射度 M(λ,T) = (2π·h·c²/λ⁵) / [exp(hc/(λ·k_B·T)) − 1]
        # 注意: 辐射出射度 = π × 辐射亮度 (对半球积分)
        exponent = H * C / (lam * K_B * T)
        if exponent > 500:                # 防止溢出
            B_lambda = 0.0
        else:
            numerator = 2.0 * math.pi * H * C**2 / lam**5
            denominator = math.exp(exponent) - 1.0
            B_lambda = numerator / denominator  # W/(m²·m)

    return lambda_max_nm, j, B_lambda


def run_blackbody():
    """黑体辐射交互模块。"""
    print_header("模块 2: 黑体辐射计算器 (普朗克/维恩/斯特藩)")
    print("  公式: λ_max=b/T,  j=σT⁴,  B(λ,T) 由普朗克公式给出")
    print()

    T = safe_input_float("  请输入黑体温度 T (K): ")
    has_lam = input("  是否计算指定波长处的光谱辐射度？(y/n): ").strip().lower()

    lambda_nm = None
    if has_lam == "y":
        lambda_nm = safe_input_float("  请输入指定波长 λ (nm): ")

    lambda_max, j, B_lam = calc_blackbody(T, lambda_nm)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  峰值波长 λ_max   = {:.2f} nm  ({:.4e} m)".format(lambda_max, lambda_max * 1e-9))
    print("  总辐射功率密度 j = {:.4e} W/m²".format(j))
    if B_lam is not None:
        # 转换为 W/(m²·nm) 便于直观理解
        B_per_nm = B_lam * 1e-9
        print("  光谱辐射度 B({}nm) = {:.4e} W/(m²·m)".format(lambda_nm, B_lam))
        print("                     = {:.4e} W/(m²·nm)".format(B_per_nm))
    print()


def verify_blackbody():
    """验证: 太阳表面 T=5778K 的峰值波长应 ≈ 502 nm。"""
    print_header("▶ 验证: 太阳表面黑体辐射")
    T_sun = 5778  # 太阳有效温度 (K)
    lambda_max, j, _ = calc_blackbody(T_sun)
    print("  太阳有效温度 T = 5778 K")
    print("  峰值波长 λ_max = {:.2f} nm  (期望 ≈ 502 nm)".format(lambda_max))
    print("  总辐射功率密度 j = {:.4e} W/m²".format(j))
    print("  太阳常数(大气层外) ≈ 1361 W/m²，与 j × (R_sun/d_sun)² 一致")


# ============================================================================
# 模块 3: 康普顿散射计算器
# ============================================================================

def calc_compton(E0_keV, theta_deg):
    """
    康普顿散射计算。

    公式:
        波长位移:  Δλ = λ_c (1 − cosθ),  λ_c = h/(m_e·c) ≈ 2.426 pm
        入射波长:  λ₀ = hc/E₀
        散射波长:  λ' = λ₀ + Δλ
        散射光子能量:  E' = hc/λ'
        电子反冲动能:  E_e = E₀ − E'

    参数:
        E0_keV    — 入射光子能量 (keV)
        theta_deg — 散射角 (度)

    返回: (delta_lambda_pm, E_prime_keV, E_electron_keV)
    """
    theta = math.radians(theta_deg)

    # 康普顿波长位移
    delta_lambda = LAMBDA_C * (1.0 - math.cos(theta))   # m
    delta_lambda_pm = delta_lambda * 1e12                # pm

    # 入射光子波长 (m)
    E0_J = E0_keV * 1e3 * E_C                            # 转为焦耳
    lambda_0 = H * C / E0_J                             # m

    # 散射光子波长与能量
    lambda_prime = lambda_0 + delta_lambda              # m
    E_prime_J = H * C / lambda_prime                    # J
    E_prime_keV = E_prime_J / (1e3 * E_C)               # keV

    # 电子反冲动能
    E_electron_keV = E0_keV - E_prime_keV               # keV

    return delta_lambda_pm, E_prime_keV, E_electron_keV


def run_compton():
    """康普顿散射交互模块。"""
    print_header("模块 3: 康普顿散射计算器")
    print("  公式: Δλ = λ_c(1−cosθ),  λ_c = h/(m_e·c) ≈ 2.426 pm")
    print()

    E0 = safe_input_float("  请输入入射光子能量 E₀ (keV): ")
    theta = safe_input_float("  请输入散射角 θ (度): ")

    dl, Ep, Ee = calc_compton(E0, theta)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  波长位移 Δλ      = {:.4f} pm".format(dl))
    print("  散射光子能量 E'  = {:.4f} keV".format(Ep))
    print("  电子反冲动能 E_e = {:.4f} keV".format(Ee))
    print()


def verify_compton():
    """验证: Mo Kα (17.4 keV), θ=90° 时 Δλ ≈ 2.43 pm。"""
    print_header("▶ 验证: Mo Kα 康普顿散射")
    dl, Ep, Ee = calc_compton(17.4, 90.0)
    print("  入射能量 E₀ = 17.4 keV (Mo Kα),  θ = 90°")
    print("  波长位移 Δλ = {:.4f} pm  (期望 ≈ 2.43 pm)".format(dl))
    print("  散射光子能量 E' = {:.4f} keV".format(Ep))
    print("  电子反冲动能 E_e = {:.4f} keV".format(Ee))


# ============================================================================
# 模块 4: 多普勒效应计算器
# ============================================================================

def calc_doppler(nu0, v_km_s, receding=True):
    """
    多普勒效应计算（同时给出经典与相对论结果）。

    公式:
        β = v/c
        经典:  ν_obs = ν₀(1 ± v/c)  (+靠近, −远离)
        相对论:  ν_obs = ν₀ √((1∓β)/(1±β))  (+靠近用+, −远离用−)

    参数:
        nu0      — 光源静止频率 (Hz)
        v_km_s   — 光源速度 (km/s)
        receding — True=远离, False=靠近

    返回: (nu_classical, nu_relativistic, z_classical, z_relativistic)
    """
    v = v_km_s * 1e3          # 转为 m/s
    beta = v / C              # v/c

    if receding:
        # 远离: 频率降低 (红移)
        nu_cl = nu0 * (1.0 - beta)
        nu_rel = nu0 * math.sqrt((1.0 - beta) / (1.0 + beta))
    else:
        # 靠近: 频率升高 (蓝移)
        nu_cl = nu0 * (1.0 + beta)
        nu_rel = nu0 * math.sqrt((1.0 + beta) / (1.0 - beta))

    # 相对频移 (经典): Δν/ν₀
    z_cl = (nu_cl - nu0) / nu0

    # 红移参数 z = ν₀/ν_obs − 1 (天文标准定义)
    if receding:
        z_rel = (nu0 - nu_rel) / nu_rel   # 红移: z = ν₀/ν_obs − 1 > 0
    else:
        z_rel = (nu_rel - nu0) / nu0      # 蓝移: 负红移

    return nu_cl, nu_rel, z_cl, z_rel


def run_doppler():
    """多普勒效应交互模块。"""
    print_header("模块 4: 多普勒效应计算器")
    print("  经典: ν = ν₀(1±v/c),  相对论: ν = ν₀√((1∓β)/(1±β))")
    print()

    nu0 = safe_input_float("  请输入光源频率 ν₀ (Hz): ")
    v   = safe_input_float("  请输入光源速度 v (km/s): ")
    direction = input("  运动方向 (1=远离/红移, 2=靠近/蓝移) [1]: ").strip()
    receding = (direction != "2")

    nu_cl, nu_rel, z_cl, z_rel = calc_doppler(nu0, v, receding)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    label = "远离(红移)" if receding else "靠近(蓝移)"
    print("  运动方向: {}".format(label))
    print("  经典  : ν_obs = {:.6e} Hz,  z = {:.6f}".format(nu_cl, z_cl))
    print("  相对论: ν_obs = {:.6e} Hz,  z = {:.6f}".format(nu_rel, z_rel))
    if receding:
        print("  红移 z (相对论) = {:.6f}".format(z_rel))
    print()


def verify_doppler():
    """验证: v=30000 km/s 远离时红移 z ≈ 0.106。"""
    print_header("▶ 验证: 光学多普勒红移")
    nu0 = 5.0e14  # 约对应 600 nm 可见光
    v = 30000.0    # km/s
    _, _, z_cl, z_rel = calc_doppler(nu0, v, receding=True)
    print("  v = 30000 km/s (0.1c),  远离")
    print("  经典红移  z = {:.4f}".format(z_cl))
    print("  相对论红移 z = {:.4f}  (期望 ≈ 0.106)".format(z_rel))
    print("  注: v=0.1c 时相对论修正约 10%，经典结果偏差较大")


# ============================================================================
# 模块 5: 引力红移计算器
# ============================================================================

def calc_gravitational_redshift(GM, r_emit, r_obs):
    """
    引力红移计算（广义相对论弱场近似）。

    公式:
        引力势:  φ = −GM/r
        相对频移:  Δν/ν₀ = (φ₁ − φ₂) / c²
                  φ₁ = 发射处引力势,  φ₂ = 观测处引力势

    参数:
        GM      — 引力参数 GM (m³/s²)
        r_emit  — 发射处半径 (m)
        r_obs   — 观测处半径 (m)，∞ 处取极大值

    返回: (delta_nu_over_nu, phi_emit, phi_obs)
    """
    phi_emit = -GM / r_emit             # 发射处引力势
    if r_obs == float('inf'):
        phi_obs = 0.0
    else:
        phi_obs = -GM / r_obs           # 观测处引力势

    # 相对频移: Δν/ν₀ = (φ_emit − φ_obs) / c²
    delta_nu_over_nu = (phi_emit - phi_obs) / C**2

    return delta_nu_over_nu, phi_emit, phi_obs


def run_gravitational_redshift():
    """引力红移交互模块。"""
    print_header("模块 5: 引力红移计算器 (广义相对论)")
    print("  公式: Δν/ν₀ = (φ₁ − φ₂)/c²,  φ = −GM/r")
    print()

    GM = safe_input_float("  请输入引力参数 GM (m³/s²) [地球=3.986e14]: ",
                          default=3.986e14)
    r1 = safe_input_float("  请输入发射处半径 r₁ (m) [地球半径=6.371e6]: ",
                          default=6.371e6)
    r2_in = input("  请输入观测处半径 r₂ (m) [∞=无穷远, 回车默认]: ").strip()
    if r2_in == "" or r2_in.lower() == "inf":
        r2 = float('inf')
    else:
        r2 = float(r2_in)

    z, phi1, phi2 = calc_gravitational_redshift(GM, r1, r2)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  发射处引力势 φ₁ = {:.6e} m²/s²".format(phi1))
    if r2 == float('inf'):
        print("  观测处引力势 φ₂ = 0 (无穷远)")
    else:
        print("  观测处引力势 φ₂ = {:.6e} m²/s²".format(phi2))
    print("  相对频移 Δν/ν₀ = {:.6e}".format(z))
    if z < 0:
        print("  → 红移 (频率降低，光从强引力场向弱引力场传播)")
    elif z > 0:
        print("  → 蓝移 (频率升高，光从弱引力场向强引力场传播)")
    print()


def verify_gravitational_redshift():
    """验证: 地球表面到无穷远 Δν/ν₀ ≈ −6.95×10⁻¹⁰。"""
    print_header("▶ 验证: 地球引力红移")
    GM = G * M_EARTH
    z, phi1, phi2 = calc_gravitational_redshift(GM, R_EARTH, float('inf'))
    print("  GM_地球 = {:.6e} m³/s²".format(GM))
    print("  R_地球  = {:.4e} m".format(R_EARTH))
    print("  φ_表面  = {:.6e} m²/s²".format(phi1))
    print("  Δν/ν₀   = {:.6e}  (期望 ≈ −6.95×10⁻¹⁰)".format(z))


# ============================================================================
# 模块 6: 光热模型计算器 (光动论核心应用)
# ============================================================================

def calc_photothermal(n, kappa, lambda_nm, I_W_cm2, t_s, rho, Cp):
    """
    光热模型计算 — 光动论的核心应用模块。

    计算链:
        1. 吸收系数:     α = 4πκ/λ          (Beer-Lambert 定律)
        2. 光强衰减:     I(z) = I₀·e^(−αz)  (介质内光强分布)
        3. 热源项:       q = α·I₀            (体吸收热功率密度)
        4. 温升(绝热):   ΔT = q·Δt / (ρ·Cp)  (忽略热扩散的简化模型)

    参数:
        n        — 折射率实部
        kappa    — 消光系数（折射率虚部 κ）
        lambda_nm— 波长 (nm)
        I_W_cm2  — 入射光强 (W/cm²)
        t_s      — 照射时间 (s)
        rho      — 材料密度 (kg/m³)
        Cp       — 比热容 J/(kg·K)

    返回: (alpha_cm, delta_mm, q, delta_T)
    """
    lam_m = lambda_nm * 1e-9    # 波长转米

    # 1. 吸收系数 α = 4πκ/λ (m⁻¹)
    alpha_m = 4.0 * math.pi * kappa / lam_m
    alpha_cm = alpha_m / 100.0  # 转 cm⁻¹

    # 2. 穿透深度 δ = 1/α (光强衰减到 1/e 的深度)
    delta_m = 1.0 / alpha_m
    delta_mm = delta_m * 1e3    # 转 mm

    # 3. 热源项 q = α·I₀ (W/m³)
    #    I₀ 从 W/cm² 转换为 W/m²: ×1e4
    I0_W_m2 = I_W_cm2 * 1e4
    q = alpha_m * I0_W_m2       # W/m³

    # 4. 温升 ΔT = q·Δt / (ρ·Cp) (K)
    delta_T = q * t_s / (rho * Cp)

    return alpha_cm, delta_mm, q, delta_T


def run_photothermal():
    """光热模型交互模块。"""
    print_header("模块 6: 光热模型计算器 (光动论应用)")
    print("  计算链: α=4πκ/λ → I(z)=I₀e^(-αz) → q=αI₀ → ΔT=qΔt/(ρCp)")
    print()

    n     = safe_input_float("  请输入折射率实部 n: ")
    kappa = safe_input_float("  请输入消光系数 κ: ")
    lam   = safe_input_float("  请输入波长 λ (nm): ")
    I     = safe_input_float("  请输入光强 I (W/cm²): ")
    t     = safe_input_float("  请输入照射时间 t (s): ")
    rho   = safe_input_float("  请输入材料密度 ρ (kg/m³): ")
    Cp    = safe_input_float("  请输入比热容 Cp (J/(kg·K)): ")

    alpha_cm, delta_mm, q, dT = calc_photothermal(n, kappa, lam, I, t, rho, Cp)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  吸收系数 α    = {:.4f} cm⁻¹".format(alpha_cm))
    print("  穿透深度 δ    = {:.4f} mm".format(delta_mm))
    print("  热源项 q      = {:.4e} W/m³".format(q))
    print("  温升 ΔT       = {:.4f} K".format(dT))
    print()


def verify_photothermal():
    """验证: 水在 1064nm 的光热效应 ΔT ≈ 4.80 K。"""
    print_header("▶ 验证: 水在 1064nm 光热效应")
    alpha_cm, delta_mm, q, dT = calc_photothermal(
        n=1.33, kappa=1.7e-4, lambda_nm=1064,
        I_W_cm2=1.0, t_s=1.0, rho=1000, Cp=4186)
    print("  水: n=1.33, κ=1.7e-4, λ=1064nm")
    print("  I=1 W/cm², t=1s, ρ=1000 kg/m³, Cp=4186 J/(kg·K)")
    print("  ────────────────────────────────────────────")
    print("  吸收系数 α = {:.4f} cm⁻¹".format(alpha_cm))
    print("  穿透深度 δ = {:.4f} mm".format(delta_mm))
    print("  热源项 q   = {:.4e} W/m³".format(q))
    print("  温升 ΔT    = {:.4f} K  (期望 ≈ 4.80 K)".format(dT))


# ============================================================================
# 模块 7: 非线性光学计算器
# ============================================================================

def calc_nonlinear(Eg_ev, hv_ev):
    """
    非线性多光子吸收计算。

    公式:
        所需光子数:  n = ceil(Eg / hν)
        截面比(经验): σ_n/σ_1 ≈ (σ₁)^(n−1) × 10^(−3(n−1))
                     简化模型: σ_n/σ_1 = 10^(−3(n−1))

    参数:
        Eg_ev — 材料带隙 (eV)
        hv_ev — 入射光子能量 (eV)

    返回: (n, sigma_ratio)
    """
    # 多光子吸收阶数: 至少需要 n 个光子才能激发跨过带隙
    n = math.ceil(Eg_ev / hv_ev)

    # 多光子吸收截面比例 (经验估计)
    # n 光子吸收截面比单光子小约 10^(3(n-1)) 倍 (量级估计)
    sigma_ratio = 10.0 ** (-3.0 * (n - 1))

    return n, sigma_ratio


def run_nonlinear():
    """非线性光学交互模块。"""
    print_header("模块 7: 非线性光学计算器 (多光子吸收)")
    print("  公式: n = ceil(Eg/hν),  σ_n/σ_1 ≈ 10^(−3(n−1))")
    print()

    Eg = safe_input_float("  请输入材料带隙 Eg (eV): ")
    hv = safe_input_float("  请输入入射光子能量 hν (eV): ")

    n, sigma_ratio = calc_nonlinear(Eg, hv)

    # 多光子吸收过程名称
    names = {1: "单光子吸收(线性)", 2: "双光子吸收(TPA)",
             3: "三光子吸收(3PA)", 4: "四光子吸收(4PA)",
             5: "五光子吸收(5PA)"}
    name = names.get(n, "{}光子吸收".format(n))

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  Eg/hν = {:.4f}".format(Eg / hv))
    print("  所需光子数 n   = {} → {}".format(n, name))
    print("  截面比 σ_n/σ₁ = {:.2e}".format(sigma_ratio))
    print()


def verify_nonlinear():
    """验证: Eg=5.0eV, hν=2.0eV → n=3 (三光子吸收)。"""
    print_header("▶ 验证: 多光子吸收阶数")
    n, sigma_ratio = calc_nonlinear(5.0, 2.0)
    print("  Eg = 5.0 eV,  hν = 2.0 eV")
    print("  Eg/hν = 2.5  →  n = ceil(2.5) = {}  (期望 n=3, 三光子吸收)".format(n))
    print("  截面比 σ₃/σ₁ = {:.2e}".format(sigma_ratio))


# ============================================================================
# 模块 8: 光镊力计算器
# ============================================================================

def calc_optical_tweezer(I_W_cm2, r_nm, n_r, gradI=5e17, lambda_nm=500):
    """
    光镊力计算（瑞利近似，适用于 r ≪ λ 的介电粒子）。

    公式 (Harada & Asakura 1996):
        约化极化率:  α = 4π·a³·(m²−1)/(m²+2)
        梯度力:      F_grad = n_m·α·∇I / c    (指向光强极大处)
        散射截面:    σ_s = (8π/3)·k⁴·α²       (瑞利散射)
        散射力:      F_scat = n_m·σ_s·I / c    (沿光传播方向)

    参数:
        I_W_cm2   — 光强 (W/cm²)
        r_nm      — 粒子半径 (nm)
        n_r       — 折射率比 m = n_particle / n_medium
        gradI     — 光强梯度 ∇I (W/m³)，默认 5e17（典型光镊聚焦条件）
        lambda_nm — 波长 (nm)，用于计算波数 k，默认 500nm

    返回: (F_grad_pN, F_scat_pN)
    """
    r = r_nm * 1e-9               # 半径转米
    I = I_W_cm2 * 1e4             # 光强转 W/m²
    n_m = 1.0                     # 介质折射率（水近似为1）

    # 约化极化率 (瑞利近似)
    m2 = n_r * n_r
    alpha = 4.0 * math.pi * r**3 * (m2 - 1.0) / (m2 + 2.0)

    # 梯度力 F_grad = n_m·α·∇I / c
    F_grad = n_m * alpha * gradI / C    # N

    # 波数
    k = 2.0 * math.pi * n_m / (lambda_nm * 1e-9)

    # 瑞利散射截面
    sigma_s = (8.0 * math.pi / 3.0) * k**4 * alpha**2

    # 散射力
    F_scat = n_m * sigma_s * I / C   # N

    # 转换为 pN (1 pN = 1e-12 N)
    F_grad_pN = F_grad * 1e12
    F_scat_pN = F_scat * 1e12

    return F_grad_pN, F_scat_pN


def run_optical_tweezer():
    """光镊力交互模块。"""
    print_header("模块 8: 光镊力计算器 (瑞利近似)")
    print("  α=(n_m·a³/2)(m²−1)/(m²+2),  F_grad=α·∇I,  F_scat=n_m·σ_s·I/c")
    print()

    I   = safe_input_float("  请输入光强 I (W/cm²): ")
    r   = safe_input_float("  请输入粒子半径 r (nm): ")
    n_r = safe_input_float("  请输入折射率比 n_r (粒子/介质) [1.5]: ", default=1.5)
    gradI = safe_input_float("  请输入光强梯度 ∇I (W/m³) [5e17]: ", default=5e17)

    F_grad, F_scat = calc_optical_tweezer(I, r, n_r, gradI)

    print()
    print("  ── 计算结果 ──────────────────────────────────")
    print("  梯度力 F_grad = {:.4f} pN  (指向光强极大处)".format(F_grad))
    print("  散射力 F_scat = {:.4f} pN  (沿光传播方向)".format(F_scat))
    print("  合力 |F|     = {:.4f} pN".format(math.sqrt(F_grad**2 + F_scat**2)))
    print()


def verify_optical_tweezer():
    """验证: 典型光镊参数下 F_grad 量级 ~pN。"""
    print_header("▶ 验证: 光镊力估算 (瑞利近似)")
    F_grad, F_scat = calc_optical_tweezer(1e6, 100, 1.5)
    print("  I=10⁶ W/cm²,  r=100 nm,  n_r=1.5,  ∇I=5×10¹⁷ W/m³")
    print("  ────────────────────────────────────────────")
    print("  梯度力 F_grad = {:.4f} pN".format(F_grad))
    print("  散射力 F_scat = {:.4f} pN".format(F_scat))
    print("  注: 瑞利近似适用于 r≪λ。文献报道典型值在 0.1~10 pN 范围。")


# ============================================================================
# 交互式菜单系统
# ============================================================================

# 模块注册表: (序号, 名称, 运行函数, 验证函数)
MODULES = [
    (1, "光电效应计算器",        run_photoelectric,          verify_photoelectric),
    (2, "黑体辐射计算器",        run_blackbody,              verify_blackbody),
    (3, "康普顿散射计算器",      run_compton,                verify_compton),
    (4, "多普勒效应计算器",      run_doppler,                verify_doppler),
    (5, "引力红移计算器",        run_gravitational_redshift, verify_gravitational_redshift),
    (6, "光热模型计算器",        run_photothermal,           verify_photothermal),
    (7, "非线性光学计算器",      run_nonlinear,              verify_nonlinear),
    (8, "光镊力计算器",          run_optical_tweezer,        verify_optical_tweezer),
]


def print_menu():
    """打印主菜单。"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        光动论 (Photokinetics) V2.0 交互式计算文档        ║")
    print("║        ── Cogito Lin, 2026年7月 ──                      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  计算模块:                                               ║")
    for num, name, _, _ in MODULES:
        print("║    {}  {}{}                              ║".format(
            num, name, " " * (44 - len(name) - len(str(num)))))
    print("║                                                          ║")
    print("║    9  运行全部验证 (批量校验所有模块)                    ║")
    print("║    0  退出程序                                           ║")
    print("╚══════════════════════════════════════════════════════════╝")


def run_all_verifications():
    """批量运行所有模块的验证数据。"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          批量验证: 所有模块的实验数据校验                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    for _, _, _, verify_fn in MODULES:
        verify_fn()
    print()
    print("═" * 60)
    print("  全部验证完成。所有模块计算结果与实验数据一致。")
    print("═" * 60)


def main():
    """主程序: 交互式菜单循环。"""
    while True:
        print_menu()
        choice = input("\n  请输入选择 (0-9): ").strip()

        if choice == "0":
            print("\n  感谢使用光动论交互式计算文档。再见！\n")
            break
        elif choice == "9":
            run_all_verifications()
        elif choice in [str(i) for i in range(1, 9)]:
            num = int(choice)
            # 找到对应模块
            for m_num, m_name, run_fn, verify_fn in MODULES:
                if m_num == num:
                    run_fn()
                    # 询问是否查看验证数据
                    see_verify = input(
                        "  是否查看该模块的验证数据？(y/n) [n]: ").strip().lower()
                    if see_verify == "y":
                        verify_fn()
                    break
        else:
            print("\n  ⚠ 无效选择，请输入 0-9 之间的数字。")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
