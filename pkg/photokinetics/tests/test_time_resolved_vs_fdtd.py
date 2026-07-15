"""含时热传导模块 vs FDTD 对比验证。

测试场景: 突破绝热近似 (t·D·α² > 0.01) 的中等脉冲场景，
此时绝热近似失效，必须用含时解。
"""
import torch
import numpy as np
import time

# 添加路径
import sys
sys.path.insert(0, '.')

from photokinetics import calc_photothermal_timed


def fdtd_photothermal(n, kappa, wavelength_nm, I0, rho, Cp, depth_m, time_s,
                      k_thermal, nz=500, dt=None, domain_factor=10):
    """
    1D-FDTD 数值求解含源热传导方程。
    方程: ∂T/∂t = D·∂²T/∂z² + q(z)/(ρCp)
    边界: 两端绝热 (∂T/∂z = 0)
    域: [0, domain_factor·depth_m]，模拟半无限域
    """
    D = k_thermal / (rho * Cp)
    alpha = 4.0 * np.pi * kappa / (n * wavelength_nm * 1e-9)

    # 扩展计算域，模拟半无限域
    # 热扩散距离 ~ √(Dt)，域要远大于此
    diffusion_length = max(np.sqrt(D * time_s), depth_m)
    domain_length = max(domain_factor * depth_m, 5 * diffusion_length)

    # 空间步长
    dz = domain_length / nz
    # 时间步长（CFL条件: dt < dz²/(2D)）
    if dt is None:
        dt = 0.4 * dz * dz / (2.0 * D)
    nt = int(time_s / dt) + 1
    dt = time_s / nt  # 调整以精确匹配 time_s

    # 空间网格
    z = np.linspace(0, domain_length, nz + 1)
    # 热源分布 q(z) = α·I₀·exp(-αz)
    q = alpha * I0 * np.exp(-alpha * z) / (rho * Cp)

    # 初始温度
    T = np.zeros(nz + 1)

    # FDTD 迭代（显式 FTCS 格式）
    coeff = D * dt / (dz * dz)
    q_factor = dt  # q 已经除以 ρCp

    # 使用 numpy 向量化
    # 边界条件: 绝热 (∂T/∂z = 0)
    for _ in range(nt):
        T[1:-1] = T[1:-1] + coeff * (T[2:] - 2.0 * T[1:-1] + T[:-2]) + q_factor * q[1:-1]
        T[0] = T[1]
        T[-1] = T[-2]

    return z, T, nt


def test_water_1064nm_breaking_adiabatic():
    """测试1: 水@1064nm，中等脉冲，突破绝热近似。"""
    print("=" * 70)
    print("测试1: 水 @ 1064nm, 中等脉冲 (突破绝热近似)")
    print("=" * 70)

    # 参数
    n, kappa, lam = 1.33, 0.00012, 1064
    I0, rho, Cp = 1e7, 1000, 4186
    k_thermal = 0.598  # W/(m·K)
    D = k_thermal / (rho * Cp)

    alpha = 4.0 * np.pi * kappa / (n * lam * 1e-9)
    tau_thermal = 1.0 / (D * alpha * alpha)

    print(f"  α = {alpha:.2f} /m, D = {D:.4e} m²/s")
    print(f"  热扩散特征时间 τ = 1/(D·α²) = {tau_thermal:.4e} s")
    print()

    # 测试多个时间点（从绝热到稳态，避免太长FDTD跑不完）
    # 水的热扩散很慢，τ=1/(D·α²)很大，所以0.1s仍在过渡区
    # 用更短时间，让FDTD能跑完
    test_times = [1e-4, 1e-3, 1e-2, 5e-2]
    depth_mm = 0.1  # 0.1mm 深度

    print(f"  {'时间(s)':<12} {'t·D·α²':<12} {'regime':<14} {'绝热(K)':<12} {'含时(K)':<12} {'FDTD(K)':<12} {'误差%':<10}")
    print("  " + "-" * 90)

    for t in test_times:
        # 光动论含时解
        result = calc_photothermal_timed(
            n, kappa, lam, I0, rho, Cp, depth_mm, t, k_thermal=k_thermal
        )
        T_adiabatic = result['T_adiabatic'].item()
        T_timed = result['T_timed'].item()
        regime = result['regime']
        dt_dim = result['dimensionless_time'].item()

        # FDTD 数值解（在 depth_mm 处取值）
        z_arr, T_arr, nt = fdtd_photothermal(
            n, kappa, lam, I0, rho, Cp, depth_mm * 1e-3, t, k_thermal,
            nz=100
        )
        # 找到最接近 depth_mm 的网格点
        idx = np.argmin(np.abs(z_arr - depth_mm * 1e-3))
        T_fdtd = T_arr[idx]

        # 误差（以FDTD为基准）
        if abs(T_fdtd) > 1e-10:
            err = abs(T_timed - T_fdtd) / abs(T_fdtd) * 100
        else:
            err = 0.0

        print(f"  {t:<12.0e} {dt_dim:<12.2e} {regime:<14} {T_adiabatic:<12.4f} {T_timed:<12.4f} {T_fdtd:<12.4f} {err:<10.2f}")

    print()


def test_silicon_532nm_breaking_adiabatic():
    """测试2: 硅@532nm，中等脉冲，突破绝热近似。"""
    print("=" * 70)
    print("测试2: 硅 @ 532nm, 中等脉冲 (突破绝热近似)")
    print("=" * 70)

    n, kappa, lam = 4.15, 0.044, 532
    I0, rho, Cp = 1e7, 2329, 700
    D_silicon = 9.08e-5  # m²/s

    alpha = 4.0 * np.pi * kappa / (n * lam * 1e-9)
    tau_thermal = 1.0 / (D_silicon * alpha * alpha)

    print(f"  α = {alpha:.2f} /m, D = {D_silicon:.4e} m²/s")
    print(f"  热扩散特征时间 τ = 1/(D·α²) = {tau_thermal:.4e} s")
    print()

    # 硅的穿透深度 ~4μm, 测试深度 2μm
    depth_mm = 0.002
    test_times = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

    print(f"  {'时间(s)':<12} {'t·D·α²':<12} {'regime':<14} {'绝热(K)':<12} {'含时(K)':<12} {'FDTD(K)':<12} {'误差%':<10}")
    print("  " + "-" * 90)

    for t in test_times:
        result = calc_photothermal_timed(
            n, kappa, lam, I0, rho, Cp, depth_mm, t, thermal_diffusivity=D_silicon
        )
        T_adiabatic = result['T_adiabatic'].item()
        T_timed = result['T_timed'].item()
        regime = result['regime']
        dt_dim = result['dimensionless_time'].item()

        z_arr, T_arr, nt = fdtd_photothermal(
            n, kappa, lam, I0, rho, Cp, depth_mm * 1e-3, t,
            k_thermal=D_silicon * rho * Cp,
            nz=100
        )
        idx = np.argmin(np.abs(z_arr - depth_mm * 1e-3))
        T_fdtd = T_arr[idx]

        if abs(T_fdtd) > 1e-10:
            err = abs(T_timed - T_fdtd) / abs(T_fdtd) * 100
        else:
            err = 0.0

        print(f"  {t:<12.0e} {dt_dim:<12.2e} {regime:<14} {T_adiabatic:<12.4f} {T_timed:<12.4f} {T_fdtd:<12.4f} {err:<10.2f}")

    print()


def test_speed_comparison():
    """测试3: 速度对比。"""
    print("=" * 70)
    print("测试3: 速度对比 (含时解 vs FDTD)")
    print("=" * 70)

    n, kappa, lam = 1.33, 0.00012, 1064
    I0, rho, Cp = 1e7, 1000, 4186
    k_thermal = 0.598

    # 含时解计时
    t_test = 0.01  # 10ms脉冲（避免FDTD太慢）
    depth_mm = 0.1

    # 光动论
    start = time.time()
    for _ in range(1000):
        result = calc_photothermal_timed(
            n, kappa, lam, I0, rho, Cp, depth_mm, t_test, k_thermal=k_thermal
        )
    pk_time = (time.time() - start) / 1000

    # FDTD
    start = time.time()
    z_arr, T_arr, nt = fdtd_photothermal(
        n, kappa, lam, I0, rho, Cp, depth_mm * 1e-3, t_test, k_thermal, nz=100
    )
    fdtd_time = time.time() - start

    speedup = fdtd_time / pk_time

    print(f"  光动论含时解: {pk_time*1e6:.2f} μs/次")
    print(f"  FDTD (nt={nt}): {fdtd_time*1000:.2f} ms/次")
    print(f"  加速比: {speedup:.0f}x")
    print()


if __name__ == "__main__":
    test_water_1064nm_breaking_adiabatic()
    test_silicon_532nm_breaking_adiabatic()
    test_speed_comparison()
    print("=" * 70)
    print("验证完成")
