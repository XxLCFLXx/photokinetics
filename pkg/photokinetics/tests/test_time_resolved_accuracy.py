"""含时热传导模块 vs 精确数值积分 对比验证。

用 scipy.integrate.quad 作为"真值"，
验证光动论含时解 (tanh 模型) 的精度。
"""
import torch
import numpy as np
import math
from scipy import integrate
from scipy.special import erfcx as scipy_erfcx

import sys
sys.path.insert(0, '.')

from photokinetics import calc_photothermal_timed


def exact_timed_solution(n, kappa, wavelength_nm, I0, rho, Cp, z, t, D):
    """
    精确含时解 (scipy 数值积分)。
    T(z,t) = (I₀·α)/(ρCp) · ∫₀ᵗ I(τ) dτ
    其中 I(τ) = 0.5·exp(-u²)·[erfcx(w1) + erfcx(w2)]
    """
    alpha = 4.0 * math.pi * kappa / (n * wavelength_nm * 1e-9)

    if t <= 0:
        return 0.0

    def integrand(tau):
        if tau < 1e-30:
            return 0.0
        sqrt_Dtau = math.sqrt(D * tau)
        u = z / (2.0 * sqrt_Dtau)
        v = alpha * sqrt_Dtau
        w1 = v - u
        w2 = v + u
        exp_neg_u2 = math.exp(-u * u)
        return 0.5 * exp_neg_u2 * (scipy_erfcx(w1) + scipy_erfcx(w2))

    result, _ = integrate.quad(integrand, 0, t, limit=200)
    return I0 * alpha / (rho * Cp) * result


def test_accuracy_across_regimes():
    """测试不同 regime 下的精度。"""
    print("=" * 80)
    print("含时解精度验证 (tanh模型 vs 精确数值积分)")
    print("=" * 80)

    # 测试矩阵: (材料, n, kappa, lam, rho, Cp, D, z_mm, 描述)
    materials = [
        # 水 @ 1064nm (弱吸收, 慢扩散)
        ("水@1064nm", 1.33, 0.00012, 1064, 1000, 4186, 0.598/(1000*4186), 0.1),
        # 硅 @ 532nm (强吸收, 快扩散)
        ("硅@532nm",  4.15, 0.044,   532, 2329, 700,  9.08e-5,             0.002),
        # 锗 @ 532nm (极强吸收)
        ("锗@532nm",  4.00, 1.45,    532, 5323, 320,  3.50e-5,             0.0005),
    ]

    for name, n, kappa, lam, rho, Cp, D, z_mm in materials:
        print(f"\n--- {name} ---")
        alpha = 4.0 * math.pi * kappa / (n * lam * 1e-9)
        z = z_mm * 1e-3
        tau_thermal = 1.0 / (D * alpha * alpha)
        k_thermal = D * rho * Cp

        print(f"  α={alpha:.2f}/m, D={D:.2e}m²/s, τ={tau_thermal:.2e}s, z={z_mm}mm")
        print(f"  {'t(s)':<12} {'t·D·α²':<12} {'regime':<12} {'绝热(K)':<12} {'tanh(K)':<12} {'精确(K)':<12} {'tanh误差%':<12} {'绝热误差%':<12}")
        print("  " + "-" * 100)

        # 测试时间点: 跨越绝热→过渡→稳态
        test_times = [
            tau_thermal * 1e-3,   # 深绝热区
            tau_thermal * 1e-2,   # 绝热区
            tau_thermal * 0.1,    # 绝热边缘
            tau_thermal * 0.5,    # 过渡区
            tau_thermal * 1.0,    # 过渡区
            tau_thermal * 2.0,    # 过渡→稳态
            tau_thermal * 5.0,    # 稳态
            tau_thermal * 10.0,   # 深稳态
        ]

        for t in test_times:
            # 光动论含时解 (tanh 模型)
            result = calc_photothermal_timed(
                n, kappa, lam, 1e7, rho, Cp, z_mm, t, thermal_diffusivity=D
            )
            T_adiabatic = result['T_adiabatic'].item()
            T_tanh = result['T_timed'].item()
            T_steady = result['T_steady'].item()
            regime = result['regime']
            dt_dim = result['dimensionless_time'].item()

            # 精确解
            T_exact = exact_timed_solution(n, kappa, lam, 1e7, rho, Cp, z, t, D)

            # 误差
            if abs(T_exact) > 1e-10:
                err_tanh = abs(T_tanh - T_exact) / abs(T_exact) * 100
                err_adiabatic = abs(T_adiabatic - T_exact) / abs(T_exact) * 100
            else:
                err_tanh = 0.0
                err_adiabatic = 0.0

            print(f"  {t:<12.2e} {dt_dim:<12.2e} {regime:<12} {T_adiabatic:<12.6f} {T_tanh:<12.6f} {T_exact:<12.6f} {err_tanh:<12.2f} {err_adiabatic:<12.2f}")

    print()


def test_speed():
    """速度对比。"""
    print("=" * 80)
    print("速度对比: tanh模型 vs 精确数值积分")
    print("=" * 80)

    import time

    n, kappa, lam = 1.33, 0.00012, 1064
    rho, Cp = 1000, 4186
    D = 0.598 / (rho * Cp)
    z_mm = 0.1
    t = 0.1

    # tanh 模型
    start = time.time()
    for _ in range(1000):
        result = calc_photothermal_timed(
            n, kappa, lam, 1e7, rho, Cp, z_mm, t, thermal_diffusivity=D
        )
    tanh_time = (time.time() - start) / 1000

    # 精确积分
    start = time.time()
    for _ in range(100):
        T = exact_timed_solution(n, kappa, lam, 1e7, rho, Cp, z_mm*1e-3, t, D)
    exact_time = (time.time() - start) / 100

    print(f"  tanh 模型:    {tanh_time*1e6:.2f} μs/次")
    print(f"  精确积分:     {exact_time*1e6:.2f} μs/次")
    print(f"  加速比:       {exact_time/tanh_time:.0f}x")
    print(f"  (tanh模型牺牲精度换速度，绝热/稳态区精确，过渡区有误差)")


if __name__ == "__main__":
    test_accuracy_across_regimes()
    test_speed()
    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)
