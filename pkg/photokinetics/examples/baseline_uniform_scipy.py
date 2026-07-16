"""基线 1：scipy 均匀 κ 假设下的等效反演。

核心论点：解析解假设 κ 是常数，无法表示非均匀 κ(z)。
本基线用 scipy.optimize.least_squares 反演一个"等效"标量 κ_eff，
展示其局限性：
  1. κ_eff 与真值 κ(z) 在缺陷层处偏差巨大
  2. 用 κ_eff 重建的 T(z,t) 在缺陷层附近误差大
  3. 只能给出"等效平均"参数，无法揭示空间分布

物理模型（前向）:
    T_pred(z, t; κ_eff) = calc_photothermal_timed(n, κ_eff, λ, I0, ρ, Cp, z, t, D)

反演:
    min_{κ_eff}  Σ_i (T_pred(z_i, t_i; κ_eff) - T_obs_i)²

用法:
    python -m examples.baseline_uniform_scipy
"""
import time
import numpy as np
import torch
from scipy.optimize import least_squares

from photokinetics import calc_photothermal_timed
from examples._finite_difference import (
    generate_ground_truth, sample_observations,
    N_POLY, KAPPA0, WAVELENGTH_NM, RHO_POLY, CP_POLY, D_POLY,
    DEFECT_Z0, DEFECT_AMP,
)


def _residual_uniform(kappa_eff, z_obs_m, t_obs_s, T_obs, I0):
    """scipy 残差: T_pred(z, t; κ_eff) - T_obs。

    z_obs_m  — 观测点 z (m)
    t_obs_s  — 观测点 t (s)
    T_obs    — 观测温升 (K)
    """
    T_pred = np.zeros_like(T_obs)
    with torch.no_grad():
        for i, (z, t) in enumerate(zip(z_obs_m, t_obs_s)):
            result = calc_photothermal_timed(
                N_POLY, float(kappa_eff[0]), WAVELENGTH_NM, I0,
                RHO_POLY, CP_POLY, z * 1e3, float(t),  # 注意 z 单位 mm
                thermal_diffusivity=D_POLY,
            )
            T_pred[i] = result["T_timed"].item()
    return T_pred - T_obs


def run_scipy_baseline(I0=1e5, seed=42, noise=0.01,
                       z_obs_um=None, t_obs_s=None):
    """scipy 均匀 κ 反演基线。

    返回:
        dict with:
            kappa_eff        — 反演得到的等效 κ_eff
            kappa0_true      — 基底真值 κ₀
            kappa_defect_true— 缺陷层真值 κ(z₀) = (1+A)·κ₀
            elapsed_s        — 反演耗时
            nfev             — 函数求值次数
            T_pred_obs       — 观测点上的预测 T
            T_obs            — 观测真值
            z_obs_m, t_obs_s — 观测点坐标
            mse_obs          — 观测点 MSE
    """
    if z_obs_um is None:
        z_obs_um = [1, 5, 10, 20, 30]
    if t_obs_s is None:
        t_obs_s = [0.001, 0.005, 0.01, 0.05]

    # 生成真值 + 采样观测
    gt = generate_ground_truth(I0=I0)
    z_obs, t_obs, T_obs = sample_observations(
        gt, z_obs_um, t_obs_s, noise=noise, seed=seed,
    )

    # 初值：基底 κ₀
    x0 = np.array([KAPPA0])

    t_start = time.perf_counter()
    result = least_squares(
        _residual_uniform, x0,
        args=(z_obs, t_obs, T_obs, I0),
        method="trf",
        bounds=([1e-6], [1.0]),
        xtol=1e-12, ftol=1e-12, max_nfev=200,
    )
    elapsed = time.perf_counter() - t_start

    kappa_eff = float(result.x[0])
    kappa_defect_true = KAPPA0 * (1.0 + DEFECT_AMP)  # 4·κ₀

    # 观测点预测
    T_pred = T_obs + result.fun

    return {
        "kappa_eff": kappa_eff,
        "kappa0_true": KAPPA0,
        "kappa_defect_true": kappa_defect_true,
        "elapsed_s": elapsed,
        "nfev": result.nfev,
        "T_pred_obs": T_pred,
        "T_obs": T_obs,
        "z_obs_m": z_obs,
        "t_obs_s": t_obs,
        "mse_obs": float(np.mean((T_pred - T_obs) ** 2)),
        "z_obs_um": np.array(z_obs_um),
        "t_obs_s_arr": np.array(t_obs_s),
    }


def reconstruct_field_uniform(kappa_eff, z_array_m, t_array_s, I0):
    """用反演得到的 κ_eff 重建 T(z,t) 场（均匀 κ 假设）。

    返回: T_field shape [Nz, Nt]
    """
    Nz = len(z_array_m)
    Nt = len(t_array_s)
    T_field = np.zeros((Nz, Nt))
    with torch.no_grad():
        for i, z in enumerate(z_array_m):
            for j, t in enumerate(t_array_s):
                result = calc_photothermal_timed(
                    N_POLY, kappa_eff, WAVELENGTH_NM, I0,
                    RHO_POLY, CP_POLY, z * 1e3, float(t),
                    thermal_diffusivity=D_POLY,
                )
                T_field[i, j] = result["T_timed"].item()
    return T_field


def run():
    """主入口。"""
    print("=" * 70)
    print("基线 1：scipy 均匀 κ 假设下的等效反演")
    print("=" * 70)

    r = run_scipy_baseline(I0=1e5, seed=42, noise=0.01)

    print(f"\n真值: κ₀ = {r['kappa0_true']:.4f}, κ(z₀) = {r['kappa_defect_true']:.4f}")
    print(f"反演: κ_eff = {r['kappa_eff']:.4f}")
    print(f"  vs κ₀:   误差 {abs(r['kappa_eff']-r['kappa0_true'])/r['kappa0_true']:.2%}")
    print(f"  vs κ(z₀): 误差 {abs(r['kappa_eff']-r['kappa_defect_true'])/r['kappa_defect_true']:.2%}")
    print(f"\n耗时: {r['elapsed_s']:.2f}s, nfev = {r['nfev']}")
    print(f"观测点 MSE: {r['mse_obs']:.4e}")

    print("\n核心局限:")
    print(f"  κ_eff = {r['kappa_eff']:.4f} 是一个'等效平均'值")
    print(f"  既不是 κ₀ = {r['kappa0_true']:.4f}, 也不是 κ(z₀) = {r['kappa_defect_true']:.4f}")
    print(f"  无法揭示 κ(z) 的空间分布（缺陷层位置/形状）")
    return r


if __name__ == "__main__":
    run()
