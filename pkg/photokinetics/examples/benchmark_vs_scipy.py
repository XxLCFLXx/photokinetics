"""可微物理 vs scipy 基线对比。

用 scipy.optimize.least_squares + 有限差分梯度反演同样的 κ 和 D，
对比 photokinetics 可微物理方案的精度、速度、代码复杂度。

用法：
    python -m examples.benchmark_vs_scipy
"""
import time
import torch
import numpy as np
from scipy.optimize import least_squares
from photokinetics import calc_photothermal_timed
from examples.fit_transient_photothermal import (
    N_SI, KAPPA_TRUE, WAVELENGTH, RHO_SI, CP_SI, D_TRUE, I0,
    Z_POINTS_MM, T_POINTS_S, make_observations, forward, run_inversion,
)
from examples._common import positive_parameter, relative_mse, optimize


# ===== scipy 基线 =====

def _residual_scipy(params, z_np, t_np, T_obs_np):
    """scipy 残差函数：参数 → 预测 − 观测。"""
    kappa, D = params
    T_pred = np.zeros_like(T_obs_np)
    with torch.no_grad():
        for i, z in enumerate(z_np):
            for j, t in enumerate(t_np):
                result = calc_photothermal_timed(
                    N_SI, float(kappa), WAVELENGTH, I0,
                    RHO_SI, CP_SI, float(z), float(t),
                    thermal_diffusivity=float(D)
                )
                T_pred[i, j] = result['T_timed'].item()
    return (T_pred - T_obs_np).flatten()


def run_scipy_inversion(seed=42, noise=0.01):
    """scipy 反演。"""
    z_grid, t_grid, T_obs, _ = make_observations(seed=seed, noise=noise)
    z_np = z_grid.numpy()
    t_np = t_grid.numpy()
    T_obs_np = T_obs.numpy()

    # 初始猜测：用可微方案相同的初始量级
    # softplus(-1.0) ≈ 0.313, softplus(-9.0) ≈ 1.23e-4
    x0 = np.array([0.313, 1.23e-4])

    t_start = time.perf_counter()
    result = least_squares(
        _residual_scipy, x0,
        args=(z_np, t_np, T_obs_np),
        method='trf',  # Trust Region Reflective, 支持边界
        bounds=([1e-6, 1e-10], [np.inf, np.inf]),
        xtol=1e-10, ftol=1e-10, max_nfev=100,
    )
    elapsed = time.perf_counter() - t_start

    return {
        'kappa_fit': float(result.x[0]),
        'D_fit': float(result.x[1]),
        'elapsed_s': elapsed,
        'nfev': result.nfev,
        'cost': float(result.cost),
    }


# ===== 可微物理方案 =====

def run_differentiable_inversion(seed=42, noise=0.01, steps=500, lr=0.05):
    """可微物理反演（直接复用案例1的 run_inversion）。"""
    t_start = time.perf_counter()
    result = run_inversion(seed=seed, noise=noise, steps=steps, lr=lr)
    elapsed = time.perf_counter() - t_start

    result['elapsed_s'] = elapsed
    result['nfev'] = steps
    return result


# ===== 批量场景对比 =====

def run_batch_comparison(n_datasets=10, seed_base=42, noise=0.01):
    """批量反演对比：n 组数据，每组反演 κ 和 D。

    可微方案：张量维度扩展到 [n, z, t] 一次前向
    scipy 方案：循环 n 次
    """
    print(f"\n--- 批量对比（{n_datasets} 组数据）---")

    # scipy 循环
    t_start = time.perf_counter()
    scipy_results = []
    for i in range(n_datasets):
        r = run_scipy_inversion(seed=seed_base + i, noise=noise)
        scipy_results.append(r)
    scipy_elapsed = time.perf_counter() - t_start

    # 可微方案循环（首版未做张量化批量，先测串行耗时作为公平对比）
    t_start = time.perf_counter()
    diff_results = []
    for i in range(n_datasets):
        r = run_differentiable_inversion(seed=seed_base + i, noise=noise, steps=300, lr=0.05)
        diff_results.append(r)
    diff_elapsed = time.perf_counter() - t_start

    print(f"  scipy  循环 {n_datasets} 次: {scipy_elapsed:.2f}s ({scipy_elapsed/n_datasets:.2f}s/次)")
    print(f"  可微   循环 {n_datasets} 次: {diff_elapsed:.2f}s ({diff_elapsed/n_datasets:.2f}s/次)")
    print(f"  速度比: {scipy_elapsed/diff_elapsed:.2f}x")

    return {
        'n_datasets': n_datasets,
        'scipy_elapsed': scipy_elapsed,
        'diff_elapsed': diff_elapsed,
        'speedup': scipy_elapsed / diff_elapsed,
    }


def run():
    """主入口。"""
    print("=" * 70)
    print("可微物理 vs scipy 基线对比")
    print("=" * 70)
    print(f"真值: κ={KAPPA_TRUE}, D={D_TRUE} m²/s")
    print(f"场景: 硅 @ 532nm, 1% 噪声")
    print()

    # 单点对比
    print("--- 单点反演对比 ---")
    r_scipy = run_scipy_inversion(seed=42, noise=0.01)
    r_diff = run_differentiable_inversion(seed=42, noise=0.01, steps=300, lr=0.05)

    kappa_true, D_true = KAPPA_TRUE, D_TRUE

    print(f"\n  scipy.optimize.least_squares:")
    print(f"    κ_fit = {r_scipy['kappa_fit']:.6f} (误差 {abs(r_scipy['kappa_fit']-kappa_true)/kappa_true:.4%})")
    print(f"    D_fit = {r_scipy['D_fit']:.4e} (误差 {abs(r_scipy['D_fit']-D_true)/D_true:.4%})")
    print(f"    耗时 = {r_scipy['elapsed_s']:.2f}s, nfev = {r_scipy['nfev']}")

    print(f"\n  可微物理 (Adam + autograd):")
    print(f"    κ_fit = {r_diff['kappa_fit']:.6f} (误差 {abs(r_diff['kappa_fit']-kappa_true)/kappa_true:.4%})")
    print(f"    D_fit = {r_diff['D_fit']:.4e} (误差 {abs(r_diff['D_fit']-D_true)/D_true:.4%})")
    print(f"    耗时 = {r_diff['elapsed_s']:.2f}s, steps = {r_diff['nfev']}")

    print(f"\n  速度比 (scipy/可微): {r_scipy['elapsed_s']/r_diff['elapsed_s']:.2f}x")
    print(f"  精度比: scipy 误差 {abs(r_scipy['kappa_fit']-kappa_true)/kappa_true:.4%} vs "
          f"可微误差 {abs(r_diff['kappa_fit']-kappa_true)/kappa_true:.4%}")

    # 批量对比（可选，跳过以节省时间）
    # run_batch_comparison(n_datasets=5, seed_base=100, noise=0.01)
    print("\n注：批量对比默认跳过（scipy 有限差分耗时较长），如需运行请取消注释。")


if __name__ == "__main__":
    run()
