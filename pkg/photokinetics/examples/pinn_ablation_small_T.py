"""消融实验：降低 T 网络容量以抑制 shortcut 问题。

假设: T 网络容量过大时直接拟合数据，绕过 κ→α→I→q→T 物理链路，
      κ 网络失去梯度信号。降低 T 网络容量可强制其依赖物理链路。

配置: T 网络 1 层 × 16 神经元（原 3×64，参数量 65 → 2113 → 65）
      其余超参数与 case 4 主实验一致（lambda_data=5, lambda_phys=50, cosine lr）

用法:
    python -m examples.pinn_ablation_small_T
"""
import os
import json
import time
import numpy as np
import torch

from examples._finite_difference import (
    generate_ground_truth, sample_observations, kappa_true_func,
    N_POLY, KAPPA0, WAVELENGTH_NM, RHO_POLY, CP_POLY, D_POLY,
    DEFECT_Z0,
)
from examples._pinn_nonuniform import (
    TemperatureNetwork, KappaNetwork, train_pinn, L_Z, L_T, T_SCALE,
)
from examples.pinn_inverse_kappa import _eval_kappa_field, _eval_temperature_field


def run_ablation(T_hidden_dim=16, T_num_layers=1, steps=5000, seed=42):
    """跑一次消融实验。"""
    print("=" * 70)
    print(f"消融实验: T 网络 {T_num_layers}×{T_hidden_dim} (原 3×64)")
    print("=" * 70)

    I0 = 1e5
    torch.manual_seed(seed)

    # 1. 真值 + 观测（与主实验一致）
    gt = generate_ground_truth(I0=I0)
    z_obs_um = [1, 5, 10, 20, 30]
    t_obs_s = [0.001, 0.005, 0.01, 0.05]
    z_obs, t_obs, T_obs = sample_observations(
        gt, z_obs_um, t_obs_s, noise=0.01, seed=seed,
    )
    z_obs_t = torch.tensor(z_obs, dtype=torch.float32).unsqueeze(1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
    T_obs_t = torch.tensor(T_obs, dtype=torch.float32).unsqueeze(1)

    # 2. 训练（与主实验超参数一致）
    T_net = TemperatureNetwork(hidden_dim=T_hidden_dim, num_layers=T_num_layers)
    k_net = KappaNetwork()

    # 打印参数量对比
    n_params_T = sum(p.numel() for p in T_net.parameters())
    n_params_k = sum(p.numel() for p in k_net.parameters())
    print(f"T 网络参数量: {n_params_T} (原 3×64 = {2*64 + 64*64 + 64*1 + 64 + 64 + 1})")
    print(f"κ 网络参数量: {n_params_k}")

    t_start = time.perf_counter()
    history, components_history = train_pinn(
        T_net, k_net, z_obs_t, t_obs_t, T_obs_t,
        I0=I0, D=D_POLY, rho=RHO_POLY, Cp=CP_POLY,
        n=N_POLY, wavelength_nm=WAVELENGTH_NM,
        kappa_prior=KAPPA0,
        steps=steps, lr=1e-3, n_col=256, seed=seed,
        lambda_data=5.0, lambda_phys=50.0, lambda_prior=0.001,
        verbose=True, verbose_every=max(steps // 10, 100),
        lr_schedule="cosine", lr_min=1e-5,
        phys_warmup_steps=max(steps // 10, 200),
    )
    elapsed = time.perf_counter() - t_start

    # 3. 评估 κ 反演
    eval_z_um = np.linspace(0, 30, 60).tolist()
    eval_z_m = np.array(eval_z_um) * 1e-6
    kappa_pred = _eval_kappa_field(k_net, eval_z_m)
    kappa_true = kappa_true_func(eval_z_m)
    kappa_pred_at_defect = float(_eval_kappa_field(k_net, np.array([DEFECT_Z0]))[0])
    kappa_true_at_defect = float(kappa_true_func(np.array([DEFECT_Z0]))[0])
    kappa_mse = float(np.mean(((kappa_pred - kappa_true) / kappa_true) ** 2))
    kappa_err_defect = abs(kappa_pred_at_defect - kappa_true_at_defect) / kappa_true_at_defect

    # 4. 评估 T 场
    eval_t_arr = np.linspace(0.001, 0.1, 30)
    T_pred_field = _eval_temperature_field(T_net, eval_z_m, eval_t_arr)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (gt["z_array"], gt["t_array"]), gt["T_field"],
        bounds_error=False, fill_value=0.0,
    )
    Z_eval, T_eval = np.meshgrid(eval_z_m, eval_t_arr, indexing="ij")
    T_true_field = interp(np.stack([Z_eval.ravel(), T_eval.ravel()], axis=-1)).reshape(
        T_pred_field.shape
    )
    mask = np.abs(T_true_field) > 1e-6
    T_field_mse = float(np.mean((T_pred_field - T_true_field)[mask] ** 2))

    # 5. 打印结果
    print("\n" + "=" * 70)
    print("消融实验结果")
    print("=" * 70)
    print(f"κ(z₀) 预测: {kappa_pred_at_defect:.4f}, 真值: {kappa_true_at_defect:.4f}, "
          f"误差: {kappa_err_defect:.2%}")
    print(f"κ(z)  相对 MSE: {kappa_mse:.4e}")
    print(f"T(z,t) MSE: {T_field_mse:.4e} K²")
    print(f"训练耗时: {elapsed:.1f}s")

    # 与原配置对比
    ablation_label = f"{T_num_layers}x{T_hidden_dim}"
    print("\n--- 与原配置 (3x64) 对比 ---")
    print(f"{'指标':<25} {'原 3x64':<15} {('消融 ' + ablation_label):<15} {'改善':<10}")
    print("-" * 65)
    print(f"{'kappa(z0) 误差':<25} {'75.00%':<15} {kappa_err_defect:.2%}")
    print(f"{'T(z,t) MSE (K2)':<25} {'7.15':<15} {T_field_mse:.4f}")

    # 6. 保存结果
    paper_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "paper",
    )
    out_path = os.path.join(paper_dir, f"pinn_ablation_T{T_num_layers}x{T_hidden_dim}.json")
    summary = {
        "config": {
            "T_hidden_dim": T_hidden_dim,
            "T_num_layers": T_num_layers,
            "T_params": n_params_T,
            "steps": steps,
            "lambda_data": 5.0,
            "lambda_phys": 50.0,
            "lambda_prior": 0.001,
            "lr_schedule": "cosine",
            "phys_warmup_steps": max(steps // 10, 200),
        },
        "results": {
            "kappa_pred_at_defect": kappa_pred_at_defect,
            "kappa_true_at_defect": kappa_true_at_defect,
            "kappa_err_at_defect": kappa_err_defect,
            "kappa_mse_rel": kappa_mse,
            "T_field_mse_K2": T_field_mse,
            "elapsed_s": elapsed,
            "kappa_pred_array": kappa_pred.tolist(),
            "kappa_true_array": kappa_true.tolist(),
            "eval_z_um": eval_z_um,
            "history": [float(h) for h in history],
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    return summary


if __name__ == "__main__":
    run_ablation(T_hidden_dim=16, T_num_layers=1, steps=5000, seed=42)
