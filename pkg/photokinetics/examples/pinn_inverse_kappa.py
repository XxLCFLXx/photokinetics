"""PINN 非均匀 κ(z) 反演主入口。

整合 PINN 训练、基线对比、结果评估。

流程:
    1. 生成真值 T(z,t)（隐式 FD，非均匀 κ）
    2. 采样稀疏观测
    3. 训练 PINN（双网络：T_θ + κ_φ）
    4. 评估：
       - κ(z) 反演误差（缺陷层处）
       - T(z,t) 场重建 MSE
       - 外推 MSE（无观测区域）
    5. 对比基线 1（scipy 均匀 κ）和基线 2（纯数据 NN）

用法:
    python -m examples.pinn_inverse_kappa
"""
import os
import time
import numpy as np
import torch

from examples._finite_difference import (
    generate_ground_truth, sample_observations, kappa_true_func,
    N_POLY, KAPPA0, WAVELENGTH_NM, RHO_POLY, CP_POLY, D_POLY,
    DEFECT_Z0, DEFECT_AMP, DEFECT_SIGMA,
)
from examples._pinn_nonuniform import (
    TemperatureNetwork, KappaNetwork, train_pinn, L_Z, L_T, T_SCALE,
)


def _eval_kappa_field(k_net, z_array_m):
    """评估 κ_φ(z) 在 z_array_m 上的预测值。

    返回 1D numpy array（即使输入是标量）。
    """
    z_array_m = np.atleast_1d(np.asarray(z_array_m, dtype=np.float64))
    z_norm = torch.tensor(z_array_m / L_Z, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        k_pred = k_net(z_norm).squeeze(-1).numpy()
    return k_pred


def _eval_temperature_field(T_net, z_array_m, t_array_s):
    """评估 T_θ(z, t) 在 (z_array_m, t_array_s) 网格上的预测值。

    返回 T_field shape [Nz, Nt]
    """
    Nz = len(z_array_m)
    Nt = len(t_array_s)
    T_field = np.zeros((Nz, Nt))
    with torch.no_grad():
        for i, z in enumerate(z_array_m):
            z_norm = torch.tensor([[z / L_Z]], dtype=torch.float32).expand(Nt, 1)
            t_norm = torch.tensor(
                np.array(t_array_s) / L_T, dtype=torch.float32
            ).unsqueeze(1)
            zt = torch.cat([z_norm, t_norm], dim=1)
            T_pred = T_net(zt).squeeze().numpy() * T_SCALE
            T_field[i, :] = T_pred
    return T_field


def run_inversion(I0=1e5, seed=42, noise=0.01, steps=5000, lr=1e-3, n_col=256,
                  z_obs_um=None, t_obs_s=None,
                  lambda_data=1.0, lambda_phys=1.0, lambda_prior=0.01,
                  eval_z_um=None, eval_t_s=None,
                  verbose=True,
                  lr_schedule="constant", lr_min=1e-5,
                  phys_warmup_steps=0, phys_warmup_max=None,
                  T_hidden_dim=64, T_num_layers=3):
    """完整 PINN 反演流程。

    新增参数:
        T_hidden_dim  — T 网络隐藏层宽度（默认 64，实验时可降到 16 以抑制 shortcut）
        T_num_layers  — T 网络隐藏层数（默认 3，实验时可降到 1）

    默认使用 lambda_phys=1.0 和 constant 学习率（适合任意步数）。
    完整实验建议通过 run() 调用，使用 cosine lr + lambda_phys=20 + warmup。

    返回:
        dict with:
            kappa_pred_at_defect — κ_φ(z₀) 预测值
            kappa_true_at_defect — κ_true(z₀) 真值
            kappa_pred_array     — κ_φ(z) 在评估网格上
            kappa_true_array     — κ_true(z) 在评估网格上
            kappa_mse            — κ(z) 反演 MSE (相对)
            T_field_mse          — T(z,t) 场重建 MSE (绝对 K²)
            T_field_mse_rel      — T(z,t) 场重建 MSE (相对)
            T_field_mse_extrapolation — 无观测区域 MSE (绝对 K²)
            T_field_mse_extrapolation_rel — 无观测区域 MSE (相对)
            history              — PINN 训练 loss 历史
            components_history   — 各 loss 分量历史
            T_net, k_net         — 训练后的网络
            elapsed_s            — 训练耗时
            gt                   — 真值场 dict
            z_obs, t_obs, T_obs  — 观测点
    """
    if z_obs_um is None:
        z_obs_um = [1, 5, 10, 20, 30]
    if t_obs_s is None:
        t_obs_s = [0.001, 0.005, 0.01, 0.05]
    if eval_z_um is None:
        eval_z_um = np.linspace(0, 30, 60).tolist()
    if eval_t_s is None:
        eval_t_s = np.linspace(0.001, 0.1, 30).tolist()

    torch.manual_seed(seed)

    # ===== 1. 生成真值 + 采样观测 =====
    if verbose:
        print("--- 1. 生成真值 T(z,t) ---")
    gt = generate_ground_truth(I0=I0)
    z_obs, t_obs, T_obs = sample_observations(
        gt, z_obs_um, t_obs_s, noise=noise, seed=seed,
    )

    z_obs_t = torch.tensor(z_obs, dtype=torch.float32).unsqueeze(1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
    T_obs_t = torch.tensor(T_obs, dtype=torch.float32).unsqueeze(1)

    # ===== 2. 训练 PINN =====
    if verbose:
        print(f"--- 2. 训练 PINN ({steps} 步) ---")
    T_net = TemperatureNetwork(hidden_dim=T_hidden_dim, num_layers=T_num_layers)
    k_net = KappaNetwork()

    t_start = time.perf_counter()
    history, components_history = train_pinn(
        T_net, k_net, z_obs_t, t_obs_t, T_obs_t,
        I0=I0, D=D_POLY, rho=RHO_POLY, Cp=CP_POLY,
        n=N_POLY, wavelength_nm=WAVELENGTH_NM,
        kappa_prior=KAPPA0,
        steps=steps, lr=lr, n_col=n_col, seed=seed,
        lambda_data=lambda_data, lambda_phys=lambda_phys, lambda_prior=lambda_prior,
        verbose=verbose, verbose_every=max(steps // 10, 100),
        lr_schedule=lr_schedule, lr_min=lr_min,
        phys_warmup_steps=phys_warmup_steps, phys_warmup_max=phys_warmup_max,
    )
    elapsed = time.perf_counter() - t_start

    # ===== 3. 评估 κ(z) 反演 =====
    if verbose:
        print("--- 3. 评估 κ(z) 反演 ---")
    eval_z_m = np.array(eval_z_um) * 1e-6
    kappa_pred = _eval_kappa_field(k_net, eval_z_m)
    kappa_true = kappa_true_func(eval_z_m)

    kappa_pred_at_defect = float(_eval_kappa_field(k_net, np.array([DEFECT_Z0]))[0])
    kappa_true_at_defect = float(kappa_true_func(np.array([DEFECT_Z0]))[0])

    # 相对 MSE
    kappa_mse = float(np.mean(((kappa_pred - kappa_true) / kappa_true) ** 2))

    # ===== 4. 评估 T(z,t) 场重建 =====
    if verbose:
        print("--- 4. 评估 T(z,t) 场重建 ---")
    eval_t_arr = np.array(eval_t_s)
    T_pred_field = _eval_temperature_field(T_net, eval_z_m, eval_t_arr)

    # 真值场插值到评估网格
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (gt["z_array"], gt["t_array"]), gt["T_field"],
        bounds_error=False, fill_value=0.0,
    )
    Z_eval, T_eval = np.meshgrid(eval_z_m, eval_t_arr, indexing="ij")
    T_true_field = interp(np.stack([Z_eval.ravel(), T_eval.ravel()], axis=-1)).reshape(
        T_pred_field.shape
    )

    # 全场 MSE（绝对 K² 和相对）
    mask_nonzero = np.abs(T_true_field) > 1e-6
    diff_sq = (T_pred_field - T_true_field) ** 2
    T_field_mse_abs = float(np.mean(diff_sq[mask_nonzero]))
    T_field_mse_rel = float(np.mean(
        (diff_sq[mask_nonzero] / (T_true_field[mask_nonzero] ** 2))
    ))

    # 外推 MSE：无观测区域
    # 观测点 z ∈ {1, 5, 10, 20, 30} μm, t ∈ {0.001, 0.005, 0.01, 0.05} s
    # 外推点：z 不在观测集合中
    z_obs_set = set(z_obs_um)
    extra_mask = np.array(
        [[z_um not in z_obs_set for z_um in eval_z_um] for _ in eval_t_s]
    ).T  # shape [Nz, Nt]
    extra_valid = extra_mask & mask_nonzero
    if np.any(extra_valid):
        T_field_mse_extrapolation = float(np.mean(diff_sq[extra_valid]))  # K²
        T_field_mse_extrapolation_rel = float(np.mean(
            diff_sq[extra_valid] / (T_true_field[extra_valid] ** 2)
        ))
    else:
        T_field_mse_extrapolation = T_field_mse_abs
        T_field_mse_extrapolation_rel = T_field_mse_rel

    if verbose:
        print(f"\n===== PINN 反演结果 =====")
        print(f"κ(z₀) 预测: {kappa_pred_at_defect:.4f}, 真值: {kappa_true_at_defect:.4f}, "
              f"误差: {abs(kappa_pred_at_defect-kappa_true_at_defect)/kappa_true_at_defect:.2%}")
        print(f"κ(z)  相对 MSE: {kappa_mse:.4e}")
        print(f"T(z,t) 绝对 MSE: {T_field_mse_abs:.4e} K², 相对: {T_field_mse_rel:.4e}")
        print(f"T(z,t) 外推 MSE: {T_field_mse_extrapolation:.4e} K² (绝对)")
        print(f"训练耗时: {elapsed:.2f}s ({steps} 步)")

    return {
        "kappa_pred_at_defect": kappa_pred_at_defect,
        "kappa_true_at_defect": kappa_true_at_defect,
        "kappa_pred_array": kappa_pred,
        "kappa_true_array": kappa_true,
        "kappa_mse": kappa_mse,
        "T_field_mse": T_field_mse_abs,           # 绝对 K² (与基线一致)
        "T_field_mse_rel": T_field_mse_rel,        # 相对
        "T_field_mse_extrapolation": T_field_mse_extrapolation,        # 绝对 K²
        "T_field_mse_extrapolation_rel": T_field_mse_extrapolation_rel,  # 相对
        "history": history,
        "components_history": components_history,
        "T_net": T_net,
        "k_net": k_net,
        "elapsed_s": elapsed,
        "gt": gt,
        "z_obs": z_obs, "t_obs": t_obs, "T_obs": T_obs,
        "eval_z_um": eval_z_um, "eval_t_s": eval_t_s,
        "T_pred_field": T_pred_field, "T_true_field": T_true_field,
    }


def run_full_comparison(steps=5000, seed=42):
    """运行 PINN + 两个基线的完整对比。"""
    print("=" * 70)
    print("PINN 非均匀 κ(z) 反演：完整对比")
    print("=" * 70)

    # PINN（完整实验用优化超参数）
    print("\n>>> 1. PINN 反演")
    pinn = run_inversion(
        steps=steps, seed=seed,
        lambda_data=5.0,            # 增强数据拟合，给 κ 网络更强梯度信号
        lambda_phys=50.0,           # 增强物理约束
        lambda_prior=0.001,         # 降低先验，让 κ 网络自由探索
        lr_schedule="cosine",       # 余弦退火
        phys_warmup_steps=max(steps // 10, 200),  # 更短的 warmup
    )

    # 基线 1: scipy 均匀 κ
    print("\n>>> 2. 基线 1: scipy 均匀 κ")
    from examples.baseline_uniform_scipy import run_scipy_baseline
    bl1 = run_scipy_baseline(I0=1e5, seed=seed, noise=0.01)

    # 基线 2: 纯数据 NN
    print("\n>>> 3. 基线 2: 纯数据 NN")
    from examples.baseline_pure_nn import run_pure_nn_baseline
    bl2 = run_pure_nn_baseline(I0=1e5, seed=seed, noise=0.01, steps=steps)

    # 汇总
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    print(f"{'方法':<25} {'κ(z) 反演':<15} {'T场 MSE':<15} {'外推 MSE':<15}")
    print("-" * 70)
    print(f"{'scipy 均匀 κ':<25} {'N/A (标量)':<15} "
          f"{bl1['mse_obs']:.4e}    {'N/A':<15}")
    print(f"{'纯数据 NN':<25} {'N/A':<15} "
          f"{bl2['loss_obs']:.4e}    {bl2['loss_extrapolation']:.4e}")
    print(f"{'PINN':<25} "
          f"{pinn['kappa_pred_at_defect']:.4f} vs {pinn['kappa_true_at_defect']:.4f}  "
          f"{pinn['T_field_mse']:.4e}    {pinn['T_field_mse_extrapolation']:.4e}")

    return {"pinn": pinn, "baseline_scipy": bl1, "baseline_pure_nn": bl2}


def run():
    """主入口。"""
    return run_full_comparison(steps=5000, seed=42)


def _save_results_to_json(results, out_path):
    """将完整对比结果保存为 JSON（仅可序列化字段）。"""
    import json
    import os

    def _to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return obj

    pinn = results["pinn"]
    bl1 = results["baseline_scipy"]
    bl2 = results["baseline_pure_nn"]

    summary = {
        "meta": {
            "steps": len(pinn["history"]) - 1 if pinn["history"] else 0,
            "seed": 42,
            "lambda_phys": 20.0,
            "lr_schedule": "cosine",
            "phys_warmup_steps": max(len(pinn["history"]) // 5, 100) if pinn["history"] else 0,
        },
        "pinn": {
            "kappa_pred_at_defect": _to_serializable(pinn["kappa_pred_at_defect"]),
            "kappa_true_at_defect": _to_serializable(pinn["kappa_true_at_defect"]),
            "kappa_rel_error_at_defect": abs(
                pinn["kappa_pred_at_defect"] - pinn["kappa_true_at_defect"]
            ) / pinn["kappa_true_at_defect"],
            "kappa_mse_rel": _to_serializable(pinn["kappa_mse"]),
            "T_field_mse_abs_K2": _to_serializable(pinn["T_field_mse"]),
            "T_field_mse_rel": _to_serializable(pinn["T_field_mse_rel"]),
            "T_field_mse_extrapolation_abs_K2": _to_serializable(
                pinn["T_field_mse_extrapolation"]
            ),
            "T_field_mse_extrapolation_rel": _to_serializable(
                pinn["T_field_mse_extrapolation_rel"]
            ),
            "elapsed_s": _to_serializable(pinn["elapsed_s"]),
            "kappa_pred_array": _to_serializable(pinn["kappa_pred_array"]),
            "kappa_true_array": _to_serializable(pinn["kappa_true_array"]),
            "eval_z_um": _to_serializable(pinn["eval_z_um"]),
            "history": _to_serializable(np.array(pinn["history"])),
            "components_history": {
                k: _to_serializable(np.array([c[k] for c in pinn["components_history"]]))
                for k in pinn["components_history"][0].keys()
            },
        },
        "baseline_scipy": {
            "kappa_eff": _to_serializable(bl1["kappa_eff"]),
            "kappa0_true": _to_serializable(bl1["kappa0_true"]),
            "kappa_defect_true": _to_serializable(bl1["kappa_defect_true"]),
            "mse_obs_K2": _to_serializable(bl1["mse_obs"]),
            "elapsed_s": _to_serializable(bl1["elapsed_s"]),
            "nfev": _to_serializable(bl1["nfev"]),
        },
        "baseline_pure_nn": {
            "loss_obs_K2": _to_serializable(bl2["loss_obs"]),
            "loss_extrapolation_K2": _to_serializable(bl2["loss_extrapolation"]),
            "elapsed_s": _to_serializable(bl2["elapsed_s"]),
            "history": _to_serializable(np.array(bl2["history"])),
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    results = run()
    paper_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "paper",
    )
    _save_results_to_json(
        results,
        out_path=os.path.join(paper_dir, "pinn_results_case4.json"),
    )

    # 训练完成后直接生成配图（避免重复训练）
    print("\n>>> 生成配图...")
    from examples.make_zhihu_figures_case4 import (
        plot_kappa_inversion,
        plot_temperature_field_comparison,
        plot_training_loss,
        plot_method_comparison,
        ensure_output_dir,
    )
    ensure_output_dir()
    pinn = results["pinn"]
    bl1 = results["baseline_scipy"]
    bl2 = results["baseline_pure_nn"]
    plot_kappa_inversion(pinn)
    plot_temperature_field_comparison(pinn, bl2)
    plot_training_loss(pinn)
    plot_method_comparison(pinn, bl1, bl2)
    print("\n所有配图已生成。")
