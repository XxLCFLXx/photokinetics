"""基线 2：纯数据驱动 NN（无物理约束）。

核心论点：纯数据 NN 没有物理约束，只在观测点附近拟合，
        在无观测区域完全失败（外推能力差）。

模型:
    T_data(z, t) — 普通 MLP，输入 (z, t)，输出 T
    无 κ 网络，无 PDE 残差，无 IC/BC 约束

损失:
    L = MSE(T_data(z_obs, t_obs), T_obs)

预期:
    - 观测点 loss 低（过拟合）
    - 无观测区域 loss 高（外推失败）
    - 无法反演 κ(z)

用法:
    python -m examples.baseline_pure_nn
"""
import time
import numpy as np
import torch
import torch.nn as nn

from examples._finite_difference import (
    generate_ground_truth, sample_observations,
)
from examples._pinn_nonuniform import L_Z, L_T, T_SCALE


class PureDataNetwork(nn.Module):
    """纯数据驱动温度场网络 T_data(z̄, t̄) → T̄。

    架构与 PINN 的 TemperatureNetwork 相同（公平对比），
    但无任何物理约束。

    输入: (z̄, t̄) ∈ [0, 1]² (归一化)
    输出: T̄ (归一化), 实际 T = T_SCALE · T̄
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, zt):
        return self.net(zt)


def _eval_mse(net, z_m, t_s, T_true):
    """在给定点上计算 MSE。

    z_m, t_s — numpy arrays (实际量纲 m, s)
    T_true   — numpy array (K)
    """
    z_norm = torch.tensor(z_m / L_Z, dtype=torch.float32).unsqueeze(1)
    t_norm = torch.tensor(t_s / L_T, dtype=torch.float32).unsqueeze(1)
    zt = torch.cat([z_norm, t_norm], dim=1)
    with torch.no_grad():
        T_pred = net(zt).squeeze().numpy() * T_SCALE
    return float(np.mean((T_pred - T_true) ** 2))


def run_pure_nn_baseline(I0=1e5, seed=42, noise=0.01, steps=2000, lr=1e-3,
                         z_obs_um=None, t_obs_s=None):
    """纯数据驱动 NN 基线。

    返回:
        dict with:
            loss_obs          — 观测点最终 MSE (K²)
            loss_extrapolation— 无观测区域 MSE (K²)
            elapsed_s         — 训练耗时
            net               — 训练后的网络
            history           — loss 历史
    """
    if z_obs_um is None:
        z_obs_um = [1, 5, 10, 20, 30]
    if t_obs_s is None:
        t_obs_s = [0.001, 0.005, 0.01, 0.05]

    torch.manual_seed(seed)

    # 生成真值 + 采样观测
    gt = generate_ground_truth(I0=I0)
    z_obs, t_obs, T_obs = sample_observations(
        gt, z_obs_um, t_obs_s, noise=noise, seed=seed,
    )

    # 归一化观测点
    z_obs_t = torch.tensor(z_obs / L_Z, dtype=torch.float32).unsqueeze(1)
    t_obs_t = torch.tensor(t_obs / L_T, dtype=torch.float32).unsqueeze(1)
    T_obs_t = torch.tensor(T_obs / T_SCALE, dtype=torch.float32).unsqueeze(1)
    zt_obs = torch.cat([z_obs_t, t_obs_t], dim=1)

    # 训练纯数据 NN
    net = PureDataNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    t_start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad()
        T_pred = net(zt_obs)
        loss = torch.mean((T_pred - T_obs_t) ** 2)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    elapsed = time.perf_counter() - t_start

    # 观测点 MSE（实际量纲 K²）
    loss_obs = _eval_mse(net, z_obs, t_obs, T_obs)

    # 外推点 MSE：选无观测的 (z, t) 点
    # 真值场中 z ∈ {3, 7, 15, 25} μm, t ∈ {0.002, 0.02, 0.08} s
    # 这些点不在观测集合中
    z_extra_um = [3, 7, 15, 25]
    t_extra_s = [0.002, 0.02, 0.08]
    z_extra_m = np.array(z_extra_um) * 1e-6
    t_extra_arr = np.array(t_extra_s)

    # 用真值场插值得到外推点真值
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (gt["z_array"], gt["t_array"]), gt["T_field"],
        bounds_error=False, fill_value=0.0,
    )
    Z_extra, T_extra = np.meshgrid(z_extra_m, t_extra_arr, indexing="ij")
    pts = np.stack([Z_extra.ravel(), T_extra.ravel()], axis=-1)
    T_extra_true = interp(pts)

    loss_extrapolation = _eval_mse(
        net, Z_extra.ravel(), T_extra.ravel(), T_extra_true,
    )

    return {
        "loss_obs": loss_obs,
        "loss_extrapolation": loss_extrapolation,
        "elapsed_s": elapsed,
        "net": net,
        "history": history,
        "z_extra_um": z_extra_um,
        "t_extra_s": t_extra_s,
    }


def run():
    """主入口。"""
    print("=" * 70)
    print("基线 2：纯数据驱动 NN（无物理约束）")
    print("=" * 70)

    r = run_pure_nn_baseline(I0=1e5, seed=42, noise=0.01, steps=2000)

    print(f"\n训练: {len(r['history'])} 步, 耗时 {r['elapsed_s']:.2f}s")
    print(f"\n观测点 MSE:     {r['loss_obs']:.4e} K²")
    print(f"外推点 MSE:     {r['loss_extrapolation']:.4e} K²")
    print(f"外推/观测 比:   {r['loss_extrapolation']/max(r['loss_obs'],1e-30):.1f}x")

    print("\n核心局限:")
    print("  纯数据 NN 在观测点过拟合，但在无观测区域完全失败")
    print("  无法提供任何物理一致性（不满足 PDE、IC、BC）")
    print("  无法反演 κ(z)（根本没有 κ 网络）")
    return r


if __name__ == "__main__":
    run()
