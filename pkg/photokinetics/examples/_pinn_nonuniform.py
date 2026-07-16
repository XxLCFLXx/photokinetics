"""PINN 核心：非均匀 κ(z) 反演的双网络模型与可微物理约束。

架构:
    1. TemperatureNetwork T_θ(z̄, t̄) — 温度场（归一化输入输出）
    2. KappaNetwork κ_φ(z̄) — 消光系数场（归一化输入输出）

归一化:
    z̄ = z / L_z,  t̄ = t / L_t,  T̄ = T / T_scale,  k̄ = κ / k_scale
    L_z = 30e-6 m, L_t = 0.1 s, T_scale = 50 K, k_scale = 0.02

归一化 PDE:
    ∂T/∂t = D·∂²T/∂z² + q/(ρCp)
    → (T_scale/L_t)·∂T̄/∂t̄ = (D·T_scale/L_z²)·∂²T̄/∂z̄² + q/(ρCp)
    → ∂T̄/∂t̄ = D̃·∂²T̄/∂z̄² + q̃
    其中 D̃ = D·L_t/L_z², q̃ = (L_t/T_scale)·q/(ρCp)

物理约束链路:
    z → κ_φ(z) → α(z) = 4πκ/(nλ) → I(z) = I₀·exp(-∫αdz) → q(z) = α·I
    全程可微 (torch.autograd)

用法:
    from examples._pinn_nonuniform import (
        TemperatureNetwork, KappaNetwork,
        compute_total_loss, train_pinn,
    )
"""
import numpy as np
import torch
import torch.nn as nn


# ====================================================================
# 归一化常数（与 _finite_difference 默认网格一致）
# ====================================================================

L_Z = 30e-6          # m, 空间特征长度（关注区域 0~30μm）
L_T = 0.1            # s, 时间特征长度（模拟时间 0~0.1s）
T_SCALE = 50.0       # K, 温升特征尺度（量级 ~50K）
K_SCALE = 0.02       # 无量纲，κ 特征尺度（峰值 4κ₀ = 0.02）


# ====================================================================
# 双网络定义
# ====================================================================

class TemperatureNetwork(nn.Module):
    """温度场网络 T_θ(z̄, t̄) → T̄。

    输入: (z̄, t̄) ∈ [0, 1]²
    输出: T̄ (无量纲温升), 实际 T = T_SCALE · T̄

    3 层 × 64 神经元, tanh 激活。
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

        # 小尺度初始化（避免初始输出过大）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, zt):
        """zt shape [N, 2], 列 0 = z̄, 列 1 = t̄。返回 [N, 1]。"""
        return self.net(zt)


class KappaNetwork(nn.Module):
    """消光系数场网络 κ_φ(z̄) → κ。

    输入: z̄ = z/L_Z ∈ [0, 1]
    输出: κ(z)（实际量纲，正值）

    3 层 × 32 神经元, tanh 激活。
    输出: κ = κ₀ · (1 + 3·sigmoid(raw - 5))，保证 κ ∈ [κ₀, 4κ₀]
    初始 raw≈0 → κ ≈ κ₀·1.02 ≈ κ₀ (sigmoid(-5)≈0.0067)
    """

    def __init__(self, hidden_dim=32, num_layers=3, kappa0=0.005, peak_factor=3.0):
        super().__init__()
        self.kappa0 = kappa0
        self.peak_factor = peak_factor  # κ ∈ [κ₀, κ₀·(1+peak_factor)]

        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # 标准初始化（gain=1），让网络对 z 敏感
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_norm):
        """z_norm shape [N, 1]。返回 κ(z) shape [N, 1]（实际量纲，正值）。"""
        raw = self.net(z_norm)
        # κ = κ₀ · (1 + peak_factor · sigmoid(raw - bias))
        # bias=5 让初始 raw≈0 时 sigmoid(-5)≈0.0067 → κ ≈ κ₀·1.02
        # raw=5 时 sigmoid(0)=0.5 → κ = κ₀·2.5
        # raw=10 时 sigmoid(5)≈0.993 → κ ≈ κ₀·4
        bias = 5.0
        k = self.kappa0 * (1.0 + self.peak_factor * torch.sigmoid(raw - bias))
        return k


# ====================================================================
# 可微光强积分 I(z) = I₀·exp(-∫₀ᶻ α(z')dz')
# ====================================================================

def compute_intensity_torch(z, kappa, I0, n, wavelength_nm):
    """非均匀比尔-朗伯光强（torch 可微版）。

    I(z) = I₀ · exp(-cumtrapz(α, z)),  α(z) = 4π κ(z)/(nλ)

    参数:
        z             — shape [N, 1], requires_grad 不要求，按 z 升序排列
        kappa         — shape [N, 1], κ(z)（实际量纲）
        I0            — 标量
        n             — 折射率
        wavelength_nm — 波长 (nm)

    返回: I(z) shape [N, 1]
    """
    lam = wavelength_nm * 1e-9
    alpha = 4.0 * np.pi * kappa / (n * lam)  # [N, 1]

    # 累积梯形积分: optical_depth[i] = ∫₀^{z_i} α(z') dz'
    # 用差分形式: depth[i] = depth[i-1] + 0.5·(α[i-1]+α[i])·(z[i]-z[i-1])
    dz = z[1:] - z[:-1]                          # [N-1, 1]
    avg_alpha = 0.5 * (alpha[1:] + alpha[:-1])   # [N-1, 1]
    integrand = avg_alpha * dz                    # [N-1, 1]

    # cumsum 得到累积积分，前面补 0
    optical_depth = torch.cat([
        torch.zeros_like(z[:1]),
        torch.cumsum(integrand, dim=0),
    ], dim=0)

    return I0 * torch.exp(-optical_depth)


def compute_source_torch(z, kappa, I0, n, wavelength_nm):
    """热源 q(z) = α(z)·I(z)。"""
    lam = wavelength_nm * 1e-9
    alpha = 4.0 * np.pi * kappa / (n * lam)
    I = compute_intensity_torch(z, kappa, I0, n, wavelength_nm)
    return alpha * I


def compute_intensity_torch_unsorted(z, kappa, I0, n, wavelength_nm):
    """对无序 z 输入的包装: 排序 → 积分 → 恢复顺序。"""
    z_sorted, sort_idx = torch.sort(z, dim=0)
    k_sorted = kappa[sort_idx].squeeze(-1).unsqueeze(-1) if kappa.dim() == 2 else kappa[sort_idx]
    # 修正: sort_idx shape [N, 1], 用于索引 [N, 1] 的 kappa
    k_sorted = torch.gather(kappa, 0, sort_idx.expand_as(kappa))
    I_sorted = compute_intensity_torch(z_sorted, k_sorted, I0, n, wavelength_nm)
    # 恢复原始顺序
    inv_idx = torch.argsort(sort_idx, dim=0)
    I_original = torch.gather(I_sorted, 0, inv_idx.expand_as(I_sorted))
    return I_original


# ====================================================================
# 物理残差 r = ∂T/∂t - D·∂²T/∂z² - q/(ρCp)
# ====================================================================

def compute_physics_residual(T_net, k_net, z_phys, t_phys,
                             I0, D, rho, Cp, n, wavelength_nm):
    """计算 PDE 残差 r_phys = ∂T/∂t - D·∂²T/∂z² - q/(ρCp)。

    参数:
        T_net, k_net  — 网络
        z_phys, t_phys — shape [N, 1], requires_grad=True (实际量纲: m, s)

    返回: r_phys shape [N, 1]
    """
    # 归一化输入
    z_norm = z_phys / L_Z
    t_norm = t_phys / L_T

    # 前向 + autograd
    zt = torch.cat([z_norm, t_norm], dim=1)
    T_norm = T_net(zt)  # [N, 1], 归一化 T̄

    # 反归一化得到实际 T
    T = T_norm * T_SCALE

    # 一阶导: ∂T/∂t
    dT_dt = torch.autograd.grad(
        T.sum(), t_phys, create_graph=True, retain_graph=True
    )[0]

    # 一阶导: ∂T/∂z
    dT_dz = torch.autograd.grad(
        T.sum(), z_phys, create_graph=True, retain_graph=True
    )[0]

    # 二阶导: ∂²T/∂z²
    d2T_dz2 = torch.autograd.grad(
        dT_dz.sum(), z_phys, create_graph=True
    )[0]

    # κ(z), α(z), I(z), q(z) — 注意需要对 z 排序做积分
    k = k_net(z_norm)  # [N, 1], 实际量纲 κ
    q = compute_source_torch_unsorted(z_phys, k, I0, n, wavelength_nm)

    # PDE 残差
    r = dT_dt - D * d2T_dz2 - q / (rho * Cp)
    return r


def compute_source_torch_unsorted(z, kappa, I0, n, wavelength_nm):
    """无序 z 输入的热源 q(z) = α(z)·I(z)。"""
    z_sorted, sort_idx = torch.sort(z, dim=0)
    k_sorted = torch.gather(kappa, 0, sort_idx.expand_as(kappa))
    lam = wavelength_nm * 1e-9
    alpha_sorted = 4.0 * np.pi * k_sorted / (n * lam)
    I_sorted = compute_intensity_torch(z_sorted, k_sorted, I0, n, wavelength_nm)
    q_sorted = alpha_sorted * I_sorted
    inv_idx = torch.argsort(sort_idx, dim=0)
    q = torch.gather(q_sorted, 0, inv_idx.expand_as(q_sorted))
    return q


# ====================================================================
# 总损失
# ====================================================================

def compute_total_loss(T_net, k_net,
                       z_obs, t_obs, T_obs,
                       z_col, t_col,
                       I0, D, rho, Cp, n, wavelength_nm,
                       kappa_prior,
                       n_ic=50, n_bc=50, n_prior=50,
                       lambda_data=1.0, lambda_phys=1.0,
                       lambda_ic=1.0, lambda_bc=1.0, lambda_prior=0.01,
                       seed=None):
    """计算 PINN 总损失。

    L = λ_data·L_data + λ_phys·L_phys + λ_ic·L_ic + λ_bc·L_bc + λ_prior·L_prior

    参数:
        z_obs, t_obs, T_obs — 观测点 (实际量纲: m, s, K), shape [N, 1]
        z_col, t_col        — 物理残差配点 (实际量纲: m, s), shape [N, 1]
        kappa_prior         — κ 先验值 (实际量纲, e.g. 0.005)

    返回: loss (标量 tensor)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ===== L_data: 观测点 MSE =====
    z_obs_norm = z_obs / L_Z
    t_obs_norm = t_obs / L_T
    zt_obs = torch.cat([z_obs_norm, t_obs_norm], dim=1)
    T_pred_norm = T_net(zt_obs)
    T_pred = T_pred_norm * T_SCALE
    L_data = torch.mean((T_pred - T_obs) ** 2)

    # ===== L_phys: PDE 残差 =====
    z_col_grad = z_col.detach().clone().requires_grad_(True)
    t_col_grad = t_col.detach().clone().requires_grad_(True)
    r = compute_physics_residual(
        T_net, k_net, z_col_grad, t_col_grad,
        I0, D, rho, Cp, n, wavelength_nm,
    )
    # 归一化残差：除以源项特征尺度 q_scale/(ρCp)
    q_scale = 4.0 * np.pi * 0.02 / (n * wavelength_nm * 1e-9) * I0 / (rho * Cp)
    L_phys = torch.mean((r / q_scale) ** 2)

    # ===== L_ic: T(z, 0) = 0 =====
    z_ic = torch.rand(n_ic, 1, device=z_obs.device) * L_Z
    t_ic = torch.zeros_like(z_ic)
    z_ic_norm = z_ic / L_Z
    t_ic_norm = torch.zeros_like(z_ic)
    zt_ic = torch.cat([z_ic_norm, t_ic_norm], dim=1)
    T_ic_norm = T_net(zt_ic)
    T_ic = T_ic_norm * T_SCALE
    L_ic = torch.mean(T_ic ** 2) / (T_SCALE ** 2)  # 归一化

    # ===== L_bc: ∂T/∂z|_{z=0} = 0 (Neumann) =====
    z_bc = torch.zeros(n_bc, 1, requires_grad=True, device=z_obs.device)
    t_bc = torch.rand(n_bc, 1, device=z_obs.device) * L_T
    z_bc_norm = z_bc / L_Z
    t_bc_norm = t_bc / L_T
    zt_bc = torch.cat([z_bc_norm, t_bc_norm], dim=1)
    T_bc_norm = T_net(zt_bc)
    T_bc = T_bc_norm * T_SCALE
    dT_dz_bc = torch.autograd.grad(
        T_bc.sum(), z_bc, create_graph=True
    )[0]
    # 归一化: dT/dz 量级 T_SCALE/L_Z
    L_bc = torch.mean((dT_dz_bc / (T_SCALE / L_Z)) ** 2)

    # ===== L_prior: κ(z) 接近 κ₀ =====
    z_prior = torch.rand(n_prior, 1, device=z_obs.device) * 1.0  # 归一化 [0, 1]
    k_prior_pred = k_net(z_prior)
    L_prior = torch.mean(((k_prior_pred - kappa_prior) / K_SCALE) ** 2)

    loss = (lambda_data * L_data + lambda_phys * L_phys
            + lambda_ic * L_ic + lambda_bc * L_bc + lambda_prior * L_prior)

    # 缓存各项用于诊断
    loss._components = {
        "data": L_data.item(),
        "phys": L_phys.item(),
        "ic": L_ic.item(),
        "bc": L_bc.item(),
        "prior": L_prior.item(),
    }
    return loss


# ====================================================================
# 训练循环
# ====================================================================

def train_pinn(T_net, k_net, z_obs, t_obs, T_obs,
               I0, D, rho, Cp, n, wavelength_nm, kappa_prior,
               steps=5000, lr=1e-3, n_col=256, seed=None,
               lambda_data=1.0, lambda_phys=1.0,
               lambda_ic=1.0, lambda_bc=1.0, lambda_prior=0.01,
               verbose=True, verbose_every=500,
               lr_schedule="cosine", lr_min=1e-5,
               phys_warmup_steps=0, phys_warmup_max=None,
               k_lr_multiplier=100.0):
    """PINN 训练循环（带学习率调度和物理损失 warmup）。

    参数:
        lr_schedule — "cosine" 余弦退火, "constant" 恒定
        lr_min      — 余弦退火最小学习率
        phys_warmup_steps — 前多少步逐步增加 lambda_phys（从 0 到 lambda_phys）
        phys_warmup_max   — warmup 结束时的 lambda_phys 值（默认=lambda_phys）
        k_lr_multiplier   — κ 网络学习率乘数（克服梯度消失，默认 100×）

    返回:
        history: list[float], 每步总 loss
        components_history: list[dict], 每步各 loss 分量
    """
    import math

    if seed is not None:
        torch.manual_seed(seed)

    if phys_warmup_max is None:
        phys_warmup_max = lambda_phys

    # κ 网络单独更高学习率（克服 κ→α→I→q→T 链路的梯度衰减）
    params = [
        {"params": T_net.parameters(), "lr": lr},
        {"params": k_net.parameters(), "lr": lr * k_lr_multiplier},
    ]
    optimizer = torch.optim.Adam(params)

    history = []
    components_history = []

    for step in range(steps):
        # 学习率调度（保持 κ 网络的乘数比例）
        if lr_schedule == "cosine" and steps > 1:
            lr_t = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / (steps - 1)))
            optimizer.param_groups[0]["lr"] = lr_t
            optimizer.param_groups[1]["lr"] = lr_t * k_lr_multiplier
        elif lr_schedule == "constant":
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * k_lr_multiplier

        # 物理 warmup
        if phys_warmup_steps > 0 and step < phys_warmup_steps:
            w = step / phys_warmup_steps
            lambda_phys_t = w * phys_warmup_max
        else:
            lambda_phys_t = lambda_phys

        # 每步重新采样配点
        z_col = torch.rand(n_col, 1) * L_Z
        t_col = torch.rand(n_col, 1) * L_T

        optimizer.zero_grad()
        loss = compute_total_loss(
            T_net, k_net,
            z_obs, t_obs, T_obs,
            z_col, t_col,
            I0, D, rho, Cp, n, wavelength_nm,
            kappa_prior,
            lambda_data=lambda_data, lambda_phys=lambda_phys_t,
            lambda_ic=lambda_ic, lambda_bc=lambda_bc,
            lambda_prior=lambda_prior,
        )
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        components_history.append(loss._components)

        if verbose and (step % verbose_every == 0 or step == steps - 1):
            c = loss._components
            print(f"  step {step:5d}: total={loss.item():.4e}  "
                  f"data={c['data']:.2e} phys={c['phys']:.2e} "
                  f"ic={c['ic']:.2e} bc={c['bc']:.2e} prior={c['prior']:.2e}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} λ_phys={lambda_phys_t:.1f}")

    return history, components_history
