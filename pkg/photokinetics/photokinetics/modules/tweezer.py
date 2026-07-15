"""光镊力模块（瑞利近似）。"""
import torch
from photokinetics.constants import C


def calc_tweezer_force(a_m, n_particle, n_medium, wavelength_nm, grad_I):
    """
    光镊力计算（瑞利近似）。

    公式:
        α = 4πa³ (m²-1)/(m²+2)     极化率, m = n_particle/n_medium
        F_grad = n_medium · α · ∇I / c    梯度力
        F_scat = n_medium · α · ∇I / (2c)  散射力（近似）

    参数:
        a_m           — 颗粒半径 (m), scalar or tensor
        n_particle    — 颗粒折射率, scalar or tensor
        n_medium      — 介质折射率, scalar or tensor
        wavelength_nm — 波长 (nm), scalar or tensor
        grad_I        — 光强梯度 (W/m³), scalar or tensor

    返回: dict with keys: alpha, F_grad, F_scat, F_total
    """
    a = torch.as_tensor(a_m, dtype=torch.float32)
    n_p = torch.as_tensor(n_particle, dtype=torch.float32)
    n_m = torch.as_tensor(n_medium, dtype=torch.float32)
    grad_I = torch.as_tensor(grad_I, dtype=torch.float32)

    m = n_p / n_m
    m2 = m**2

    alpha = 4.0 * torch.pi * a**3 * (m2 - 1.0) / (m2 + 2.0)
    F_grad = n_m * alpha * grad_I / C
    F_scat = n_m * alpha * grad_I / (2.0 * C)
    F_total = F_grad + F_scat

    return {'alpha': alpha, 'F_grad': F_grad, 'F_scat': F_scat, 'F_total': F_total}
