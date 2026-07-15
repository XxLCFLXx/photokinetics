"""引力红移模块。"""
import torch
from photokinetics.constants import G, C


def calc_gravitational_redshift(M, r1, r2):
    """
    引力红移计算（广义相对论）。

    公式:
        Δν/ν₀ = GM(1/r₁ - 1/r₂)/c²

    参数:
        M   — 质量 (kg), scalar or tensor
        r1  — 发射半径 (m), scalar or tensor
        r2  — 观测半径 (m), scalar or tensor

    返回: (delta_nu_over_nu0, z) — 相对频移和红移参数
    """
    M = torch.as_tensor(M, dtype=torch.float32)
    r1 = torch.as_tensor(r1, dtype=torch.float32)
    r2 = torch.as_tensor(r2, dtype=torch.float32)

    delta_nu_over_nu0 = G * M * (1.0 / r1 - 1.0 / r2) / C**2
    z = delta_nu_over_nu0

    return delta_nu_over_nu0, z
