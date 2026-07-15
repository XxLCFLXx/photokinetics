"""多普勒效应模块。"""
import torch
from photokinetics.constants import C


def calc_doppler(nu0, v_km_s, receding=True):
    """
    多普勒效应计算（同时给出经典与相对论结果）。

    公式:
        β = v/c
        经典:  ν_obs = ν₀(1 ± v/c)  (+靠近, −远离)
        相对论:  ν_obs = ν₀ √((1∓β)/(1±β))  (+靠近用+, −远离用−)
        红移参数:  z = ν₀/ν_obs − 1 (天文标准定义, 远离为正)

    参数:
        nu0      — 光源静止频率 (Hz), scalar or tensor
        v_km_s   — 光源速度 (km/s), scalar or tensor
        receding — True=远离, False=靠近

    返回: (nu_classical, nu_relativistic, z_classical, z_relativistic)
    """
    nu0 = torch.as_tensor(nu0, dtype=torch.float32)
    v = torch.as_tensor(v_km_s, dtype=torch.float32) * 1e3
    beta = v / C

    if receding:
        nu_cl = nu0 * (1.0 - beta)
        nu_rel = nu0 * torch.sqrt((1.0 - beta) / (1.0 + beta))
    else:
        nu_cl = nu0 * (1.0 + beta)
        nu_rel = nu0 * torch.sqrt((1.0 + beta) / (1.0 - beta))

    z_cl = (nu_cl - nu0) / nu0

    if receding:
        z_rel = (nu0 - nu_rel) / nu_rel
    else:
        z_rel = (nu_rel - nu0) / nu0

    return nu_cl, nu_rel, z_cl, z_rel
