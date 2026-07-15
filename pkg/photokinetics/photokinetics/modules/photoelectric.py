"""光电效应模块。"""
import torch
from photokinetics.constants import HC_EV_NM


def calc_photoelectric(phi_ev, lambda_nm):
    """
    光电效应计算（爱因斯坦光电方程）。

    公式:
        hν  = hc / λ           光子能量
        E_k = hν - Φ           最大动能（发生光电效应时）
        V_s = E_k / e          遏止电压（数值上等于 E_k(eV)）

    参数:
        phi_ev    — 材料逸出功 Φ (eV), scalar or tensor
        lambda_nm — 入射波长 λ (nm), scalar or tensor

    返回: (hν, occurs, E_k, V_s) — 全部为 torch.Tensor
    """
    phi = torch.as_tensor(phi_ev, dtype=torch.float32)
    lam = torch.as_tensor(lambda_nm, dtype=torch.float32)

    hv = HC_EV_NM / lam
    occurs = hv > phi
    ek = torch.where(occurs, hv - phi, torch.tensor(0.0))
    vs = ek

    return hv, occurs, vs, ek
