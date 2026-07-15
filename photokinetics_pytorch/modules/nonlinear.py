import torch
from ..constants import HC_EV_NM


def calc_nonlinear_order(Eg_eV, wavelength_nm):
    """
    非线性光学多光子吸收阶数判定。
    
    公式:
        N = ceil(Eg / hν)
        其中 hν = hc/λ (eV)
    
    参数:
        Eg_eV         — 禁带宽度 (eV), scalar or tensor
        wavelength_nm — 入射波长 (nm), scalar or tensor
    
    返回: (N, hv_eV) — 吸收阶数和光子能量
    """
    Eg = torch.as_tensor(Eg_eV, dtype=torch.float32)
    lam = torch.as_tensor(wavelength_nm, dtype=torch.float32)
    
    hv_eV = HC_EV_NM / lam
    
    N = torch.ceil(Eg / hv_eV)
    N = torch.clamp(N, min=1)
    
    return N, hv_eV
