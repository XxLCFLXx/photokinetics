"""光热模型模块（光动论四步解析模型）。"""
import torch
from photokinetics.constants import C


def calc_photothermal(n, kappa, wavelength_nm, I0, rho, Cp, depth_mm, time_s):
    """
    光动论四步光热解析模型（绝热近似）。

    公式:
        Step 1: α = 4πκ / (nλ)           吸收系数
        Step 2: I(z) = I₀·exp(-αz)       光强衰减
        Step 3: q(z) = α·I(z)            热源密度
        Step 4: ΔT = q·t / (ρCp)         温升（绝热近似）

    参数:
        n              — 折射率, scalar or tensor
        kappa          — 消光系数, scalar or tensor
        wavelength_nm  — 波长 (nm), scalar or tensor
        I0             — 入射光强 (W/m²), scalar or tensor
        rho            — 密度 (kg/m³), scalar or tensor
        Cp             — 比热 (J/(kg·K)), scalar or tensor
        depth_mm       — 深度 (mm), scalar or tensor
        time_s         — 照射时间 (s), scalar or tensor

    返回: dict with keys: alpha, I_z, q_z, dT
    """
    n = torch.as_tensor(n, dtype=torch.float32)
    kappa = torch.as_tensor(kappa, dtype=torch.float32)
    lam = torch.as_tensor(wavelength_nm, dtype=torch.float32) * 1e-9
    I0 = torch.as_tensor(I0, dtype=torch.float32)
    rho = torch.as_tensor(rho, dtype=torch.float32)
    Cp = torch.as_tensor(Cp, dtype=torch.float32)
    z = torch.as_tensor(depth_mm, dtype=torch.float32) * 1e-3
    time_s = torch.as_tensor(time_s, dtype=torch.float32)

    alpha = 4.0 * torch.pi * kappa / (n * lam)
    I_z = I0 * torch.exp(-alpha * z)
    q_z = alpha * I_z
    dT = q_z * time_s / (rho * Cp)

    return {'alpha': alpha, 'I_z': I_z, 'q_z': q_z, 'dT': dT}
