import torch
from ..constants import H, C, E_C, LAMBDA_C


def calc_compton(E0_keV, theta_deg):
    """
    康普顿散射计算。
    
    公式:
        波长位移:  Δλ = λ_c (1 − cosθ),  λ_c = h/(m_e·c) ≈ 2.426 pm
        入射波长:  λ₀ = hc/E₀
        散射波长:  λ' = λ₀ + Δλ
        散射光子能量:  E' = hc/λ'
        电子反冲动能:  E_e = E₀ − E'
    
    参数:
        E0_keV    — 入射光子能量 (keV), scalar or tensor
        theta_deg — 散射角 (度), scalar or tensor
    
    返回: (delta_lambda_pm, E_prime_keV, E_electron_keV) — 全部为 torch.Tensor
    """
    E0 = torch.as_tensor(E0_keV, dtype=torch.float32)
    theta = torch.as_tensor(theta_deg, dtype=torch.float32)
    
    theta_rad = torch.deg2rad(theta)
    
    delta_lambda = LAMBDA_C * (1.0 - torch.cos(theta_rad))
    delta_lambda_pm = delta_lambda * 1e12
    
    E0_J = E0 * 1e3 * E_C
    lambda_0 = H * C / E0_J
    
    lambda_prime = lambda_0 + delta_lambda
    E_prime_J = H * C / lambda_prime
    E_prime_keV = E_prime_J / (1e3 * E_C)
    
    E_electron_keV = E0 - E_prime_keV
    
    return delta_lambda_pm, E_prime_keV, E_electron_keV
