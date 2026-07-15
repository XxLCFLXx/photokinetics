"""黑体辐射模块。"""
import torch
from photokinetics.constants import B_WIEN, SIGMA, H, C, K_B


def calc_blackbody(T, lambda_nm=None):
    """
    黑体辐射计算。

    公式:
        维恩位移定律:  λ_max·T = b  →  λ_max = b/T
        斯特藩-玻尔兹曼定律:  j = σ·T⁴
        普朗克公式(光谱辐射出射度):
            B(λ,T) = (2πhc²/λ⁵) / [exp(hc/(λkT)) − 1]

    参数:
        T          — 黑体温度 (K), scalar or tensor
        lambda_nm  — 指定波长 (nm), scalar or tensor, optional

    返回: (λ_max, j, B_lambda) — 全部为 torch.Tensor
    """
    T = torch.as_tensor(T, dtype=torch.float32)

    lambda_max_m = B_WIEN / T
    lambda_max_nm = lambda_max_m * 1e9

    j = SIGMA * T**4

    B_lambda = None
    if lambda_nm is not None:
        lam = torch.as_tensor(lambda_nm, dtype=torch.float32) * 1e-9
        exponent = H * C / (lam * K_B * T)
        safe_exp = torch.where(exponent > 500, torch.tensor(500.0), exponent)
        numerator = 2.0 * torch.pi * H * C**2 / lam**5
        denominator = torch.exp(safe_exp) - 1.0
        B_lambda = numerator / denominator

    return lambda_max_nm, j, B_lambda
