import torch

# CODATA 2018 推荐值
H         = torch.tensor(6.62607015e-34)     # 普朗克常数 (J·s) — 精确值
H_EV      = torch.tensor(4.135667696e-15)    # 普朗克常数 (eV·s)
C         = torch.tensor(2.99792458e8)       # 真空光速 (m/s) — 精确值
K_B       = torch.tensor(1.380649e-23)       # 玻尔兹曼常数 (J/K) — 精确值
SIGMA     = torch.tensor(5.670374419e-8)     # 斯特藩-玻尔兹曼常数 W/(m²·K⁴)
B_WIEN    = torch.tensor(2.897771955e-3)     # 维恩位移常数 (m·K)
E_C       = torch.tensor(1.602176634e-19)    # 电子电荷 (C) — 精确值
M_E       = torch.tensor(9.1093837015e-31)   # 电子静止质量 (kg)
EPSILON_0 = torch.tensor(8.8541878128e-12)   # 真空介电常数 (F/m)
G         = torch.tensor(6.67430e-11)        # 万有引力常数 m³/(kg·s²)
M_EARTH   = torch.tensor(5.9722e24)          # 地球质量 (kg)
R_EARTH   = torch.tensor(6.371e6)            # 地球平均半径 (m)

# 常用组合常数
HC_EV_NM  = torch.tensor(1239.841984)        # hc (eV·nm)
LAMBDA_C  = torch.tensor(2.4263102389e-12)   # 电子康普顿波长 λ_c = h/(m_e·c) (m)
