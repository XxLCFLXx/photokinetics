"""
Photokinetics V2.0 — 可微光学物理引擎 (Differentiable Optics Engine)

基于光子动能传递的光学统一计算框架，使用 PyTorch 实现，天然支持自动微分。

8个核心模块:
    - 光电效应 (photoelectric)
    - 黑体辐射 (blackbody)
    - 康普顿散射 (compton)
    - 多普勒效应 (doppler)
    - 引力红移 (gravitational)
    - 光热模型 (photothermal)  — 核心应用
    - 非线性光学 (nonlinear)
    - 光镊力 (tweezer)

快速开始:
    >>> import torch
    >>> from photokinetics import calc_photothermal
    >>> I0 = torch.tensor(1e7, requires_grad=True)
    >>> result = calc_photothermal(1.33, 0.00012, 1064, I0, 1000, 4186, 1.0, 1.0)
    >>> result['dT'].backward()
    >>> print(I0.grad)  # d(ΔT)/d(I₀)
"""

from photokinetics.constants import *
from photokinetics.utils import *
from photokinetics.modules.photoelectric import calc_photoelectric
from photokinetics.modules.blackbody import calc_blackbody
from photokinetics.modules.compton import calc_compton
from photokinetics.modules.doppler import calc_doppler
from photokinetics.modules.gravitational import calc_gravitational_redshift
from photokinetics.modules.photothermal import calc_photothermal
from photokinetics.modules.time_resolved import (
    calc_photothermal_timed,
    calc_photothermal_auto,
)
from photokinetics.modules.nonlinear import calc_nonlinear_order
from photokinetics.modules.tweezer import calc_tweezer_force

__version__ = "2.1.0"
__description__ = "Photokinetics V2.1 - Differentiable optics engine with time-resolved photothermal model"
__author__ = "Cogito Lin"
__email__ = "noreply@example.com"
__license__ = "MIT"
