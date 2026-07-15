"""案例3测试：光镊鲁棒粒径设计。"""
import torch
import pytest
from examples.design_optical_tweezer import (
    PARTICLE_RADII,
    run_design,
    forward,
)


def test_particle_radii_in_rayleigh_regime():
    """粒径在瑞利近似域内（a ≪ λ）。"""
    wavelength_nm = 1064.0
    for a in PARTICLE_RADII:
        assert (a * 1e6) / wavelength_nm < 0.1, \
            f"粒径 {a*1e6}μm 不满足瑞利近似（λ={wavelength_nm}nm）"


def test_forward_gradient_available():
    """前向梯度可用。"""
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))
    loss = forward(raw_grad_I)
    loss.backward()

    assert raw_grad_I.grad is not None
    assert torch.isfinite(raw_grad_I.grad).all()


def test_forward_multi_objective():
    """多目标损失包含跟踪+鲁棒+功率三项。"""
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))
    result = forward(raw_grad_I, return_components=True)

    assert 'L_track' in result
    assert 'L_robust' in result
    assert 'L_power' in result
    assert 'L_total' in result


def test_run_design_loss_decreases():
    """优化后 loss 下降。"""
    result = run_design(steps=200, lr=0.05)
    assert result['final_loss'] < result['initial_loss']


def test_run_design_target_force_achieved():
    """优化后平均力达到目标（误差 <5%）。"""
    result = run_design(steps=200, lr=0.05)
    F_target = 10e-12  # 10 pN
    F_mean = result['forces_final'].mean().item()
    err = abs(F_mean - F_target) / F_target
    assert err < 0.05, f"目标力误差 {err:.2%} 超过 5%"


def test_run_design_minimum_force_satisfied():
    """优化后最弱粒径力 ≥ 下限。"""
    result = run_design(steps=200, lr=0.05)
    F_min_limit = 5e-12  # 5 pN
    F_weakest = result['forces_final'].min().item()
    assert F_weakest >= F_min_limit * 0.9, \
        f"最弱粒径力 {F_weakest*1e12:.2f} pN 低于下限 {F_min_limit*1e12:.2f} pN（允许10%裕量）"
