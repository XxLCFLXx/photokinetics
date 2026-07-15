"""案例1测试：含时光热参数反演。"""
import torch
import pytest
from examples.fit_transient_photothermal import (
    make_observations,
    forward,
    run_inversion,
)


def test_make_observations_shape():
    """合成观测数据形状正确。"""
    z_grid, t_grid, T_obs, params_true = make_observations(seed=42, noise=0.0)
    assert T_obs.shape[0] == 10
    assert T_obs.shape[1] == 10
    assert 'kappa' in params_true
    assert 'D' in params_true
    assert params_true['kappa'] == pytest.approx(0.044, abs=1e-6)
    assert params_true['D'] == pytest.approx(9.08e-5, abs=1e-10)


def test_make_observations_reproducible():
    """固定种子可复现。"""
    _, _, T1, _ = make_observations(seed=42, noise=0.01)
    _, _, T2, _ = make_observations(seed=42, noise=0.01)
    assert torch.allclose(T1, T2)


def test_forward_gradient_available():
    """前向计算梯度可用。"""
    raw_kappa = torch.nn.Parameter(torch.tensor(-2.0))
    raw_D = torch.nn.Parameter(torch.tensor(-10.0))
    z_grid, t_grid, T_obs, _ = make_observations(seed=42, noise=0.0)

    loss = forward(raw_kappa, raw_D, z_grid, t_grid, T_obs)
    loss.backward()

    assert raw_kappa.grad is not None
    assert torch.isfinite(raw_kappa.grad).all()
    assert raw_D.grad is not None
    assert torch.isfinite(raw_D.grad).all()


def test_run_inversion_noiseless_recovery():
    """无噪声下参数恢复误差 <5%。"""
    result = run_inversion(seed=42, noise=0.0, steps=500, lr=0.05)
    kappa_err = abs(result['kappa_fit'] - result['kappa_true']) / result['kappa_true']
    D_err = abs(result['D_fit'] - result['D_true']) / result['D_true']
    assert kappa_err < 0.05, f"κ 误差 {kappa_err:.2%} 超过 5%"
    assert D_err < 0.05, f"D 误差 {D_err:.2%} 超过 5%"


def test_run_inversion_loss_decreases():
    """优化后 loss 下降。"""
    result = run_inversion(seed=42, noise=0.01, steps=300, lr=0.05)
    assert result['final_loss'] < result['initial_loss']


def test_run_inversion_noisy_recovery():
    """1% 噪声下参数恢复误差 <10%。"""
    result = run_inversion(seed=42, noise=0.01, steps=500, lr=0.05)
    kappa_err = abs(result['kappa_fit'] - result['kappa_true']) / result['kappa_true']
    D_err = abs(result['D_fit'] - result['D_true']) / result['D_true']
    assert kappa_err < 0.10, f"κ 误差 {kappa_err:.2%} 超过 10%"
    assert D_err < 0.10, f"D 误差 {D_err:.2%} 超过 10%"
