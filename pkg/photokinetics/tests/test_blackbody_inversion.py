"""案例2测试：黑体光谱测温。"""
import torch
import pytest
from examples.fit_blackbody_temperature import (
    make_observations,
    forward,
    run_inversion,
)


def test_make_observations_shape():
    """合成光谱形状正确。"""
    wavelengths, I_obs, params_true = make_observations(seed=42, noise=0.0)
    assert len(wavelengths) == 32
    assert I_obs.shape == (32,)
    assert params_true['T'] == pytest.approx(5800.0, abs=0.1)
    assert params_true['gain'] == pytest.approx(1.0, abs=1e-6)


def test_make_observations_reproducible():
    """固定种子可复现。"""
    _, I1, _ = make_observations(seed=42, noise=0.02)
    _, I2, _ = make_observations(seed=42, noise=0.02)
    assert torch.allclose(I1, I2)


def test_forward_gradient_available():
    """前向梯度可用。"""
    raw_T = torch.nn.Parameter(torch.tensor(2.0))
    wavelengths, I_obs, params_true = make_observations(seed=42, noise=0.0)
    gain = params_true['gain']

    loss = forward(raw_T, gain, wavelengths, I_obs)
    loss.backward()

    assert raw_T.grad is not None
    assert torch.isfinite(raw_T.grad).all()


def test_run_inversion_noiseless_recovery():
    """无噪声下温度恢复误差 <1%。"""
    result = run_inversion(seed=42, noise=0.0, steps=300, lr=0.1)
    T_err = abs(result['T_fit'] - result['T_true']) / result['T_true']
    assert T_err < 0.01, f"温度误差 {T_err:.2%} 超过 1%"


def test_run_inversion_loss_decreases():
    """优化后 loss 下降。"""
    result = run_inversion(seed=42, noise=0.02, steps=200, lr=0.1)
    assert result['final_loss'] < result['initial_loss']


def test_run_inversion_noisy_recovery():
    """2% 噪声下温度恢复误差 <3%。"""
    result = run_inversion(seed=42, noise=0.02, steps=300, lr=0.1)
    T_err = abs(result['T_fit'] - result['T_true']) / result['T_true']
    assert T_err < 0.03, f"温度误差 {T_err:.2%} 超过 3%"
