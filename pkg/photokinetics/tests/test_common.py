"""共享工具层测试。"""
import torch
import pytest
from examples._common import (
    positive_parameter,
    relative_mse,
    log_mse,
    optimize,
)


def test_positive_parameter_basic():
    """softplus 变换保证正值。"""
    raw = torch.nn.Parameter(torch.tensor(-5.0))
    val = positive_parameter(raw, scale=1.0, floor=0.0)
    assert val.item() > 0
    assert val.requires_grad


def test_positive_parameter_floor():
    """floor 参数生效。"""
    raw = torch.nn.Parameter(torch.tensor(-100.0))
    val = positive_parameter(raw, scale=1.0, floor=10.0)
    assert val.item() >= 10.0


def test_positive_parameter_scale():
    """scale 参数缩放。"""
    raw = torch.nn.Parameter(torch.tensor(0.0))
    val = positive_parameter(raw, scale=1e6, floor=0.0)
    assert 690000 < val.item() < 700000


def test_relative_mse_uniform():
    """均匀误差时 relative_mse 等于普通 mse。"""
    pred = torch.tensor([1.1, 2.2, 3.3])
    target = torch.tensor([1.0, 2.0, 3.0])
    result = relative_mse(pred, target)
    assert abs(result.item() - 0.01) < 1e-6


def test_relative_mse_avoids_high_value_dominance():
    """相对 MSE 能放大小值的大相对误差。"""
    # 第一点：绝对误差1，相对误差1%；第二点：绝对误差9，相对误差90%
    pred = torch.tensor([101.0, 1.0])
    target = torch.tensor([100.0, 10.0])
    result = relative_mse(pred, target)
    # 相对 MSE = (0.0001 + 0.81)/2 ≈ 0.405，小值误差被放大
    assert result.item() > 0.1
    # 对比直接 MSE = 41，高值主导
    direct_mse = torch.mean((pred - target) ** 2)
    assert result.item() < direct_mse.item()


def test_log_mse_cross_magnitude():
    """跨数量级数据用 log_mse。"""
    pred = torch.tensor([1e3, 1e-3])
    target = torch.tensor([1e3, 1e-3])
    result = log_mse(pred, target)
    assert result.item() < 1e-10


def test_log_mse_relative_error():
    """log_mse 等价于对数域相对误差。"""
    pred = torch.tensor([110.0])
    target = torch.tensor([100.0])
    result = log_mse(pred, target)
    expected = (torch.log(torch.tensor(110.0)) - torch.log(torch.tensor(100.0))) ** 2
    assert abs(result.item() - expected.item()) < 1e-6


def test_optimize_basic():
    """优化循环能降低 loss。"""
    x = torch.nn.Parameter(torch.tensor(5.0))
    target = torch.tensor(1.0)

    def closure():
        return (x - target) ** 2

    history = optimize([x], closure, steps=100, lr=0.1)
    assert history[-1] < history[0]
    assert abs(x.item() - 1.0) < 0.1


def test_optimize_callback():
    """callback 被调用。"""
    x = torch.nn.Parameter(torch.tensor(5.0))
    calls = []

    def closure():
        return x ** 2

    def callback(step, loss):
        calls.append((step, loss.item()))

    optimize([x], closure, steps=10, lr=0.1, callback=callback)
    assert len(calls) == 10
    assert calls[0][0] == 0
    assert calls[-1][0] == 9
