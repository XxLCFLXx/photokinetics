"""Photokinetics 实战案例共享工具层。

提供正值参数化、损失函数和通用优化循环，
被三个实战案例脚本复用。
"""
import torch


def positive_parameter(raw, scale=1.0, floor=0.0):
    """
    通过 softplus 变换保证参数始终为正。

    参数:
        raw    — 原始参数（torch.nn.Parameter，无约束）
        scale  — 缩放系数
        floor  — 下限（保证 val >= floor）

    返回: 正值张量，保持梯度
    """
    return floor + scale * torch.nn.functional.softplus(raw)


def relative_mse(prediction, target, eps=1e-12):
    """
    相对误差 MSE，避免高值主导。

    L = mean(((pred - target) / clamp(|target|, min=eps))²)
    """
    scale = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(((prediction - target) / scale) ** 2)


def log_mse(prediction, target, eps=1e-30):
    """
    对数域 MSE，适用于跨数量级数据（如 Planck 光谱）。

    L = mean((log(pred + eps) - log(target + eps))²)
    """
    return torch.mean((torch.log(prediction + eps) - torch.log(target + eps)) ** 2)


def optimize(parameters, closure, steps, lr, callback=None):
    """
    Adam 优化循环，返回每步 loss 历史。

    参数:
        parameters — torch.nn.Parameter 列表
        closure    — 无参函数，返回 loss 张量
        steps      — 优化步数
        lr         — 学习率
        callback   — 可选，每步调用 callback(step, loss)

    返回: list[float]，每步 loss 值
    """
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = closure()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if callback is not None:
            callback(step, loss)
    return history
