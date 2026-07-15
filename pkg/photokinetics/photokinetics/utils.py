"""Photokinetics 工具函数。"""
import torch


def fmt_sci(value, sig=4):
    """将数值格式化为科学计数法字符串。"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if value == 0:
        return "0"
    return "{:.{}e}".format(value, sig - 1)


def fmt_fix(value, sig=4):
    """将数值格式化为定点数。"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if abs(value) < 1e-4 or abs(value) >= 1e6:
        return fmt_sci(value, sig)
    return "{:.{}g}".format(value, sig)


def to_tensor(x, dtype=torch.float32, device=None):
    """将输入转换为 torch.Tensor。"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)
