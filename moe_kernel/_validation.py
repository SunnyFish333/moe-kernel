"""共用的张量检查，只读形状、布局和设备。"""

import torch


ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)


def check_tensor(tensor, name, *, ndim, dtypes, device=None):
    if tensor.ndim != ndim or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} 必须是 {ndim} 维连续 CUDA 张量")
    if tensor.dtype not in dtypes:
        raise ValueError(f"{name} 的 dtype 必须是 {dtypes}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} 必须在 {device} 上")
