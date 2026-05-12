import numpy as np
import torch

"""
逐元素比较任意维度的 NumPy 数组和 Torch 张量
"""


def compare_np_torch(np_arr: np.ndarray, torch_tensor: torch.Tensor, atol=1e-6, rtol=1e-5):
    """
    逐元素比较任意维度的 NumPy 数组和 Torch 张量。
    打印最大绝对误差和相对误差，如有不一致则返回 False。
    """
    t_np = torch_tensor.detach().cpu().numpy()
    if np_arr.shape != t_np.shape:
        print(f"Shape mismatch: np {np_arr.shape} vs torch {t_np.shape}")
        return False

    diff = np_arr - t_np
    abs_diff = np.abs(diff)
    max_abs = abs_diff.max()
    # 避免除以 0
    denom = np.maximum(np.abs(t_np), 1e-12)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max()

    print(f"max_abs_diff = {max_abs:.6e}, max_rel_diff = {max_rel:.6e}")
    ok = np.allclose(np_arr, t_np, atol=atol, rtol=rtol)
    print(f"allclose = {ok} (atol={atol}, rtol={rtol})")
    return ok
