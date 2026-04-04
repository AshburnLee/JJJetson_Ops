import torch
import numpy as np
import cpy_transpose_me
import utils


def _to_f_numpy(x):
    """保证 NumPy 视图为列主序，便于按内存顺序打印。"""
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return a if a.flags.f_contiguous else np.asfortranarray(a.copy())

def print_col_major_3d(x, title):
    a = _to_f_numpy(x)
    b = a.shape[2]
    print(f"\n{title}: \n")
    for k in range(b):
        print(f"  {a[:, :, k]}")

def test_transpose():
    """
    测试 cpy_transpose：逻辑形状 (dim0, dim1, batch)，列主序存储；kernel 交换 dim0 与 dim1，
    batch 维 [2] 不变。Torch 侧经 from_numpy(F) 得到列主序 stride。
    """
    torch.manual_seed(24)
    # baseline (256,128,4)
    shape = (256,128,4)  # (dim0, dim1, batch)
    dst_shape = (shape[1], shape[0], shape[2])  # (dim1, dim0, batch)

    # 先用 C 连续张量固定随机序列，再转为列主序 NumPy，再交还给 Torch
    tmp_c = torch.randint(0, 50, shape, dtype=torch.float32)
    src_torch = torch.from_numpy(np.asfortranarray(tmp_c.numpy()))

    dst_c = torch.transpose(tmp_c, 0, 1)
    dst_torch = torch.from_numpy(np.asfortranarray(dst_c.numpy()))

    src_np = src_torch.numpy()
    dst_np = np.zeros(dst_shape, dtype=np.float32, order="F")

    cpy_transpose_me.cpy_trans(src_np, dst_np, list(src_np.shape))
    dst_me = torch.from_numpy(dst_np)
    
    
    print_col_major_3d(src_torch, "src")
    print_col_major_3d(dst_torch, "dst_torch")
    print_col_major_3d(dst_me, "dst_me")
    
    ok = utils.compare_np_torch(dst_np, dst_torch, atol=1e-3, rtol=1e-3)
    print("Passed" if ok else "Failed")
    

if __name__ == "__main__":
    test_transpose()
