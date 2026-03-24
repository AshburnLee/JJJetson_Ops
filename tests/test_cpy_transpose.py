import torch
import numpy as np
import cpy_transpose_me
import utils 


def test_transpose():
    """
    测试 cpy_transpose kernel 在连续存储情况下的正确性
    """
    torch.manual_seed(24)
    
    shape = (2, 3, 4)
    dst_shape = (shape[0], shape[2], shape[1])
    src_torch = torch.randint(0, 50, shape, dtype=torch.float32)
    
    ## col-major data
    # src_np = np.asfortranarray(src_torch.numpy())
    # dst_np = np.zeros(dst_shape, dtype=np.float32, order="F")
    ## row-major data
    src_np = np.asarray(src_torch.numpy())
    dst_np = np.zeros(dst_shape, dtype=np.float32)
    
    # gt
    dst_torch = torch.transpose(src_torch, 1, 2)
    # me
    cpy_transpose_me.cpy_trans(src_np, dst_np, list(src_np.shape))
    dst_me = torch.from_numpy(dst_np)
    
    print("src.shape:\n", src_torch.shape)
    print("dst_torch shape:\n", dst_torch.shape)
    print("dst_me.shape: \n", dst_me.shape)
    print("src_torch: \n", src_torch)
    print("dst_torch: \n", dst_torch)
    print("dst_me: \n", dst_me)
    
    ok = utils.compare_np_torch(dst_np, dst_torch, atol=1e-3, rtol=1e-3)
    print("Passed" if ok else "Failed")
    

if __name__ == "__main__":
    test_transpose()
