import numpy as np
import torch
import roll_me
import utils
import os

def test_roll():
    shape=(2048,256,1,1)
    # col-major
    src = np.asfortranarray(np.random.randint(0, 10, size=(shape[0], shape[1], shape[2],shape[3])).astype(np.float32))
    dst = np.zeros((shape[0], shape[1], shape[2],shape[3]), dtype=np.float32, order="F")
    src_ref = torch.tensor(src)

    # dims 和 shifts 按 col-major 顺序传入：
    # dims = [ne0_0, ne0_1, ne0_2, ne0_3]，shift 对应每一维的 roll
    roll_me.roll(src, dst, [shape[0], shape[1], shape[2],shape[3]], [-1, -2, 0, 0])
    dst_ref = torch.roll(src_ref, shifts=(-1, -2), dims=(0, 1))
    debug_mode = os.environ.get("DEBUG_MY_OPS", "") == "1"
    if debug_mode:
        print("src: \n", src)
        print("output: \n", dst)
        print("target: \n", dst_ref)

    # 逐元素比较
    ok = utils.compare_np_torch(dst, dst_ref)
    if ok:
        print("Passed")
    else:
        print("Failed")


if __name__ == "__main__":
    test_roll()
