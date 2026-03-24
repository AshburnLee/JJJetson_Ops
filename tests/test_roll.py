import numpy as np
import torch
import roll_me
import utils

def test_roll():
    # col-major
    src = np.asfortranarray(np.random.randint(0, 10, size=(2, 4)).astype(np.float32))
    print(src)
    dst = np.zeros((2, 4), dtype=np.float32, order="F")
    src_ref = torch.tensor(src)

    # dims 和 shifts 按 col-major 顺序传入：
    # dims = [ne0_0, ne0_1, ne0_2, ne0_3]，shift 对应每一维的 roll
    roll_me.roll(src, dst, [2, 4, 1, 1], [-1, -2, 0, 0])
    dst_ref = torch.roll(src_ref, shifts=(-1, -2), dims=(0, 1))

    print("output: \n", dst)
    print("target: \n", dst_ref)

    # 逐元素比较
    utils.compare_np_torch(dst, dst_ref)


if __name__ == "__main__":
    test_roll()
