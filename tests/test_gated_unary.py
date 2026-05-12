import gated_unary_me
import numpy as np
import torch
import torch.nn.functional as F


def create_storage(physical_n0, n0, n1, n2, n3) -> torch.Tensor:
    # 构造底层物理存储：如 12×10×16×32 = 61440 个 float
    # physical_n0 = 12          # 每行物理宽度（8个真实 + 4个padding）
    storage = torch.zeros(n3 * n2 * n1 * physical_n0, dtype=torch.float32)

    # 填充真实数据：从 1 开始递增
    real_data = torch.arange(1, n0 * n1 * n2 * n3 + 1, dtype=torch.float32)  # 长度 40960

    # 计算真实数据应该写入的物理位置（跳过每行的4个padding）
    idx = 0
    for b in range(n3):
        for c in range(n2):
            for row in range(n1):
                start = b * (n2 * n1 * physical_n0) + c * (n1 * physical_n0) + row * physical_n0
                storage[start : start + n0] = real_data[idx : idx + n0]
                idx += n0
                # start+n0 到 start+(12-1) 保持为 0，稍后统一填 -999

    # 把所有 padding 位置设为 -999.0
    storage.view(n3, n2, n1, physical_n0)[:, :, :, n0:] = -999.0
    return storage


def test_data_generated():
    """
    测试生成目标数据的方法
    col_tensor 是构造出来的含有padding的数据
    === view with NO padding ===
    tensor w pad shape :   torch.Size([12, 10, 16, 32]) 12个真实值 -999是真实值
    tensor w pad sample:
    tensor([[   1.,    9.,   17.,   25.,   33.,   41.,   49.,   57.,   65.,   73.],
            [   2.,   10.,   18.,   26.,   34.,   42.,   50.,   58.,   66.,   74.],
            [   3.,   11.,   19.,   27.,   35.,   43.,   51.,   59.,   67.,   75.],
            [   4.,   12.,   20.,   28.,   36.,   44.,   52.,   60.,   68.,   76.],
            [   5.,   13.,   21.,   29.,   37.,   45.,   53.,   61.,   69.,   77.],
            [   6.,   14.,   22.,   30.,   38.,   46.,   54.,   62.,   70.,   78.],
            [   7.,   15.,   23.,   31.,   39.,   47.,   55.,   63.,   71.,   79.],
            [   8.,   16.,   24.,   32.,   40.,   48.,   56.,   64.,   72.,   80.],
            [-999., -999., -999., -999., -999., -999., -999., -999., -999., -999.],
            [-999., -999., -999., -999., -999., -999., -999., -999., -999., -999.],
            [-999., -999., -999., -999., -999., -999., -999., -999., -999., -999.],
            [-999., -999., -999., -999., -999., -999., -999., -999., -999., -999.]])
    tensor w pad stride:    (1, 12, 120, 1920)

    col_tensor_nopad 是从物理位置的一个 view, view 其不含padding
    tensor w pad shape:   torch.Size([8, 10, 16, 32])
    tensor w pad sample:
    tensor([[ 1.,  9., 17., 25., 33., 41., 49., 57., 65., 73.],
            [ 2., 10., 18., 26., 34., 42., 50., 58., 66., 74.],
            [ 3., 11., 19., 27., 35., 43., 51., 59., 67., 75.],
            [ 4., 12., 20., 28., 36., 44., 52., 60., 68., 76.],
            [ 5., 13., 21., 29., 37., 45., 53., 61., 69., 77.],
            [ 6., 14., 22., 30., 38., 46., 54., 62., 70., 78.],
            [ 7., 15., 23., 31., 39., 47., 55., 63., 71., 79.],
            [ 8., 16., 24., 32., 40., 48., 56., 64., 72., 80.]])
    tensor w pad stride:  (1, 12, 120, 1920)
    """
    n0, n1, n2, n3 = 8, 10, 16, 32
    n0_pad = 12  # 在第0维度元素后添加4个padding
    storage_1d = create_storage(n0_pad, n0, n1, n2, n3)
    # print("stoirage: ", storage_1d.shape)  #torch.Size([61440])
    # print("storage_1d: ", storage_1d.stride()) # 1

    print("=== view with NO padding ===")
    # 包含 padding 的 view
    col_tensor = torch.as_strided(
        storage_1d,
        # 我想 view 的shape
        size=(n0_pad, n1, n2, n3),
        # 元素 stride [1, 12, 120, 1920] 对应 ggml nb = [4, 48, 480, 7680]
        stride=(1, n0_pad, n0_pad * n1, n0_pad * n1 * n2),
    )

    print("tensor w pad shape :  ", col_tensor.shape)
    print("tensor w pad sample:  \n", col_tensor[:, :, 0, 0])
    print("tensor w pad stride:   ", col_tensor.stride())  # (1, 12, 120, 1920)

    print("=== view with padding ===")
    ## 包含 padding 的 view
    col_tensor_nopad = torch.as_strided(
        storage_1d,
        # 我想view的 shape 其包含了padding
        size=(n0, n1, n2, n3),
        # 元素 stride [1, 12, 120, 1920] 对应 ggml nb = [4, 48, 480, 7680]
        stride=(1, n0_pad, n0_pad * n1, n0_pad * n1 * n2),
    )

    print("tensor w pad shape:  ", col_tensor_nopad.shape)
    print("tensor w pad sample: \n", col_tensor_nopad[:, :, 0, 0])
    print("tensor w pad stride: ", col_tensor_nopad.stride())


def test_gated_unary():
    """
    src0, src1, dst, dst_dim, src0_nb, src1_nb
    """
    # 使用与 test_data_generated 相同的构造方式，得到一个不含 padding 的 col-major 视图
    n0, n1, n2, n3 = 8, 10, 16, 32
    n0_pad = 12  # 第 0 维物理宽度（含 padding）

    # 底层 1D 物理存储，包含真实数据 + padding（填 -999）
    storage_1d = create_storage(n0_pad, n0, n1, n2, n3)

    # 从物理存储构造一个不含 padding 的 4D 视图：
    # shape = (n0, n1, n2, n3)，列主序，stride 对应 ggml nb = [4, 48, 480, 7680]
    src0 = torch.as_strided(
        storage_1d,
        size=(n0, n1, n2, n3),
        stride=(1, n0_pad, n0_pad * n1, n0_pad * n1 * n2),
    )

    # src1 与 src0 相同，测试 relu_gated(x, x)
    src1 = src0

    # 计算 nb（byte stride），与 CUDA kernel 中的 nb 吻合（float32 => 4 bytes）
    src0_nb = [s * 4 for s in src0.stride()]
    src1_nb = src0_nb

    # 目标输出 dst：与 kernel 的 g_id（维 0 最快扁化）一致，须列主序连续
    dst_dims = (n0, n1, n2, n3)
    dst_np = np.zeros(dst_dims, dtype=np.float32, order="F")

    # 将非连续视图转为 numpy，保留底层 buffer 与 strides
    src0_np = src0.numpy()
    src1_np = src1.numpy()

    # 调用 CUDA 实现
    print("src0_np: ", src0_np.shape)
    print("src1_np: ", src1_np.shape)
    print("src0_np stride: ", [int(i / 4) for i in src0_np.strides])
    print("src1_np stride: ", [int(i / 4) for i in src1_np.strides])
    print("dst_dims: ", dst_dims)
    print("src0_nb: ", src0_nb, [int(i / 4) for i in src0_nb])
    print("src1_nb: ", src1_nb, [int(i / 4) for i in src1_nb])

    gated_unary_me.relu_gated(src0_np, src1_np, dst_np, dst_dims, src0_nb, src1_nb)

    # GT：直接在不含 padding 的 torch view上做 gated relu
    src_ref = src0.clone()
    dst_ref = F.relu(src_ref) * src_ref

    # compare
    print("============= me ==============")
    print("me input src0: \n", src0_np[:, :, 0, 0])
    print("me output:     \n", dst_np[:, :, 0, 0])
    print("============= ref ==============")
    print("torch input:   \n", src_ref[:, :, 0, 0])
    print("torch input shape:   ", src_ref.shape)
    print("torch input stride:  ", src_ref.stride())
    print("torch output:  \n", dst_ref[:, :, 0, 0])
    print("torch output shape:  ", dst_ref.shape)
    print("torch output stride: ", dst_ref.stride())

    dst_me = torch.from_numpy(dst_np)
    torch.testing.assert_close(dst_me, dst_ref, rtol=1e-3, atol=1e-3)
    assert dst_me.shape == dst_ref.shape
    print("Passed")


def demo_relu_gated_simple():
    """
    连续内存、无 padding 的场景下调用 relu_gated，并打印输入输出。
    """
    N, M = 4, 5
    # 用简单递增整数，shape = (N, M, 1, 1)，显式使用列主序 (Fortran order)
    src_2d = np.arange(1, N * M + 1, dtype=np.float32).reshape(N, M, order="F")
    src = src_2d.reshape(N, M, 1, 1, order="F")
    dst = np.zeros_like(src, dtype=np.float32, order="F")

    dst_dims = (N, M, 1, 1)
    # 使用 numpy 的字节 strides 作为 nb，列主序
    src_nb = list(src.strides)

    print("src shape:", src.shape)
    print("dst_dims:", dst_dims)
    print("src_nb (bytes):", src_nb)

    gated_unary_me.relu_gated(src, src, dst, dst_dims, src_nb, src_nb)

    print("src (2D view):\n", src[:, :, 0, 0])
    print("dst (relu_gated(src, src), 2D view):\n", dst[:, :, 0, 0])


def demo_relu_gated_padded_colmajor():
    """
    构造一个 **列主序 (col-major)** 的 2D 张量，shape 逻辑上为 (4, 5)，
    元素为 1..20，第一维（最内层、变化最快的维度）物理上有 7 个元素，
    后 3 个是 padding（值为 999）。然后调用 relu_gated 并打印输入输出。
    """
    ne0, ne1 = 4, 5  # 逻辑形状 (ne0, ne1)
    ne0_pad = ne0 + 3  # 物理第一维含 3 个 padding

    # 1. 在列主序视角下构造连续的 4x5 真值矩阵（不含 padding）
    true_2d = np.arange(1, ne0 * ne1 + 1, dtype=np.float32).reshape(ne0, ne1, order="F")

    # 2. 在一维 buffer 上，按 col-major + padding 构造物理存储
    # 物理 2D 布局：shape_phys = (ne0_pad, ne1)，列主序
    shape_phys = (ne0_pad, ne1)
    buf = np.full(shape_phys[0] * shape_phys[1], 999.0, dtype=np.float32)

    # 物理列主序元素级 stride
    elem_stride0 = 1
    elem_stride1 = ne0_pad

    # 将 true_2d 写入物理 buffer（只写前 ne0 行，后 3 行保持 999）
    for j in range(ne1):
        for i in range(ne0):
            phys_offset = i * elem_stride0 + j * elem_stride1
            buf[phys_offset] = true_2d[i, j]
    print(buf)

    # 3. 基于物理 buffer 构造一个带 padding 的 4D col-major 视图，逻辑 shape=(4,5,1,1)
    # 元素级 stride: [1, ne0_pad, ne0_pad*ne1, ne0_pad*ne1]（高维只是一维 batch，占位）
    elem_strides_4d = (1, ne0_pad, ne0_pad * ne1, ne0_pad * ne1)
    byte_strides_4d = tuple(s * buf.itemsize for s in elem_strides_4d)

    src = np.lib.stride_tricks.as_strided(
        buf,
        shape=(ne0_pad, ne1, 1, 1),
        strides=byte_strides_4d,
    )

    # 4. 目标 dst：连续、列主序
    # dst = np.zeros_like(src, dtype=np.float32, order="F")
    dst = np.zeros((ne0, ne1, 1, 1), dtype=np.float32, order="F")
    dst_dims = (ne0, ne1, 1, 1)
    src_nb = list(src.strides)  # byte stride，含 padding

    print("=== demo_relu_gated_padded_colmajor ===")
    print("logical shape (ne0, ne1):", (ne0, ne1))
    print("physical ne0_pad:", ne0_pad)
    print("src_nb (bytes):", src_nb)
    print("src (2D logical view):\n", src[:, :, 0, 0])

    gated_unary_me.relu_gated(src, src, dst, dst_dims, src_nb, src_nb)

    print("dst (relu_gated(src, src), 2D view):\n", dst[:, :, 0, 0])


if __name__ == "__main__":
    # 连续场景
    demo_relu_gated_simple()
    # 带 padding 的列主序 2D 场景
    demo_relu_gated_padded_colmajor()
    # test_data_generated()
    # test_gated_unary()

"""
[  1.   2.   3.   4. 999. 999. 999.   5.   6.   7.   8. 999. 999. 999.
   9.  10.  11.  12. 999. 999. 999.  13.  14.  15.  16. 999. 999. 999.
  17.  18.  19.  20. 999. 999. 999.]

=== demo_relu_gated_padded_colmajor ===
logical shape (ne0, ne1): (4, 5)
physical ne0_pad: 7
src_nb (bytes): [4, 28, 140, 140]
src (2D logical view):
 [[  1.   5.   9.  13.  17.]
 [  2.   6.  10.  14.  18.]
 [  3.   7.  11.  15.  19.]
 [  4.   8.  12.  16.  20.]
 [999. 999. 999. 999. 999.]
 [999. 999. 999. 999. 999.]
 [999. 999. 999. 999. 999.]]

dst (relu_gated(src, src), 2D view):
 [[  1.  25.  81. 169. 289.]
 [  4.  36. 100. 196. 324.]
 [  9.  49. 121. 225. 361.]
 [ 16.  64. 144. 256. 400.]]

符合预期
"""
