import torch
import numpy as np
import cpy_continue_me

"""
逻辑：给定一个可能带有 padding（非连续 stride）的 src 张量，
通过传入"按字节计算的 stride"让 kernel 按有效布局读取，（即只读取有效值，不含padding）
然后把数据拷贝到一个连续的 dst buffer 中，同时完成 dtype 转换
"""

# 映射：PyTorch dtype -> NumPy dtype（用于构造 numpy buffer）
torch_2_numpy = {
    torch.float32:  np.float32,
    torch.float16:  np.float16,
    torch.bfloat16: np.float32,  # bfloat16 通过 float32 中转验证
    torch.int32:    np.int32,
}

# 映射：PyTorch dtype -> cpy_continue_me.data_type（传给 CUDA kernel）
torch_2_source = {
    torch.float32 : cpy_continue_me.data_type.f32,
    torch.float16 : cpy_continue_me.data_type.f16,
    torch.bfloat16 : cpy_continue_me.data_type.bf16,
    torch.int32 : cpy_continue_me.data_type.i32,
}


def test_continue():
    """
    测试 cpy_continue kernel 在"连续存储"情况下的正确性：
    - 使用 PyTorch 完成 dtype 转换作为 ground truth（dst_ref）
    - 使用 cpy_continue_me.cpy_con 完成从 src 到 dst_np 的拷贝 和 转换
    - 逐元素比较两者
    """
    shape = (2, 2, 2, 2)
    cases = [
        (torch.float32, torch.int32),
        (torch.int32, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        # (torch.bfloat16, torch.bfloat16), # TODO: 支持 bfloat16 直出
        # (torch.float32, torch.bfloat16),
        # (torch.bfloat16, torch.float32),
    ]

    for src_dtype, dst_dtype in cases:
        print(f"Testing {src_dtype} -> {dst_dtype}")

        # 1. 构造 PyTorch 源张量（仅用于生成数值和 ground truth）
        if src_dtype is torch.int32:
            src = torch.randint(-10, 10, shape, dtype=torch.int32)
        else:
            src = torch.randn(shape, dtype=src_dtype) * 10

        # 2. ground truth：用 PyTorch 做 dtype 转换
        if dst_dtype is torch.bfloat16:
            dst_ref = src.to(torch.bfloat16).to(torch.float32)
        else:
            dst_ref = src.to(dst_dtype)

        # 3. 用 numpy 构造 **列主序** 的 src_np，并从它得到 byte stride
        if src_dtype is torch.bfloat16:
            src_np = np.asfortranarray(src.to(torch.float32).numpy())
        else:
            src_np = np.asfortranarray(src.numpy())

        src_stride = list(src_np.strides)  # 已经是字节单位，col-major

        # 4. 连续的目标 dst_np 也用 Fortran order
        dst_np = np.zeros(shape, dtype=torch_2_numpy[dst_dtype], order="F")

        # 5. 调用 cpy_continue CUDA kernel：
        #    - src_np/dst_np: host buffer
        #    - src_dims/dst_dims: 逻辑 shape（与 torch 相同）
        #    - src_stride: 源张量按字节的 stride（支持带 padding 的情况）
        #    - dst_np.strides: 目标张量的 byte stride（这里是连续的）
        #    - src_dt/dst_dt: 源/目标数据类型（用于 kernel 内部处理）
        print("src_np.shape: ", src_np.shape)
        cpy_continue_me.cpy_con(
            src_np,
            dst_np,
            list(src_np.shape),   # [ne0_0, ne0_1, ne0_2, ne0_3]
            list(dst_np.shape),
            src_stride,           # [nb0_0, nb0_1, nb0_2, nb0_3] (bytes)
            list(dst_np.strides),
            torch_2_source[src_dtype],
            torch_2_source[dst_dtype],
        )

        # 6. 把结果转回 torch 比较
        res_me = torch.from_numpy(dst_np)

        torch.testing.assert_close(res_me, dst_ref, rtol=1e-3, atol=1e-3)
        print("OK")

    print("All tests passed!")


def test_w_padding():
    """
    使用 PyTorch 构造一个带 padding 的非连续张量（合法 stride），
    然后通过 numpy 的 strides 传给 cpy_continue，验证能正确拷贝到连续 dst。
    TODO: 没能构造出带 padding 的非连续张量
    """
    torch.manual_seed(0)
    shape = (2, 2, 3, 4)

    # 1. 连续的逻辑数据（ground truth）
    src_ref = torch.randn(shape, dtype=torch.float32)

    # 2. 在 PyTorch 中构造带 padding 的大张量：
    #    例如在 dim=2 上扩展为 5，再只用前 3，后 2 作为 padding
    big_shape = (shape[0], shape[1], shape[2] + 2, shape[3])  # (2,2,5,4)
    big = torch.zeros(big_shape, dtype=torch.float32)
    big[:, :, :shape[2], :] = src_ref

    # 非连续 view：逻辑 shape 仍为 (2,2,3,4)，但 stride 中隐含 padding
    src_view = big[:, :, :shape[2], :]  # view, 非 contiguous

    # 3. ground truth：去掉 padding 后的连续张量
    dst_ref = src_ref.clone()

    # 4. 转成 numpy，获取 dims 和 byte stride
    src_np = src_view.numpy()          # 共享底层内存 + stride（单位：字节）
    src_dims = list(src_np.shape)
    src_stride_bytes = list(src_np.strides)

    # 目标 dst 是连续存储
    dst_np = np.zeros_like(src_np)
    dst_dims = list(dst_np.shape)
    dst_stride_bytes = list(dst_np.strides)

    # 5. 调用 cpy_continue：理想情况下，dst_np 应该等于 src_ref（不含 padding）
    cpy_continue_me.cpy_con(
        src_np,
        dst_np,
        src_dims,
        dst_dims,
        src_stride_bytes,
        dst_stride_bytes,
        cpy_continue_me.data_type.f32,
        cpy_continue_me.data_type.f32,
    )

    res_me = torch.from_numpy(dst_np)

    print("src_view shape:", src_view.shape, " stride (elems):", src_view.stride())
    print("src_np   shape:", src_np.shape, " stride (bytes):", src_np.strides)
    print("dst_np   shape:", dst_np.shape, " stride (bytes):", dst_np.strides)

    print("==== src_view (torch, with padding) ====")
    print(src_view)
    print("==== src_ref (torch, ground truth) ====")
    print(src_ref)
    print("==== res_me (torch from dst_np, kernel output) ====")
    print(res_me)
    print("==== dst_ref (torch, expected) ====")
    print(dst_ref)

    torch.testing.assert_close(res_me, dst_ref, rtol=1e-5, atol=1e-5)
    print("test_w_padding OK (padding handled correctly)")


if __name__ == "__main__":
    test_continue()
    #test_w_padding()
