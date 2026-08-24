import numpy as np
import qkv_pack_fp16_me
import torch

import utils

HEAD_DIM = 32
NUM_TOKENS = 13
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
SEED = 24


def pack_ref(src: np.ndarray, head_dim: int, num_tokens: int, num_heads: int) -> np.ndarray:
    out = np.zeros((head_dim, num_tokens, num_heads, 1), dtype=np.float16, order="F")
    for h in range(num_heads):
        for t in range(num_tokens):
            for d in range(head_dim):
                src_feat = h * head_dim + d
                out[d, t, h, 0] = np.float16(src[src_feat, t, 0, 0])
    return out


def _test_pack(num_heads: int, test_name: str) -> None:
    feat_dim = HEAD_DIM * num_heads
    np.random.seed(SEED + num_heads)
    src = np.asfortranarray(np.random.randn(feat_dim, NUM_TOKENS, 1, 1).astype(np.float32))

    dst = np.zeros((HEAD_DIM, NUM_TOKENS, num_heads, 1), dtype=np.uint16, order="F")
    qkv_pack_fp16_me.forward_host(src, dst, HEAD_DIM, NUM_TOKENS, num_heads)

    ref = pack_ref(src, HEAD_DIM, NUM_TOKENS, num_heads)
    dst_f16 = dst.view(np.float16).reshape(ref.shape, order="F")
    ok = utils.compare_np_torch(dst_f16, torch.from_numpy(ref.astype(np.float32)), atol=0, rtol=0)
    assert ok, f"{test_name} mismatch"
    print(f"Passed {test_name}")


def test_qkv_pack_fp16_q_heads() -> None:
    _test_pack(NUM_Q_HEADS, "test_qkv_pack_fp16_q_heads")


def test_qkv_pack_fp16_kv_heads() -> None:
    _test_pack(NUM_KV_HEADS, "test_qkv_pack_fp16_kv_heads")


if __name__ == "__main__":
    test_qkv_pack_fp16_q_heads()
    test_qkv_pack_fp16_kv_heads()
