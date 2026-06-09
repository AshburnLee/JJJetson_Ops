#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
import torch
from flash_attn.flash_attn_interface import flash_attn_func

# 与 JJJetson_Ops/tests/fa_test_common.py 一致
HEAD_DIM = 128
TOK_Q = 13
Q_HEADS = 16
TOK_KV = 256
KV_HEADS = 8

DST_Q = (HEAD_DIM, TOK_Q, Q_HEADS, 1)
DST_KV = (HEAD_DIM, TOK_KV, KV_HEADS, 1)


def random_inputs_fortran(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    q = np.asfortranarray(np.random.randn(*DST_Q).astype(np.float16))
    k = np.asfortranarray(np.random.randn(*DST_KV).astype(np.float16))
    v = np.asfortranarray(np.random.randn(*DST_KV).astype(np.float16))
    return q, k, v


def fortran_head_d_head_to_torch_bhsgd(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(HEAD_DIM, SEQ, HEADS[,1]) Fortran → (1, SEQ, HEADS, HEAD_DIM) contiguous CUDA."""
    if arr.ndim == 4:
        arr = np.asarray(arr[..., 0])
    x = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
    t = torch.from_numpy(x).to(device=device, dtype=torch.float16).contiguous()
    # flash_attn_func 要求 4 维：(batch, seqlen, n_heads, head_dim)
    return t.unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2477)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--repeats-after-warmup",
        type=int,
        default=1,
        help="warmup 之后执行 flash_attn_func 的次数；NCU 抓 kernel 时可设为 1",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA torch is not availbale")

    dev = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)

    qnp, knp, vnp = random_inputs_fortran(args.seed)
    q = fortran_head_d_head_to_torch_bhsgd(qnp, dev)
    k = fortran_head_d_head_to_torch_bhsgd(knp, dev)
    v = fortran_head_d_head_to_torch_bhsgd(vnp, dev)

    print(
        f"Case: d={HEAD_DIM} seqlen_q={TOK_Q} seqlen_k={TOK_KV} "
        f"nheads_q={Q_HEADS} nheads_kv={KV_HEADS} dtype=fp16 softmax_scale=1.0 causal=False "
        f"seed={args.seed}"
    )

    scale = 1.0

    def run_fa() -> None:
        flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
        )

    for _ in range(args.warmup):
        run_fa()
    torch.cuda.synchronize()

    for _ in range(args.repeats_after_warmup):
        run_fa()
    torch.cuda.synchronize()

    print(
        f"done: warmup={args.warmup} repeats_after_warmup={args.repeats_after_warmup} "
        f"(attach NCU around this Python process)"
    )


if __name__ == "__main__":
    main()
