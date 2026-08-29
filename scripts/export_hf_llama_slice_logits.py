"""从本地 2 层 HF Llama 切片导出 prefill logits，给 Engine 当数值 ref（safetensors 步骤 4.1）。

本脚本只做 dump，不是单测，不进 run_tests.sh。
  1. --slice-dir 必须是已经切好的本地目录（config.json + model.safetensors）
  2. 只加载这一份切片，禁止去读 22 层全量源仓
  3. CPU + F32；不走 GPU，避免和 Engine 抢 Orin 统一内存
  4. 不联网；不走 tokenizer，token 手写

在 JJJetson_Ops 目录、cuda-ops 环境：

    python scripts/export_hf_llama_slice_logits.py \\
        --slice-dir models/tinyllama_2layer

结果是 HF的 transformers 用同一份2层f32 切片，输入TOKEN是 [1,2,3,4] 跑完了 embed、2层、lm_head
得到的 输出 logits。包括21个 tensors，shape 是（32000,4），即词表 32000, 4个位置。
最后一个位置是TOKEN 6415。

这会作为 ref 与我的engine 结果比较，两者使用相同的 21 个weight ：

~~~
同一份切片 model.safetensors（21 个 tensor，F32）
        |
        +-- HuggingFace 加载 -> 前向 [1,2,3,4] -> 存成 npy（ref）
        |
        +-- Engine Loader 加载 -> 前向 [1,2,3,4] -> 得到一张 logits
                                        |
                                        v
                                    只比这两张 logits 表
~~~

默认写出 models/tinyllama_2layer/ref_prefill_logits_t1234.npy
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# 测试输入，输入是logits 绕过了文字，因为那需要Tokenizer，当前engine 暂时不走Tokenizer
_DEFAULT_TOKENS = (1, 2, 3, 4)

# 切片 config.json 只有 ModelConfig 那几项。HF Llama 还要这些才能 from_pretrained。
# 值跟 TinyLlama 源仓一致，但不去读 22 层目录。
_LLAMA_CONFIG_DEFAULTS = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "use_cache": False,
}


def _require_slice_dir(path: str) -> str:
    # 例：path 是 models/tinyllama_2layer，里面已有 config.json 和 model.safetensors。
    if not os.path.isdir(path):
        raise SystemExit(f"--slice-dir must be an existing local directory, got {path!r}")
    cfg_path = os.path.join(path, "config.json")
    st_path = os.path.join(path, "model.safetensors")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"missing {cfg_path}")
    if not os.path.isfile(st_path):
        raise SystemExit(f"missing {st_path}")
    return os.path.abspath(path)


def _load_patched_config_dict(slice_dir: str) -> dict:
    with open(os.path.join(slice_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    n_layers = int(raw.get("num_hidden_layers", 0))
    # 防呆：22 层全量 F32 在 Orin 上会撑爆。切片必须是开发用的浅层。
    if n_layers < 1 or n_layers > 4:
        raise SystemExit(
            f"slice num_hidden_layers={n_layers} looks like a full checkpoint; "
            "use models/tinyllama_2layer (2 layers), not models/hf_src/"
        )
    for key, value in _LLAMA_CONFIG_DEFAULTS.items():
        raw.setdefault(key, value)
    # 切片权重已经是 F32；不要按源仓 BF16 去加载。
    raw["torch_dtype"] = "float32"
    return raw


def _hf_logits_to_engine_layout(hf_logits: np.ndarray) -> np.ndarray:
    """HF [T, vocab] -> Engine host [vocab, T] col-major。

    例：T=2, vocab=3，HF 行主序
        [[a, b, c],
         [d, e, f]]
    表示 token0 的 vocab logit 是 a,b,c，token1 是 d,e,f。
    Engine 要 logits[v, t]，同一份数排成
        [[a, d],
         [b, e],
         [c, f]]
    且 order="F"（先走 vocab，再走 token）。
    """
    if hf_logits.ndim != 2:
        raise SystemExit(f"expected HF logits [T, vocab], got shape {hf_logits.shape}")
    engine = np.asfortranarray(hf_logits.T.astype(np.float32, copy=False))
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump HF prefill logits from a local 2-layer Llama slice (CPU F32)."
    )
    parser.add_argument(
        "--slice-dir",
        default="models/tinyllama_2layer",
        help="local slice directory (config.json + F32 model.safetensors)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output .npy path; default <slice-dir>/ref_prefill_logits_t<ids>.npy",
    )
    parser.add_argument(
        "--tokens",
        default="1,2,3,4",
        help="comma-separated token ids, same as smoke (default 1,2,3,4)",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "need torch + transformers in cuda-ops: pip install transformers "
            f"(import failed: {exc})"
        ) from exc

    slice_dir = _require_slice_dir(args.slice_dir)
    token_ids = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    if not token_ids:
        raise SystemExit("--tokens must contain at least one id")
    cfg_dict = _load_patched_config_dict(slice_dir)
    vocab = int(cfg_dict["vocab_size"])
    for tid in token_ids:
        if tid < 0 or tid >= vocab:
            raise SystemExit(f"token id {tid} out of vocab_size={vocab}")

    # 例：tokens=[1] -> ref_prefill_logits_t1.npy；[1,2,3,4] -> ..._t1234.npy
    token_tag = "".join(str(t) for t in token_ids)
    out_path = args.out.strip() or os.path.join(slice_dir, f"ref_prefill_logits_t{token_tag}.npy")
    out_path = os.path.abspath(out_path)

    config = LlamaConfig.from_dict(cfg_dict)
    # CPU + F32。即使本机有 CUDA 也不上 GPU。
    # local_files_only：缺文件就失败，不要去 Hub 下东西。
    model = LlamaForCausalLM.from_pretrained(
        slice_dir,
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cpu")

    ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    # HF: [batch=1, T, vocab] -> [T, vocab]
    hf_btv = out.logits[0].detach().cpu().numpy()
    engine_logits = _hf_logits_to_engine_layout(hf_btv)
    t_len = len(token_ids)
    if engine_logits.shape != (vocab, t_len):
        raise SystemExit(
            f"layout mismatch: dumped {engine_logits.shape}, expected ({vocab}, {t_len})"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, engine_logits)
    last_argmax = int(np.argmax(engine_logits[:, -1]))
    print(
        f"wrote {out_path} shape={engine_logits.shape} order=F "
        f"tokens={token_ids} last_argmax={last_argmax}"
    )

    # embed 探针：HF [T, hidden] -> Engine [hidden, T] col-major，和 logits 同一套转置。
    with torch.no_grad():
        hf_emb = model.model.embed_tokens(ids)[0].detach().cpu().numpy()
    hidden_size = int(cfg_dict["hidden_size"])
    embed_engine = np.asfortranarray(hf_emb.T.astype(np.float32, copy=False))
    if embed_engine.shape != (hidden_size, t_len):
        raise SystemExit(
            f"embed layout mismatch: dumped {embed_engine.shape}, expected ({hidden_size}, {t_len})"
        )
    embed_path = os.path.join(os.path.dirname(out_path), f"ref_embed_t{token_tag}.npy")
    np.save(embed_path, embed_engine)
    print(f"wrote {embed_path} shape={embed_engine.shape} order=F")

    # hidden_states[0]=embed；其后每层一块；最后一块是 final RMSNorm 之后（Engine forward_hidden_host 出口）。
    hs = out.hidden_states
    print(f"  hf hidden_states count={len(hs)}")
    hs0 = np.asfortranarray(hs[0][0].detach().cpu().numpy().T.astype(np.float32, copy=False))
    print(f"  hs[0] vs embed max_abs={float(np.max(np.abs(hs0 - embed_engine))):.6g}")
    # Engine forward_hidden_host 出口是 final RMSNorm 之后。HF 的最后一块可能是 layer 输出，再过一次 norm。
    with torch.no_grad():
        post_norm = model.model.norm(hs[-1]).detach().cpu().numpy()
    post_norm_h = np.asfortranarray(post_norm[0].T.astype(np.float32, copy=False))
    last_h = np.asfortranarray(hs[-1][0].detach().cpu().numpy().T.astype(np.float32, copy=False))
    print(
        f"  hs[-1] vs model.norm(hs[-1]) max_abs={float(np.max(np.abs(last_h - post_norm_h))):.6g}"
    )
    final_h = post_norm_h
    if final_h.shape != (hidden_size, t_len):
        raise SystemExit(f"final hidden layout mismatch: {final_h.shape}")
    final_path = os.path.join(os.path.dirname(out_path), f"ref_hidden_final_t{token_tag}.npy")
    np.save(final_path, final_h)
    print(f"wrote {final_path} shape={final_h.shape} order=F")
    if len(hs) >= 2:
        layer0_h = np.asfortranarray(
            hs[1][0].detach().cpu().numpy().T.astype(np.float32, copy=False)
        )
        layer0_path = os.path.join(os.path.dirname(out_path), f"ref_hidden_layer0_t{token_tag}.npy")
        np.save(layer0_path, layer0_h)
        print(f"wrote {layer0_path} shape={layer0_h.shape} order=F")

    # layer0、RoPE 之后的 Q。HF 是 [B, H, T, D]，Engine 是 [H*D, T] 列主序。
    import inspect

    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    apply_fn = getattr(inspect.getmodule(type(attn)), "apply_rotary_pos_emb", None)
    if apply_fn is None:
        raise SystemExit("cannot find apply_rotary_pos_emb on the HF attention module")
    num_q_heads = int(cfg_dict["num_attention_heads"])
    num_kv_heads = int(cfg_dict.get("num_key_value_heads", num_q_heads))
    head_dim = int(cfg_dict.get("head_dim", hidden_size // num_q_heads))
    with torch.no_grad():
        h_norm = layer0.input_layernorm(model.model.embed_tokens(ids))
        bsz, seqlen, _ = h_norm.shape
        q = (
            attn.q_proj(h_norm)
            .view(bsz, seqlen, num_q_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            attn.k_proj(h_norm)
            .view(bsz, seqlen, num_kv_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            attn.v_proj(h_norm)
            .view(bsz, seqlen, num_kv_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        position_ids = torch.arange(seqlen, dtype=torch.long).unsqueeze(0)
        rotary = getattr(attn, "rotary_emb", None) or getattr(model.model, "rotary_emb", None)
        if rotary is None:
            raise SystemExit("cannot find rotary_emb on attn or model.model")
        try:
            cos_sin = rotary(v, position_ids=position_ids)
        except TypeError:
            try:
                cos_sin = rotary(v, position_ids)
            except TypeError:
                cos_sin = rotary(v, seqlen)
        if isinstance(cos_sin, tuple) and len(cos_sin) >= 2:
            cos, sin = cos_sin[0], cos_sin[1]
        else:
            raise SystemExit("rotary_emb did not return cos, sin")
        try:
            q_rot, k_rot = apply_fn(q, k, cos, sin)
        except TypeError:
            q_rot, k_rot = apply_fn(q, k, cos, sin, unsqueeze_dim=1)
    q_np = q_rot[0].detach().cpu().numpy()
    q_engine = np.zeros((num_q_heads * head_dim, t_len), dtype=np.float32, order="F")
    for ti in range(t_len):
        q_engine[:, ti] = q_np[:, ti, :].reshape(-1)
    q_path = os.path.join(os.path.dirname(out_path), f"ref_q_rope_t{token_tag}.npy")
    np.save(q_path, q_engine)
    print(f"wrote {q_path} shape={q_engine.shape} order=F")
    k_np = k_rot[0].detach().cpu().numpy()
    k_engine = np.zeros((num_kv_heads * head_dim, t_len), dtype=np.float32, order="F")
    for ti in range(t_len):
        k_engine[:, ti] = k_np[:, ti, :].reshape(-1)
    k_path = os.path.join(os.path.dirname(out_path), f"ref_k_rope_t{token_tag}.npy")
    np.save(k_path, k_engine)
    print(f"wrote {k_path} shape={k_engine.shape} order=F")
    v_np = v[0].detach().cpu().numpy()
    v_engine = np.zeros((num_kv_heads * head_dim, t_len), dtype=np.float32, order="F")
    for ti in range(t_len):
        v_engine[:, ti] = v_np[:, ti, :].reshape(-1)
    v_path = os.path.join(os.path.dirname(out_path), f"ref_v_t{token_tag}.npy")
    np.save(v_path, v_engine)
    print(f"wrote {v_path} shape={v_engine.shape} order=F")
    # HF 5.x 的 attn.forward 要 position_embeddings；这里用已经转好的 Q/K/V 自己做一遍 attention。
    with torch.no_grad():
        g = num_q_heads // num_kv_heads
        k_rep = k_rot.repeat_interleave(g, dim=1)
        v_rep = v.repeat_interleave(g, dim=1)
        scores = torch.matmul(q_rot, k_rep.transpose(-2, -1)) * (head_dim**-0.5)
        tq = q_rot.size(2)
        causal_mask = torch.triu(torch.ones(tq, tq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q_rot.dtype)
        ctx = torch.matmul(probs, v_rep)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, tq, num_q_heads * head_dim)
        attn_h = torch.nn.functional.linear(ctx, attn.o_proj.weight)[0].detach().cpu().numpy()
    attn_engine = np.asfortranarray(attn_h.T.astype(np.float32, copy=False))
    attn_path = os.path.join(os.path.dirname(out_path), f"ref_attn_out_t{token_tag}.npy")
    np.save(attn_path, attn_engine)
    print(f"wrote {attn_path} shape={attn_engine.shape} order=F")


if __name__ == "__main__":
    main()
