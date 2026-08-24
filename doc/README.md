# JJJetson_Ops 文档索引

文档按用途分三类；新增时请放入对应目录，并在 roadmap / conventions 中加链接。

## design/ — 设计 / 架构

Phase 蓝图、模块边界、生命周期、编码架构决策。

| 文件 | 内容 |
|------|------|
| [phase1_lifecycle.md](design/phase1_lifecycle.md) | Phase 1 TransformerRunner Session |
| [phase2_lifecycle.md](design/phase2_lifecycle.md) | Phase 2 Loader / Model / Engine / Sampler（§2.3 Model immutable 语义） |
| [header_cpp_layout.md](design/header_cpp_layout.md) | `.h` / `.cpp` 分工（opaque vs POD） |
| [pre_ln_vs_post_ln.md](design/pre_ln_vs_post_ln.md) | Pre-LN vs Post-LN 选型 |

## guide/ — 细节理解 / 数据流

API 契约、算子 device API、集成数据流、walkthrough 示例。

| 文件 | 内容 |
|------|------|
| [transformer_runner_device_api.md](guide/transformer_runner_device_api.md) | TransformerRunner C/Python API + KV layout |
| [attention_runner_integration.md](guide/attention_runner_integration.md) | Attention 段 layout / KV / FA 数据流 |
| [elementwise_device_api.md](guide/elementwise_device_api.md) | Elementwise 算子 |
| [rope_device_api.md](guide/rope_device_api.md) | RoPE 算子 |
| [fa_device_api.md](guide/fa_device_api.md) | Flash Attention 算子 |
| [rms_norm_device_api.md](guide/rms_norm_device_api.md) | RMSNorm / fused add |
| [qkv_pack_fp16_device_api.md](guide/qkv_pack_fp16_device_api.md) | Q pack fp16 过渡 |
| [moe_pipeline_example.md](guide/moe_pipeline_example.md) | MoE pipeline 小例 |
| [moe_dispatch_example.md](guide/moe_dispatch_example.md) | MoE dispatch 小例 |

| [inference_engine_device_api.md](guide/inference_engine_device_api.md) | InferenceEngine C/Python API + prefill/decode + token/hidden 路径 |
| [fixture_structure.md](guide/fixture_structure.md) | Weight fixture 目录：`config.txt` / `manifest.txt` / `.f32` 格式与 tensor 命名 |
| [hf_llama_weight_map.md](guide/hf_llama_weight_map.md) | HF Llama safetensors 名映射 + 2D 转置（步骤 3） |
| [generate_loop_device_api.md](guide/generate_loop_device_api.md) | GenerateLoop + Sampler（CUDA top-k/top-p/temperature） |
| [sampler-temperature.md](guide/sampler-temperature.md) | Sampler temperature：控制什么、为何需要、与 top_k 关系 |
| [sampler-top-p.md](guide/sampler-top-p.md) | Sampler top-p（nucleus）：在干什么、与 top_k/temperature 组合 |

## reference/ — 参考 / 性能

性能基线、优化点备忘，非 lifecycle 主图纸。

| 文件 | 内容 |
|------|------|
| [baseline-perf.md](reference/baseline-perf.md) | Orin 性能与优化技术点 |
