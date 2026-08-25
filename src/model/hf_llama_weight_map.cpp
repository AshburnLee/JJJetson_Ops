#include "hf_llama_weight_map.h"

#include "model_config.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static char *duplicate_c_string(const char *text) {
    if (text == nullptr) {
        return nullptr;
    }
    const size_t len = std::strlen(text);
    char *copy = static_cast<char *>(std::malloc(len + 1));
    if (copy == nullptr) {
        return nullptr;
    }
    std::memcpy(copy, text, len + 1);
    return copy;
}

static void host_tensor_free_fields(HostTensor *tensor) {
    if (tensor == nullptr) {
        return;
    }
    free(tensor->name);
    free(tensor->data);
    free(tensor->dims);
    tensor->name = nullptr;
    tensor->data = nullptr;
    tensor->dims = nullptr;
    tensor->ndim = 0;
}

static int parse_layer_index(const char *hf_name, int *out_layer) {
    static const char kPrefix[] = "model.layers.";
    if (std::strncmp(hf_name, kPrefix, sizeof(kPrefix) - 1) != 0) {
        return -1;
    }
    const char *p = hf_name + (sizeof(kPrefix) - 1);
    if (!std::isdigit(static_cast<unsigned char>(*p))) {
        return -1;
    }
    char *end = nullptr;
    const long layer = std::strtol(p, &end, 10);
    if (end == p || *end != '.') {
        return -1;
    }
    if (layer < 0 || layer > 1024) {
        return -1;
    }
    *out_layer = static_cast<int>(layer);
    return 0;
}

static int map_layer_suffix(const char *suffix, char *out_internal_suffix, size_t out_cap,
                            bool *need_transpose) {
    struct Rule {
        const char *hf_suffix;
        const char *internal_suffix;
        bool transpose;
    };
    static const Rule kRules[] = {
        {"self_attn.q_proj.weight", "w_q", true},
        {"self_attn.k_proj.weight", "w_k", true},
        {"self_attn.v_proj.weight", "w_v", true},
        {"self_attn.o_proj.weight", "w_o", true},
        {"mlp.gate_proj.weight", "w_gate", true},
        {"mlp.up_proj.weight", "w_up", true},
        {"mlp.down_proj.weight", "w_down", true},
        {"input_layernorm.weight", "w_input_layernorm", false},
        {"post_attention_layernorm.weight", "w_post_attention_layernorm", false},
    };

    for (const Rule &rule : kRules) {
        if (std::strcmp(suffix, rule.hf_suffix) == 0) {
            if (std::snprintf(out_internal_suffix, out_cap, "%s", rule.internal_suffix) >=
                static_cast<int>(out_cap)) {
                return -1;
            }
            *need_transpose = rule.transpose;
            return 0;
        }
    }
    return -1;
}

static int map_hf_llama_key(const char *hf_name, char *out_internal, size_t out_cap,
                            bool *need_transpose) {
    if (hf_name == nullptr || out_internal == nullptr || need_transpose == nullptr ||
        out_cap == 0) {
        return -1;
    }

    if (std::strcmp(hf_name, "model.embed_tokens.weight") == 0) {
        if (std::snprintf(out_internal, out_cap, "embed") >= static_cast<int>(out_cap)) {
            return -1;
        }
        *need_transpose = false;
        return 0;
    }
    if (std::strcmp(hf_name, "model.norm.weight") == 0) {
        if (std::snprintf(out_internal, out_cap, "final_norm") >= static_cast<int>(out_cap)) {
            return -1;
        }
        *need_transpose = false;
        return 0;
    }
    if (std::strcmp(hf_name, "lm_head.weight") == 0) {
        if (std::snprintf(out_internal, out_cap, "lm_head") >= static_cast<int>(out_cap)) {
            return -1;
        }
        *need_transpose = true;
        return 0;
    }

    int layer_idx = 0;
    if (parse_layer_index(hf_name, &layer_idx) != 0) {
        return -1;
    }

    static const char kPrefix[] = "model.layers.";
    const char *p = hf_name + std::strlen(kPrefix);
    while (std::isdigit(static_cast<unsigned char>(*p)) != 0) {
        ++p;
    }
    if (*p != '.') {
        return -1;
    }
    ++p;

    char internal_suffix[64];
    if (map_layer_suffix(p, internal_suffix, sizeof(internal_suffix), need_transpose) != 0) {
        return -1;
    }
    if (std::snprintf(out_internal, out_cap, "layer%d.%s", layer_idx, internal_suffix) >=
        static_cast<int>(out_cap)) {
        return -1;
    }
    return 0;
}

static int host_tensor_clone_1d(const HostTensor *src, HostTensor *dst, const char *new_name) {
    if (src->ndim != 1) {
        return -1;
    }
    const size_t byte_len = static_cast<size_t>(src->dims[0]) * sizeof(float);
    std::memset(dst, 0, sizeof(*dst));
    dst->name = duplicate_c_string(new_name);
    dst->ndim = 1;
    dst->dims = static_cast<int64_t *>(std::malloc(sizeof(int64_t)));
    if (dst->name == nullptr || dst->dims == nullptr) {
        host_tensor_free_fields(dst);
        return -1;
    }
    dst->dims[0] = src->dims[0];
    dst->data = static_cast<float *>(std::malloc(byte_len));
    if (dst->data == nullptr) {
        host_tensor_free_fields(dst);
        return -1;
    }
    std::memcpy(dst->data, src->data, byte_len);
    dst->dtype = src->dtype;
    return 0;
}

// HF [out, in] row-major -> internal [in, out] row-major
static int host_tensor_transpose_2d(const HostTensor *src, HostTensor *dst, const char *new_name) {
    if (src->ndim != 2) {
        return -1;
    }
    const int64_t out_rows = src->dims[0];
    const int64_t in_cols = src->dims[1];
    const int64_t in_rows = in_cols;
    const int64_t out_cols = out_rows;
    const size_t byte_len = static_cast<size_t>(in_rows * out_cols) * sizeof(float);

    std::memset(dst, 0, sizeof(*dst));
    dst->name = duplicate_c_string(new_name);
    dst->ndim = 2;
    dst->dims = static_cast<int64_t *>(std::malloc(2 * sizeof(int64_t)));
    if (dst->name == nullptr || dst->dims == nullptr) {
        host_tensor_free_fields(dst);
        return -1;
    }
    dst->dims[0] = in_rows;
    dst->dims[1] = out_cols;
    dst->data = static_cast<float *>(std::malloc(byte_len));
    if (dst->data == nullptr) {
        host_tensor_free_fields(dst);
        return -1;
    }

    for (int64_t r = 0; r < out_rows; ++r) {
        for (int64_t c = 0; c < in_cols; ++c) {
            dst->data[c * out_cols + r] = src->data[r * in_cols + c];
        }
    }
    dst->dtype = src->dtype;
    return 0;
}

static int host_tensor_copy_as(const HostTensor *src, HostTensor *dst, const char *new_name) {
    if (src->ndim == 1) {
        return host_tensor_clone_1d(src, dst, new_name);
    }
    if (src->ndim == 2) {
        std::memset(dst, 0, sizeof(*dst));
        dst->name = duplicate_c_string(new_name);
        dst->ndim = 2;
        dst->dims = static_cast<int64_t *>(std::malloc(2 * sizeof(int64_t)));
        if (dst->name == nullptr || dst->dims == nullptr) {
            host_tensor_free_fields(dst);
            return -1;
        }
        dst->dims[0] = src->dims[0];
        dst->dims[1] = src->dims[1];
        const size_t byte_len = static_cast<size_t>(src->dims[0] * src->dims[1]) * sizeof(float);
        dst->data = static_cast<float *>(std::malloc(byte_len));
        if (dst->data == nullptr) {
            host_tensor_free_fields(dst);
            return -1;
        }
        std::memcpy(dst->data, src->data, byte_len);
        dst->dtype = src->dtype;
        return 0;
    }
    return -1;
}

static bool internal_name_seen(const std::vector<std::string> &names, const char *name) {
    for (const std::string &existing : names) {
        if (existing == name) {
            return true;
        }
    }
    return false;
}

static void skip_json_ws(const std::string &text, size_t *pos) {
    while (*pos < text.size() && std::isspace(static_cast<unsigned char>(text[*pos])) != 0) {
        ++(*pos);
    }
}

// 在 JSON 文本里找 "key": ，返回冒号后面第一个非空白字符的下标；找不到返回 npos。
static size_t find_json_colon_value(const std::string &text, const char *key) {
    const std::string quoted = std::string("\"") + key + "\"";
    size_t search = 0;
    while (search < text.size()) {
        const size_t hit = text.find(quoted, search);
        if (hit == std::string::npos) {
            return std::string::npos;
        }
        size_t pos = hit + quoted.size();
        skip_json_ws(text, &pos);
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            skip_json_ws(text, &pos);
            return pos;
        }
        search = hit + 1;
    }
    return std::string::npos;
}

static int parse_json_int_at(const std::string &text, size_t pos, int *out) {
    if (pos >= text.size()) {
        return -1;
    }
    char *end = nullptr;
    const long value = std::strtol(text.c_str() + pos, &end, 10);
    if (end == text.c_str() + pos) {
        return -1;
    }
    *out = static_cast<int>(value);
    return 0;
}

static int parse_json_float_at(const std::string &text, size_t pos, float *out) {
    if (pos >= text.size()) {
        return -1;
    }
    char *end = nullptr;
    const float value = std::strtof(text.c_str() + pos, &end);
    if (end == text.c_str() + pos) {
        return -1;
    }
    *out = value;
    return 0;
}

static int parse_json_bool_at(const std::string &text, size_t pos, int *out) {
    if (pos >= text.size()) {
        return -1;
    }
    if (text.compare(pos, 4, "true") == 0) {
        *out = 1;
        return 0;
    }
    if (text.compare(pos, 5, "false") == 0) {
        *out = 0;
        return 0;
    }
    return parse_json_int_at(text, pos, out);
}

static int require_json_int(const std::string &text, const char *key, int *out) {
    const size_t pos = find_json_colon_value(text, key);
    if (pos == std::string::npos) {
        return -1;
    }
    return parse_json_int_at(text, pos, out);
}

} // namespace

extern "C" int hf_llama_map_weight_load_result(WeightLoadResult *result) {
    if (result == nullptr || result->tensors == nullptr || result->num_tensors <= 0) {
        return -1;
    }

    std::vector<HostTensor> mapped;
    mapped.reserve(static_cast<size_t>(result->num_tensors));
    std::vector<std::string> mapped_names;

    for (int i = 0; i < result->num_tensors; ++i) {
        const HostTensor *src = &result->tensors[i];
        if (src->name == nullptr) {
            return -1;
        }

        char internal_name[128];
        bool need_transpose = false;
        if (map_hf_llama_key(src->name, internal_name, sizeof(internal_name), &need_transpose) !=
            0) {
            std::fprintf(stderr, "hf_llama_map: skip unmapped tensor %s\n", src->name);
            continue;
        }
        if (internal_name_seen(mapped_names, internal_name)) {
            std::fprintf(stderr, "hf_llama_map: duplicate internal name %s\n", internal_name);
            return -1;
        }

        HostTensor dst{};
        int rc = -1;
        if (need_transpose) {
            rc = host_tensor_transpose_2d(src, &dst, internal_name);
        } else {
            rc = host_tensor_copy_as(src, &dst, internal_name);
        }
        if (rc != 0) {
            for (HostTensor &t : mapped) {
                host_tensor_free_fields(&t);
            }
            return -1;
        }
        mapped.push_back(dst);
        mapped_names.emplace_back(internal_name);
    }

    if (mapped.empty()) {
        return -1;
    }

    for (int i = 0; i < result->num_tensors; ++i) {
        host_tensor_free_fields(&result->tensors[i]);
    }
    free(result->tensors);

    HostTensor *out_array =
        static_cast<HostTensor *>(std::malloc(mapped.size() * sizeof(HostTensor)));
    if (out_array == nullptr) {
        for (HostTensor &t : mapped) {
            host_tensor_free_fields(&t);
        }
        result->tensors = nullptr;
        result->num_tensors = 0;
        return -1;
    }
    for (size_t i = 0; i < mapped.size(); ++i) {
        out_array[i] = mapped[i];
    }
    result->tensors = out_array;
    result->num_tensors = static_cast<int>(mapped.size());
    return 0;
}

// 读 HF Llama config.json -> ModelConfig。
//
// Big picture 里它在哪？
//   weight_loader_load_safetensors 同目录没有 config.txt 时调用。字段名是 HF 的
//   num_hidden_layers / num_attention_heads，不是内部 hidden_size 那套 11 项 key。
//   图纸：doc/guide/hf_llama_weight_map.md 步骤 4。
//
// 函数内部顺序（逐步）：
//   例：{"hidden_size":2048,"num_attention_heads":32,"num_hidden_layers":1,
//        "intermediate_size":5632,"vocab_size":32000,"max_position_embeddings":2048,
//        "rms_norm_eps":1e-5,"num_key_value_heads":4,"tie_word_embeddings":true}
//   step 1. 整文件读成字符串
//   step 2. 必填：hidden_size、intermediate_size、num_hidden_layers、num_attention_heads、
//           vocab_size、max_position_embeddings、rms_norm_eps
//   step 3. 选填：num_key_value_heads（缺则 = num_q_heads）、head_dim（缺则 2048/32=64）、
//           rope_theta（缺则 10000）、tie_word_embeddings（缺则 0）
//   step 4. model_config_validate
//
// 调用契约：path 须是普通文件；切片后 num_hidden_layers 必须等于 safetensors 里实际层数
extern "C" int hf_llama_parse_config_json(const char *path, ModelConfig *cfg) {
    if (path == nullptr || cfg == nullptr) {
        return -1;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        return -1;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string text = oss.str();
    if (text.empty()) {
        return -1;
    }

    ModelConfig parsed{};
    if (require_json_int(text, "hidden_size", &parsed.hidden_size) != 0) {
        return -1;
    }
    if (require_json_int(text, "intermediate_size", &parsed.intermediate_size) != 0) {
        return -1;
    }
    if (require_json_int(text, "num_hidden_layers", &parsed.num_layers) != 0) {
        return -1;
    }
    if (require_json_int(text, "num_attention_heads", &parsed.num_q_heads) != 0) {
        return -1;
    }
    if (require_json_int(text, "vocab_size", &parsed.vocab_size) != 0) {
        return -1;
    }
    if (require_json_int(text, "max_position_embeddings", &parsed.max_seq_len) != 0) {
        return -1;
    }

    const size_t eps_pos = find_json_colon_value(text, "rms_norm_eps");
    if (eps_pos == std::string::npos ||
        parse_json_float_at(text, eps_pos, &parsed.rms_norm_epsilon) != 0) {
        return -1;
    }

    const size_t kv_pos = find_json_colon_value(text, "num_key_value_heads");
    if (kv_pos == std::string::npos) {
        parsed.num_kv_heads = parsed.num_q_heads;
    } else if (parse_json_int_at(text, kv_pos, &parsed.num_kv_heads) != 0) {
        return -1;
    }

    const size_t head_pos = find_json_colon_value(text, "head_dim");
    if (head_pos == std::string::npos) {
        if (parsed.num_q_heads <= 0 || parsed.hidden_size % parsed.num_q_heads != 0) {
            return -1;
        }
        parsed.head_dim = parsed.hidden_size / parsed.num_q_heads;
    } else if (parse_json_int_at(text, head_pos, &parsed.head_dim) != 0) {
        return -1;
    }

    const size_t rope_pos = find_json_colon_value(text, "rope_theta");
    if (rope_pos == std::string::npos) {
        parsed.freq_base = 10000.f;
    } else if (parse_json_float_at(text, rope_pos, &parsed.freq_base) != 0) {
        return -1;
    }

    const size_t tie_pos = find_json_colon_value(text, "tie_word_embeddings");
    if (tie_pos == std::string::npos) {
        parsed.tie_word_embeddings = 0;
    } else if (parse_json_bool_at(text, tie_pos, &parsed.tie_word_embeddings) != 0) {
        return -1;
    }

    if (model_config_validate(&parsed) != 0) {
        return -1;
    }
    *cfg = parsed;
    return 0;
}
