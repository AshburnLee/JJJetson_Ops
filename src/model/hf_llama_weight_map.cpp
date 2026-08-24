#include "hf_llama_weight_map.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
