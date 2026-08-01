#include "weight_loader.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTensorNdim = 8;
constexpr int kMaxManifestTensors = 512;

static std::string trim(const std::string &s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
}

static bool parse_int(const std::string &text, int *out) {
    if (text.empty()) {
        return false;
    }
    char *end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

static bool parse_float(const std::string &text, float *out) {
    if (text.empty()) {
        return false;
    }
    char *end = nullptr;
    float value = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

static int parse_config_file(const fs::path &config_path, ModelConfig *cfg) {
    std::ifstream in(config_path);
    if (!in.is_open()) {
        return -1;
    }

    bool seen_hidden_size = false;
    bool seen_intermediate_size = false;
    bool seen_num_layers = false;
    bool seen_num_q_heads = false;
    bool seen_num_kv_heads = false;
    bool seen_head_dim = false;
    bool seen_vocab_size = false;
    bool seen_max_seq_len = false;
    bool seen_freq_base = false;
    bool seen_rms_norm_epsilon = false;
    bool seen_tie_word_embeddings = false;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            return -1;
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));

        if (key == "hidden_size") {
            if (!parse_int(value, &cfg->hidden_size))
                return -1;
            seen_hidden_size = true;
        } else if (key == "intermediate_size") {
            if (!parse_int(value, &cfg->intermediate_size))
                return -1;
            seen_intermediate_size = true;
        } else if (key == "num_layers") {
            if (!parse_int(value, &cfg->num_layers))
                return -1;
            seen_num_layers = true;
        } else if (key == "num_q_heads") {
            if (!parse_int(value, &cfg->num_q_heads))
                return -1;
            seen_num_q_heads = true;
        } else if (key == "num_kv_heads") {
            if (!parse_int(value, &cfg->num_kv_heads))
                return -1;
            seen_num_kv_heads = true;
        } else if (key == "head_dim") {
            if (!parse_int(value, &cfg->head_dim))
                return -1;
            seen_head_dim = true;
        } else if (key == "vocab_size") {
            if (!parse_int(value, &cfg->vocab_size))
                return -1;
            seen_vocab_size = true;
        } else if (key == "max_seq_len") {
            if (!parse_int(value, &cfg->max_seq_len))
                return -1;
            seen_max_seq_len = true;
        } else if (key == "freq_base") {
            if (!parse_float(value, &cfg->freq_base))
                return -1;
            seen_freq_base = true;
        } else if (key == "rms_norm_epsilon") {
            if (!parse_float(value, &cfg->rms_norm_epsilon))
                return -1;
            seen_rms_norm_epsilon = true;
        } else if (key == "tie_word_embeddings") {
            if (!parse_int(value, &cfg->tie_word_embeddings))
                return -1;
            seen_tie_word_embeddings = true;
        } else {
            return -1;
        }
    }

    if (!(seen_hidden_size && seen_intermediate_size && seen_num_layers && seen_num_q_heads &&
          seen_num_kv_heads && seen_head_dim && seen_vocab_size && seen_max_seq_len &&
          seen_freq_base && seen_rms_norm_epsilon && seen_tie_word_embeddings)) {
        return -1;
    }
    return model_config_validate(cfg);
}

static int64_t tensor_numel(const HostTensor *tensor) {
    int64_t numel = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        numel *= tensor->dims[i];
    }
    return numel;
}

static int load_f32_tensor_file(const fs::path &file_path, HostTensor *tensor) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        return -1;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_bytes = in.tellg();
    if (file_bytes < 0) {
        return -1;
    }
    in.seekg(0, std::ios::beg);

    const int64_t expected_numel = tensor_numel(tensor);
    const int64_t expected_bytes = expected_numel * static_cast<int64_t>(sizeof(float));
    if (file_bytes != expected_bytes) {
        return -1;
    }

    tensor->data = static_cast<float *>(std::malloc(static_cast<size_t>(expected_bytes)));
    if (tensor->data == nullptr) {
        return -1;
    }
    if (!in.read(reinterpret_cast<char *>(tensor->data), expected_bytes)) {
        return -1;
    }
    tensor->dtype = WEIGHT_DTYPE_F32;
    return 0;
}

static char *duplicate_c_string(const std::string &text) {
    char *copy = static_cast<char *>(std::malloc(text.size() + 1));
    if (copy == nullptr) {
        return nullptr;
    }
    std::memcpy(copy, text.c_str(), text.size() + 1);
    return copy;
}

static int parse_manifest_and_load_tensors(const fs::path &fixture_dir, WeightLoadResult *out) {
    const fs::path manifest_path = fixture_dir / "manifest.txt";
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        return -1;
    }

    std::vector<HostTensor> parsed;
    parsed.reserve(16);

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (static_cast<int>(parsed.size()) >= kMaxManifestTensors) {
            return -1;
        }

        std::istringstream iss(line);
        std::string name;
        int ndim = 0;
        if (!(iss >> name >> ndim)) {
            return -1;
        }
        if (ndim <= 0 || ndim > kMaxTensorNdim) {
            return -1;
        }

        HostTensor tensor{};
        tensor.name = duplicate_c_string(name);
        if (tensor.name == nullptr) {
            return -1;
        }
        tensor.ndim = ndim;
        tensor.dims =
            static_cast<int64_t *>(std::malloc(static_cast<size_t>(ndim) * sizeof(int64_t)));
        if (tensor.dims == nullptr) {
            free(tensor.name);
            return -1;
        }

        for (int i = 0; i < ndim; ++i) {
            long dim = 0;
            if (!(iss >> dim) || dim <= 0) {
                free(tensor.name);
                free(tensor.dims);
                return -1;
            }
            tensor.dims[i] = static_cast<int64_t>(dim);
        }

        std::string rel_path;
        if (!(iss >> rel_path)) {
            free(tensor.name);
            free(tensor.dims);
            return -1;
        }

        const fs::path file_path = fixture_dir / rel_path;
        if (load_f32_tensor_file(file_path, &tensor) != 0) {
            free(tensor.name);
            free(tensor.dims);
            return -1;
        }
        parsed.push_back(tensor);
    }

    if (parsed.empty()) {
        return -1;
    }

    out->tensors = static_cast<HostTensor *>(std::malloc(parsed.size() * sizeof(HostTensor)));
    if (out->tensors == nullptr) {
        return -1;
    }
    for (size_t i = 0; i < parsed.size(); ++i) {
        out->tensors[i] = parsed[i];
    }
    out->num_tensors = static_cast<int>(parsed.size());
    return 0;
}

} // namespace

void weight_load_result_init(WeightLoadResult *out) {
    if (out == nullptr) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
}

static void host_tensor_free(HostTensor *tensor) {
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

void weight_load_result_destroy(WeightLoadResult *result) {
    if (result == nullptr) {
        return;
    }
    if (result->tensors != nullptr) {
        for (int i = 0; i < result->num_tensors; ++i) {
            host_tensor_free(&result->tensors[i]);
        }
        free(result->tensors);
    }
    weight_load_result_init(result);
}

const HostTensor *weight_load_result_find(const WeightLoadResult *result, const char *name) {
    if (result == nullptr || name == nullptr || result->tensors == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < result->num_tensors; ++i) {
        const HostTensor *tensor = &result->tensors[i];
        if (tensor->name != nullptr && std::strcmp(tensor->name, name) == 0) {
            return tensor;
        }
    }
    return nullptr;
}

int weight_loader_load_fixture(const char *path, WeightLoadResult *out) {
    if (path == nullptr || out == nullptr) {
        return -1;
    }

    weight_load_result_destroy(out);
    weight_load_result_init(out);

    const fs::path fixture_dir(path);
    if (!fs::is_directory(fixture_dir)) {
        return -1;
    }

    ModelConfig cfg{};
    if (parse_config_file(fixture_dir / "config.txt", &cfg) != 0) {
        return -1;
    }
    if (parse_manifest_and_load_tensors(fixture_dir, out) != 0) {
        weight_load_result_destroy(out);
        return -1;
    }

    out->config = cfg;
    return 0;
}

int weight_loader_load_safetensors(const char *path, WeightLoadResult *out) {
    if (path == nullptr || out == nullptr) {
        return -1;
    }
    (void)path;
    // safetensors 解析在后续细节阶段实现。
    return -1;
}
