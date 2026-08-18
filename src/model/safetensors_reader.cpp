#include "safetensors_reader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxSafetensors = 512;

struct JsonCursor {
    const char *p;
    const char *end;
};

static void skip_ws(JsonCursor *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) {
        ++c->p;
    }
}

static int expect_char(JsonCursor *c, char ch) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != ch) {
        return -1;
    }
    ++c->p;
    return 0;
}

static int parse_string(JsonCursor *c, std::string *out) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != '"') {
        return -1;
    }
    ++c->p;
    out->clear();
    while (c->p < c->end) {
        const char ch = *c->p++;
        if (ch == '"') {
            return 0;
        }
        if (ch == '\\') {
            if (c->p >= c->end) {
                return -1;
            }
            out->push_back(*c->p++);
            continue;
        }
        out->push_back(ch);
    }
    return -1;
}

static int parse_int64(JsonCursor *c, int64_t *out) {
    skip_ws(c);
    if (c->p >= c->end || (*c->p != '-' && (*c->p < '0' || *c->p > '9'))) {
        return -1;
    }
    char *end_ptr = nullptr;
    const long long value = std::strtoll(c->p, &end_ptr, 10);
    if (end_ptr == c->p) {
        return -1;
    }
    c->p = end_ptr;
    *out = static_cast<int64_t>(value);
    return 0;
}

static int skip_json_value(JsonCursor *c);

static int skip_json_object(JsonCursor *c) {
    if (expect_char(c, '{') != 0) {
        return -1;
    }
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') {
        ++c->p;
        return 0;
    }
    while (c->p < c->end) {
        std::string key;
        if (parse_string(c, &key) != 0) {
            return -1;
        }
        if (expect_char(c, ':') != 0) {
            return -1;
        }
        if (skip_json_value(c) != 0) {
            return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) {
            return -1;
        }
        if (*c->p == '}') {
            ++c->p;
            return 0;
        }
        if (*c->p != ',') {
            return -1;
        }
        ++c->p;
    }
    return -1;
}

static int skip_json_array(JsonCursor *c) {
    if (expect_char(c, '[') != 0) {
        return -1;
    }
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        ++c->p;
        return 0;
    }
    while (c->p < c->end) {
        if (skip_json_value(c) != 0) {
            return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) {
            return -1;
        }
        if (*c->p == ']') {
            ++c->p;
            return 0;
        }
        if (*c->p != ',') {
            return -1;
        }
        ++c->p;
    }
    return -1;
}

static int skip_json_value(JsonCursor *c) {
    skip_ws(c);
    if (c->p >= c->end) {
        return -1;
    }
    const char ch = *c->p;
    if (ch == '{') {
        return skip_json_object(c);
    }
    if (ch == '[') {
        return skip_json_array(c);
    }
    if (ch == '"') {
        std::string dummy;
        return parse_string(c, &dummy);
    }
    if (ch == 't' || ch == 'f' || ch == 'n' || ch == '-' || (ch >= '0' && ch <= '9')) {
        while (c->p < c->end) {
            const char v = *c->p;
            if (v == ',' || v == '}' || v == ']' || v == ' ' || v == '\t' || v == '\n' ||
                v == '\r') {
                break;
            }
            ++c->p;
        }
        return 0;
    }
    return -1;
}

static int parse_int64_array(JsonCursor *c, std::vector<int64_t> *out) {
    out->clear();
    if (expect_char(c, '[') != 0) {
        return -1;
    }
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        ++c->p;
        return 0;
    }
    while (c->p < c->end) {
        int64_t value = 0;
        if (parse_int64(c, &value) != 0) {
            return -1;
        }
        out->push_back(value);
        skip_ws(c);
        if (c->p >= c->end) {
            return -1;
        }
        if (*c->p == ']') {
            ++c->p;
            return 0;
        }
        if (*c->p != ',') {
            return -1;
        }
        ++c->p;
    }
    return -1;
}

struct SafetensorsMeta {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

static int parse_tensor_entry(JsonCursor *c, SafetensorsMeta *meta) {
    if (expect_char(c, '{') != 0) {
        return -1;
    }
    meta->dtype.clear();
    meta->shape.clear();
    meta->data_begin = 0;
    meta->data_end = 0;

    skip_ws(c);
    if (c->p < c->end && *c->p == '}') {
        ++c->p;
        return -1;
    }

    while (c->p < c->end) {
        std::string key;
        if (parse_string(c, &key) != 0) {
            return -1;
        }
        if (expect_char(c, ':') != 0) {
            return -1;
        }
        if (key == "dtype") {
            if (parse_string(c, &meta->dtype) != 0) {
                return -1;
            }
        } else if (key == "shape") {
            if (parse_int64_array(c, &meta->shape) != 0) {
                return -1;
            }
        } else if (key == "data_offsets") {
            std::vector<int64_t> offsets;
            if (parse_int64_array(c, &offsets) != 0 || offsets.size() != 2) {
                return -1;
            }
            if (offsets[0] < 0 || offsets[1] < offsets[0]) {
                return -1;
            }
            meta->data_begin = static_cast<uint64_t>(offsets[0]);
            meta->data_end = static_cast<uint64_t>(offsets[1]);
        } else {
            if (skip_json_value(c) != 0) {
                return -1;
            }
        }
        skip_ws(c);
        if (c->p >= c->end) {
            return -1;
        }
        if (*c->p == '}') {
            ++c->p;
            break;
        }
        if (*c->p != ',') {
            return -1;
        }
        ++c->p;
    }

    if (meta->dtype.empty() || meta->shape.empty() || meta->data_end <= meta->data_begin) {
        return -1;
    }
    return 0;
}

static int parse_header_tensors(const char *json, size_t json_len,
                                std::vector<SafetensorsMeta> *out) {
    out->clear();
    JsonCursor cursor{json, json + json_len};
    if (expect_char(&cursor, '{') != 0) {
        return -1;
    }
    skip_ws(&cursor);
    if (cursor.p < cursor.end && *cursor.p == '}') {
        return -1;
    }

    while (cursor.p < cursor.end) {
        std::string name;
        if (parse_string(&cursor, &name) != 0) {
            return -1;
        }
        if (expect_char(&cursor, ':') != 0) {
            return -1;
        }
        if (name.rfind("__", 0) == 0) {
            if (skip_json_value(&cursor) != 0) {
                return -1;
            }
        } else {
            if (static_cast<int>(out->size()) >= kMaxSafetensors) {
                return -1;
            }
            SafetensorsMeta meta;
            meta.name = name;
            if (parse_tensor_entry(&cursor, &meta) != 0) {
                return -1;
            }
            out->push_back(std::move(meta));
        }
        skip_ws(&cursor);
        if (cursor.p >= cursor.end) {
            return -1;
        }
        if (*cursor.p == '}') {
            ++cursor.p;
            return out->empty() ? -1 : 0;
        }
        if (*cursor.p != ',') {
            return -1;
        }
        ++cursor.p;
    }
    return -1;
}

static char *duplicate_c_string(const std::string &text) {
    char *copy = static_cast<char *>(std::malloc(text.size() + 1));
    if (copy == nullptr) {
        return nullptr;
    }
    std::memcpy(copy, text.c_str(), text.size() + 1);
    return copy;
}

static int64_t shape_numel(const std::vector<int64_t> &shape) {
    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return -1;
        }
        numel *= dim;
    }
    return numel;
}

static int meta_to_host_tensor(const SafetensorsMeta &meta, const uint8_t *data_blob,
                               size_t data_blob_size, HostTensor *tensor) {
    if (meta.dtype != "F32") {
        std::fprintf(stderr, "safetensors_read_file: unsupported dtype %s for tensor %s\n",
                     meta.dtype.c_str(), meta.name.c_str());
        return -1;
    }
    if (meta.data_end > data_blob_size) {
        return -1;
    }
    const int64_t numel = shape_numel(meta.shape);
    if (numel < 0) {
        return -1;
    }
    const size_t byte_len = static_cast<size_t>(meta.data_end - meta.data_begin);
    if (byte_len != static_cast<size_t>(numel) * sizeof(float)) {
        return -1;
    }

    std::memset(tensor, 0, sizeof(*tensor));
    tensor->name = duplicate_c_string(meta.name);
    if (tensor->name == nullptr) {
        return -1;
    }
    tensor->ndim = static_cast<int>(meta.shape.size());
    if (tensor->ndim <= 0 || tensor->ndim > 8) {
        free(tensor->name);
        return -1;
    }
    tensor->dims =
        static_cast<int64_t *>(std::malloc(static_cast<size_t>(tensor->ndim) * sizeof(int64_t)));
    if (tensor->dims == nullptr) {
        free(tensor->name);
        return -1;
    }
    for (int i = 0; i < tensor->ndim; ++i) {
        tensor->dims[i] = meta.shape[static_cast<size_t>(i)];
    }
    tensor->data = static_cast<float *>(std::malloc(byte_len));
    if (tensor->data == nullptr) {
        free(tensor->name);
        free(tensor->dims);
        return -1;
    }
    std::memcpy(tensor->data, data_blob + meta.data_begin, byte_len);
    tensor->dtype = WEIGHT_DTYPE_F32;
    return 0;
}

} // namespace

// 读 .safetensors 二进制，解析 header + F32 data blob，产出 HostTensor 表（名称原样，无 HF 映射）。
//
// Big picture 里它在哪？
//   weight_loader_load_safetensors() -> [本函数] -> HostTensor[] -> weight_load_result_destroy 释放
//   本函数 own：calloc 出的 HostTensor 数组及其 name/dims/data；不 own ModelConfig（由 loader 另读
//   config.txt） 与 weight_loader_load_fixture 对照：fixture 走 manifest.txt + 分散
//   .f32；本函数走单文件 safetensors 格式 图纸：doc/design/phase2_lifecycle.md §1（WeightLoader
//   safetensors 步骤 1）
//
// 函数内部顺序（逐步）：
//   例：文件含 2 个 F32 tensor — embed shape [2,2]（16B），layer0.w_q shape [4]（16B）
//   step 1. 打开 path，读前 8 字节 LE uint64 -> header_size；校验 file_size >= 8 + header_size
//   step 2. 读 header_size 字节 JSON -> parse_header_tensors（跳过 __metadata__，收集
//   dtype/shape/data_offsets） step 3. 读剩余字节 -> data_blob（与 header 里 offsets 相对） step 4.
//   calloc(metas.size()) HostTensor；逐条 meta_to_host_tensor（仅 F32，memcpy 到 tensor->data）
//   step 5. *out_tensors / *out_num_tensors 交给调用方；失败时释放已分配项并返回 -1
//
// 调用契约：out_tensors 由调用方 free（经 weight_load_result_destroy）；path/out 指针非空；0 成功 /
// -1 失败
extern "C" int safetensors_read_file(const char *path, HostTensor **out_tensors,
                                     int *out_num_tensors) {
    // step 0：参数与输出初值
    if (path == nullptr || out_tensors == nullptr || out_num_tensors == nullptr) {
        return -1;
    }
    *out_tensors = nullptr;
    *out_num_tensors = 0;

    // step 1：打开文件，读 header 长度
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return -1;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    if (file_size < 8) {
        return -1;
    }
    in.seekg(0, std::ios::beg);

    uint64_t header_size = 0;
    if (!in.read(reinterpret_cast<char *>(&header_size), sizeof(header_size))) {
        return -1;
    }
    if (header_size == 0 || static_cast<std::streamoff>(8 + header_size) > file_size) {
        return -1;
    }

    // step 2：读 JSON header 并解析 tensor 元数据
    std::string header_json(static_cast<size_t>(header_size), '\0');
    if (!in.read(header_json.data(), static_cast<std::streamsize>(header_size))) {
        return -1;
    }

    // step 3：读 data blob（offsets 相对 blob 起点）
    const std::streamoff data_size = file_size - static_cast<std::streamoff>(8 + header_size);
    if (data_size < 0) {
        return -1;
    }
    std::vector<uint8_t> data_blob(static_cast<size_t>(data_size));
    if (data_size > 0) {
        if (!in.read(reinterpret_cast<char *>(data_blob.data()),
                     static_cast<std::streamsize>(data_size))) {
            return -1;
        }
    }

    std::vector<SafetensorsMeta> metas;
    if (parse_header_tensors(header_json.data(), header_json.size(), &metas) != 0) {
        return -1;
    }

    // step 4：逐 tensor 拷贝到 HostTensor
    HostTensor *tensors = static_cast<HostTensor *>(std::calloc(metas.size(), sizeof(HostTensor)));
    if (tensors == nullptr) {
        return -1;
    }

    for (size_t i = 0; i < metas.size(); ++i) {
        if (meta_to_host_tensor(metas[i], data_blob.data(), data_blob.size(), &tensors[i]) != 0) {
            for (size_t j = 0; j < i; ++j) {
                free(tensors[j].name);
                free(tensors[j].data);
                free(tensors[j].dims);
            }
            free(tensors);
            return -1;
        }
    }

    // step 5：输出所有权交给调用方
    *out_tensors = tensors;
    *out_num_tensors = static_cast<int>(metas.size());
    return 0;
}
