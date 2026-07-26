#include "rope.h"
#include "rope_cossin_cache.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(rope_global_cache_me, m) {
    m.doc() = "RoPE with model-owned global cos/sin cache";

    m.def(
        "create_cossin_cache",
        [](int max_len, int n_dims, float freq_base) -> uintptr_t {
            RopeCosSinCache *cache = rope_cossin_cache_create(max_len, n_dims, freq_base);
            if (cache == nullptr) {
                throw std::runtime_error("rope_cossin_cache_create failed");
            }
            return reinterpret_cast<uintptr_t>(cache);
        },
        py::arg("max_len"), py::arg("n_dims"), py::arg("freq_base") = 10000.f,
        "Create model-owned cos/sin cache for positions [0, max_len)");

    m.def(
        "destroy_cossin_cache",
        [](uintptr_t cache_handle) {
            rope_cossin_cache_destroy(reinterpret_cast<RopeCosSinCache *>(cache_handle));
        },
        py::arg("cache_handle"), "Destroy model-owned cos/sin cache");

    m.def(
        "forward_host",
        [](uintptr_t cache_handle, py::array_t<float> input, py::array_t<int> pos,
           py::array_t<float> output, int head_dim, int num_heads, int num_tokens, int batch) {
            auto *cache = reinterpret_cast<RopeCosSinCache *>(cache_handle);
            float *input_ptr = static_cast<float *>(input.request().ptr);
            int *pos_ptr = static_cast<int *>(pos.request().ptr);
            float *output_ptr = static_cast<float *>(output.request().ptr);
            rope_neox_forward_host(input_ptr, pos_ptr, output_ptr, head_dim, num_heads, num_tokens,
                                   batch, cache);
        },
        py::arg("cache_handle"), py::arg("input"), py::arg("pos"), py::arg("output"),
        py::arg("head_dim"), py::arg("num_heads"), py::arg("num_tokens"), py::arg("batch"),
        "Test wrapper: host I/O around rope_neox_forward_device");
}
