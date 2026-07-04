#include "rope_cossin_cache.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

extern "C" void rope_with_global_cossin_cache(float *input, int *pos, float *output,
                                              std::vector<int> &input_dims,
                                              const RopeCosSinCache *cache);

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
        "RoPE",
        [](uintptr_t cache_handle, py::array_t<float> input, py::array_t<int> pos,
           py::array_t<float> output, std::vector<int> dims) {
            auto *cache = reinterpret_cast<RopeCosSinCache *>(cache_handle);
            auto input_buf = input.request();
            auto pos_buf = pos.request();
            auto output_buf = output.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            int *pos_ptr = static_cast<int *>(pos_buf.ptr);
            float *output_ptr = static_cast<float *>(output_buf.ptr);

            rope_with_global_cossin_cache(input_ptr, pos_ptr, output_ptr, dims, cache);
        },
        py::arg("cache_handle"), py::arg("input"), py::arg("pos"), py::arg("output"),
        py::arg("input_dims"), "RoPE forward using model-owned global cos/sin cache");
}
