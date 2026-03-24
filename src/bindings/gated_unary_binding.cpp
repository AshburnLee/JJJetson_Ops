#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" void relu_gated(const float * src0, const float * src1, float * dst, 
                           const std::vector<int>& dst_dims,
                           const std::vector<int>& src0_nb, 
                           const std::vector<int>& src1_nb);
extern "C" void gelu_gated(const float * src0, const float * src1, float * dst, 
                           const std::vector<int>& dst_dims,
                           const std::vector<int>& src0_nb, 
                           const std::vector<int>& src1_nb);
extern "C" void silu_gated(const float * src0, const float * src1, float * dst, 
                           const std::vector<int>& dst_dims,
                           const std::vector<int>& src0_nb, 
                           const std::vector<int>& src1_nb);
extern "C" void gelu_erf_gated(const float * src0, const float * src1, float * dst, 
                           const std::vector<int>& dst_dims,
                           const std::vector<int>& src0_nb, 
                           const std::vector<int>& src1_nb);
extern "C" void gelu_quick_gated(const float * src0, const float * src1, float * dst, 
                           const std::vector<int>& dst_dims,
                           const std::vector<int>& src0_nb, 
                           const std::vector<int>& src1_nb);



PYBIND11_MODULE(gated_unary_me, m) {
    m.doc() = "python binding for CUDA copy gated unary kernel";
    m.def("relu_gated", [](py::buffer src0, py::buffer src1,
                           py::buffer dst,
                        const std::vector<int> dst_dims,
                        const std::vector<int> src0_nb,
                        const std::vector<int> src1_nb){

        auto src0_buff = src0.request();
        auto src1_buff = src1.request();
        auto dst_buff = dst.request();

        relu_gated(static_cast<const float*>(src0_buff.ptr), 
                    static_cast<const float*>(src1_buff.ptr), 
                    static_cast<float*>(dst_buff.ptr),
                    dst_dims,
                    src0_nb,
                    src1_nb);

    },
    py::arg("src0"),
    py::arg("src1"),
    py::arg("dst"),
    py::arg("dst_dims"),
    py::arg("src0_nb"),
    py::arg("src1_nb")
);
    m.def("gelu_gated", [](py::buffer src0, py::buffer src1,
                           py::buffer dst,
                        const std::vector<int> dst_dims,
                        const std::vector<int> src0_nb,
                        const std::vector<int> src1_nb){

        auto src0_buff = src0.request();
        auto src1_buff = src1.request();
        auto dst_buff = dst.request();

        gelu_gated(static_cast<const float*>(src0_buff.ptr), 
                    static_cast<const float*>(src1_buff.ptr), 
                    static_cast<float*>(dst_buff.ptr),
                    dst_dims,
                    src0_nb,
                    src1_nb);

    },
    py::arg("src0"),
    py::arg("src1"),
    py::arg("dst"),
    py::arg("dst_dims"),
    py::arg("src0_nb"),
    py::arg("src1_nb")
);
    m.def("silu_gated", [](py::buffer src0, py::buffer src1,
                           py::buffer dst,
                        const std::vector<int> dst_dims,
                        const std::vector<int> src0_nb,
                        const std::vector<int> src1_nb){

        auto src0_buff = src0.request();
        auto src1_buff = src1.request();
        auto dst_buff = dst.request();

        silu_gated(static_cast<const float*>(src0_buff.ptr), 
                    static_cast<const float*>(src1_buff.ptr), 
                    static_cast<float*>(dst_buff.ptr),
                    dst_dims,
                    src0_nb,
                    src1_nb);

    },
    py::arg("src0"),
    py::arg("src1"),
    py::arg("dst"),
    py::arg("dst_dims"),
    py::arg("src0_nb"),
    py::arg("src1_nb")
);
    m.def("gelu_erf_gated", [](py::buffer src0, py::buffer src1,
                           py::buffer dst,
                        const std::vector<int> dst_dims,
                        const std::vector<int> src0_nb,
                        const std::vector<int> src1_nb){

        auto src0_buff = src0.request();
        auto src1_buff = src1.request();
        auto dst_buff = dst.request();

        gelu_erf_gated(static_cast<const float*>(src0_buff.ptr), 
                    static_cast<const float*>(src1_buff.ptr), 
                    static_cast<float*>(dst_buff.ptr),
                    dst_dims,
                    src0_nb,
                    src1_nb);

    },
    py::arg("src0"),
    py::arg("src1"),
    py::arg("dst"),
    py::arg("dst_dims"),
    py::arg("src0_nb"),
    py::arg("src1_nb")
);
    m.def("gelu_quick_gated", [](py::buffer src0, py::buffer src1,
                           py::buffer dst,
                        const std::vector<int> dst_dims,
                        const std::vector<int> src0_nb,
                        const std::vector<int> src1_nb){

        auto src0_buff = src0.request();
        auto src1_buff = src1.request();
        auto dst_buff = dst.request();

        gelu_quick_gated(static_cast<const float*>(src0_buff.ptr), 
                    static_cast<const float*>(src1_buff.ptr), 
                    static_cast<float*>(dst_buff.ptr),
                    dst_dims,
                    src0_nb,
                    src1_nb);

    },
    py::arg("src0"),
    py::arg("src1"),
    py::arg("dst"),
    py::arg("dst_dims"),
    py::arg("src0_nb"),
    py::arg("src1_nb")
);
}
