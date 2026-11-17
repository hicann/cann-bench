#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <tuple>
#include <string>

#include "pytorch_npu_helper.hpp"

namespace py = pybind11;

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    at::Tensor &query,
    at::Tensor &key,
    at::Tensor &cos,
    at::Tensor &sin,
    int64_t layout,
    const std::string &rotary_mode)
{
    auto query_out = query.contiguous();
    auto key_out = key.contiguous();
    auto *rotary_mode_ptr = const_cast<char *>(rotary_mode.c_str());

    EXEC_NPU_CMD(aclnnApplyRotaryPosEmbV2, query_out, key_out, cos, sin, layout, rotary_mode_ptr);
    return std::make_tuple(query_out, key_out);
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def(
        "custom_pybind_api",
        &custom_pybind_api,
        py::arg("query"),
        py::arg("key"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("layout") = 1,
        py::arg("rotary_mode") = "half");
}
