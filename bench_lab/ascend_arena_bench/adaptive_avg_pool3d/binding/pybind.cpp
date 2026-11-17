#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <vector>

#include <pybind11/stl.h>
#include "pytorch_npu_helper.hpp"

namespace {

std::vector<int64_t> infer_output_shape(const at::Tensor& x, const std::vector<int64_t>& output_size) {
    TORCH_CHECK(x.dim() == 4 || x.dim() == 5, "adaptive_avg_pool3d expects 4D or 5D input");
    TORCH_CHECK(output_size.size() == 3, "output_size must contain exactly 3 integers");
    auto sizes = x.sizes();
    if (x.dim() == 4) {
        return {sizes[0], output_size[0], output_size[1], output_size[2]};
    }
    return {sizes[0], sizes[1], output_size[0], output_size[1], output_size[2]};
}

}  // namespace

at::Tensor custom_pybind_api(at::Tensor& x, const std::vector<int64_t>& output_size) {
    at::IntArrayRef output_size_ref(output_size.data(), static_cast<long>(output_size.size()));
    at::Tensor result = torch::empty(infer_output_shape(x, output_size), x.options());
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3d, x, output_size_ref, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
