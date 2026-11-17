#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

namespace {

at::Tensor group_quant_pybind(
    at::Tensor x,
    at::Tensor scale,
    at::Tensor group_index,
    c10::optional<at::Tensor> offset,
    int64_t dst_type)
{
    TORCH_CHECK(x.dim() == 2, "group_quant expects x to be 2D");
    TORCH_CHECK(scale.dim() == 2, "group_quant expects scale to be 2D");
    TORCH_CHECK(group_index.dim() == 1, "group_quant expects group_index to be 1D");
    TORCH_CHECK(scale.size(0) == group_index.size(0), "scale.size(0) must equal group_index.size(0)");
    TORCH_CHECK(scale.size(1) == x.size(1), "scale.size(1) must equal x.size(1)");
    TORCH_CHECK(dst_type == 2, "standardized group_quant binding only supports dst_type=2 (INT8)");
    if (offset.has_value()) {
        TORCH_CHECK(offset->numel() == 1, "offset must be scalar or shape [1]");
        TORCH_CHECK(offset->scalar_type() == scale.scalar_type(), "offset dtype must match scale dtype");
    }

    int32_t dst_type_i32 = static_cast<int32_t>(dst_type);
    at::Tensor result = torch::empty(x.sizes().vec(), x.options().dtype(at::kChar));
    EXEC_NPU_CMD(aclnnGroupQuant, x, scale, group_index, offset, dst_type_i32, result);
    return result;
}

}  // namespace

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("group_quant", &group_quant_pybind, "");
}
