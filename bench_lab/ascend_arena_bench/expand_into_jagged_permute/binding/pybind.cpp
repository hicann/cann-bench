#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

at::Tensor expand_into_jagged_permute_pybind(
    at::Tensor& permute,
    at::Tensor& input_offsets,
    at::Tensor& output_offsets,
    int64_t output_size)
{
    at::Tensor result = torch::empty({output_size}, permute.options());
    EXEC_NPU_CMD(aclnnExpandIntoJaggedPermute, permute, input_offsets, output_offsets, output_size, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("expand_into_jagged_permute", &expand_into_jagged_permute_pybind, "");
}
