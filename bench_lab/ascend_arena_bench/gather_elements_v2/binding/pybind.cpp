#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gather_elements_v2_pybind(at::Tensor &x, at::Tensor &index, int64_t dim)
{
    at::Tensor result = torch::empty_like(index, x.options());
    EXEC_NPU_CMD(aclnnGather, x, dim, index, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("gather_elements_v2", &gather_elements_v2_pybind, "");
}
