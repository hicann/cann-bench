#include <tuple>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor> group_norm_silu_quant(
    at::Tensor& self,
    at::Tensor& gamma,
    at::Tensor& beta,
    at::Tensor& quant_scale,
    int64_t group,
    double eps,
    bool activate_silu) {
    auto out = torch::empty_like(self, self.options().dtype(at::kChar));
    auto stat_shape = std::vector<int64_t>{self.size(0), group};
    auto mean_out = torch::empty(stat_shape, self.options());
    auto rstd_out = torch::empty(stat_shape, self.options());

    EXEC_NPU_CMD(aclnnGroupNormSiluQuant, self, gamma, beta, quant_scale, group, eps, activate_silu, out, mean_out, rstd_out);
    return std::make_tuple(out, mean_out, rstd_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("group_norm_silu_quant", &group_norm_silu_quant, "");
}
