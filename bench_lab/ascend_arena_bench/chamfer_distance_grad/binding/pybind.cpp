#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

namespace {

pybind11::tuple chamfer_distance_grad(
    at::Tensor& xyz1,
    at::Tensor& xyz2,
    at::Tensor& idx1,
    at::Tensor& idx2,
    at::Tensor& grad_dist1,
    at::Tensor& grad_dist2)
{
    TORCH_CHECK(xyz1.dim() == 3 && xyz1.size(-1) == 2, "xyz1 must have shape (B, N, 2)");
    TORCH_CHECK(xyz2.dim() == 3 && xyz2.size(-1) == 2, "xyz2 must have shape (B, K, 2)");
    TORCH_CHECK(idx1.dim() == 2 && idx2.dim() == 2, "idx1 and idx2 must have shape (B, N)");
    TORCH_CHECK(grad_dist1.dim() == 2 && grad_dist2.dim() == 2, "grad_dist1 and grad_dist2 must have shape (B, N)");
    TORCH_CHECK(xyz1.scalar_type() == xyz2.scalar_type(), "xyz1 and xyz2 must share dtype");
    TORCH_CHECK(grad_dist1.scalar_type() == xyz1.scalar_type(), "grad_dist1 must match xyz1 dtype");
    TORCH_CHECK(grad_dist2.scalar_type() == xyz2.scalar_type(), "grad_dist2 must match xyz2 dtype");
    TORCH_CHECK(idx1.scalar_type() == at::kInt && idx2.scalar_type() == at::kInt, "idx1/idx2 must be int32");

    at::Tensor grad_xyz1 = torch::zeros_like(xyz1);
    at::Tensor grad_xyz2 = torch::zeros_like(xyz2);
    EXEC_NPU_CMD(aclnnChamferDistanceBackward, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2);
    return pybind11::make_tuple(grad_xyz1, grad_xyz2);
}

}  // namespace

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("chamfer_distance_grad", &chamfer_distance_grad, "");
}
