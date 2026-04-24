// Add API - v3

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/add_launch.h"

torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y) {
    TORCH_CHECK(x.sizes() == y.sizes(), "shapes must match");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU");

    auto z = torch::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t len = x.numel();
    int64_t blocks, blkLen, tileSz;
    std::tie(blocks, blkLen, tileSz) = calc_add_tiling_params(len);
    void* x_ptr = x.data_ptr();
    void* y_ptr = y.data_ptr();
    void* z_ptr = z.data_ptr();
    auto dtype = x.scalar_type();

    auto launch = [=]() -> int {
        if (dtype == torch::kFloat32) launch_add_kernel_float(x_ptr, y_ptr, z_ptr, len, blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kFloat16) launch_add_kernel_half(x_ptr, y_ptr, z_ptr, len, blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kInt32) launch_add_kernel_int32(x_ptr, y_ptr, z_ptr, len, blocks, blkLen, tileSz, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Add", launch);
    return z;
}