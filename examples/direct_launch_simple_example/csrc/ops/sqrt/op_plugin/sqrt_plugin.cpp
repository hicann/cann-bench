/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/sqrt_launch.h"

torch::Tensor sqrt_npu(const torch::Tensor &x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU");
    auto z = torch::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t blocks, blkLen, tileSz;
    std::tie(blocks, blkLen, tileSz) = calc_sqrt_tiling_params(x.numel());
    void* x_ptr = x.data_ptr();
    void* z_ptr = z.data_ptr();
    auto dtype = x.scalar_type();

    auto launch = [=]() -> int {
        if      (dtype == torch::kFloat32) launch_sqrt_kernel_float   (x_ptr, z_ptr, x.numel(), blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kFloat16) launch_sqrt_kernel_half    (x_ptr, z_ptr, x.numel(), blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kBFloat16) launch_sqrt_kernel_bfloat16(x_ptr, z_ptr, x.numel(), blocks, blkLen, tileSz, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Sqrt", launch);
    return z;
}