/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/mish_launch.h"

torch::Tensor mish_npu(const torch::Tensor &x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU");
    auto y = torch::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t len = x.numel();
    auto dtype = x.scalar_type();
    int64_t dtypeSize = x.element_size();
    int64_t blocks, blkLen, tileSz;
    std::tie(blocks, blkLen, tileSz) = calc_mish_tiling_params(len, dtypeSize);
    void* x_ptr = x.data_ptr();
    void* y_ptr = y.data_ptr();

    auto launch = [=]() -> int {
        if (dtype == torch::kFloat32)
            launch_mish_kernel_float(x_ptr, y_ptr, len, blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kFloat16)
            launch_mish_kernel_half(x_ptr, y_ptr, len, blocks, blkLen, tileSz, stream);
        else if (dtype == torch::kBFloat16)
            launch_mish_kernel_bfloat16(x_ptr, y_ptr, len, blocks, blkLen, tileSz, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Mish", launch);
    return y;
}
