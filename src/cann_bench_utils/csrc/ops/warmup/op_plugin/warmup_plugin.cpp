/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file warmup_plugin.cpp
 * \brief CannBenchWarmup torch binding - MatMul warmup for freq boost
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/warmup_launch.h"

namespace cann_bench_utils {

TORCH_LIBRARY_FRAGMENT(cann_bench_utils, m)
{
    m.def("cann_bench_warmup(Tensor x, Tensor y) -> Tensor");
}

torch::Tensor warmup_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "CannBenchWarmup expects 2D tensors");
    TORCH_CHECK(x.size(0) == 10240 && x.size(1) == 10240, "CannBenchWarmup expects x shape (10240, 10240)");
    TORCH_CHECK(y.size(0) == 10240 && y.size(1) == 10240, "CannBenchWarmup expects y shape (10240, 10240)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "CannBenchWarmup expects fp16");
    TORCH_CHECK(y.scalar_type() == torch::kFloat16, "CannBenchWarmup expects fp16");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench_utils, Meta, m)
{
    m.impl("cann_bench_warmup", warmup_meta);
}

torch::Tensor warmup_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = warmup_meta(x, y);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_warmup_tiling_params();

    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        WARMUP_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        return 0;
    };

    // CRITICAL: Set kernel name to "CannBenchWarmup" for profiling filtering
    at_npu::native::OpCommand::RunOpApi("CannBenchWarmup", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench_utils, PrivateUse1, m)
{
    m.impl("cann_bench_warmup", warmup_npu);
}

} // namespace cann_bench_utils
