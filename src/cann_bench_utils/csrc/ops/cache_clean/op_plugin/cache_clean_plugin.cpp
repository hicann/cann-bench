/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file cache_clean_plugin.cpp
 * \brief CannBenchCacheClean torch binding - ReduceMax for L2 cache flush
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/cache_clean_launch.h"

namespace cann_bench_utils {

TORCH_LIBRARY_FRAGMENT(cann_bench_utils, m)
{
    m.def("cann_bench_cache_clean(Tensor x) -> Tensor");
}

torch::Tensor cache_clean_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.dim() == 3, "CannBenchCacheClean expects 3D tensor");
    TORCH_CHECK(x.size(0) == 96 && x.size(1) == 1024 && x.size(2) == 1024,
                "CannBenchCacheClean expects shape (96, 1024, 1024)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "CannBenchCacheClean expects fp16");
    return torch::empty({}, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench_utils, Meta, m)
{
    m.impl("cann_bench_cache_clean", cache_clean_meta);
}

torch::Tensor cache_clean_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto out = cache_clean_meta(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_cache_clean_tiling_params();

    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto out_ptr = (GM_ADDR)out.data_ptr();

    auto acl_call = [=]() -> int {
        CACHE_CLEAN_KERNEL_LAUNCH_HALF(x_ptr, out_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        return 0;
    };

    // CRITICAL: Set kernel name to "CannBenchCacheClean" for profiling filtering
    at_npu::native::OpCommand::RunOpApi("CannBenchCacheClean", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench_utils, PrivateUse1, m)
{
    m.impl("cann_bench_cache_clean", cache_clean_npu);
}

} // namespace cann_bench_utils
