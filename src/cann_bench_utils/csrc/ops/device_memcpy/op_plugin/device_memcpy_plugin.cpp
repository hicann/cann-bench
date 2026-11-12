/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file device_memcpy_plugin.cpp
 * \brief CannBenchDeviceMemcpy torch binding — dtype-agnostic byte copy
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/device_memcpy_launch.h"

namespace cann_bench_utils {

void device_zero_npu(const torch::Tensor &dst)
{
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                "device_zero requires an NPU tensor");
    TORCH_CHECK(dst.is_contiguous(), "device_zero requires a contiguous tensor");
    c10_npu::NPUGuard guard(dst.device().index());
    const auto bytes = dst.numel() * dst.element_size();
    if (bytes == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto ptr = dst.data_ptr();
    auto acl_call = [=]() -> int {
        return aclrtMemsetAsync(ptr, bytes, 0, bytes, stream);
    };
    // Runtime memset works even when anti-cheat removes the built-in op tree.
    // The caller must synchronize before releasing or reusing this storage.
    at_npu::native::OpCommand::RunOpApi("CannBenchDeviceZero", acl_call);
}

static void device_memcpy_check(const torch::Tensor &src, const torch::Tensor &dst)
{
    TORCH_CHECK(src.device() == dst.device(), "src and dst must be on same device");
    TORCH_CHECK(src.sizes() == dst.sizes(), "src and dst must have same shape");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src and dst must have same dtype");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(),
                "src and dst must be contiguous for linear byte copy");
}

void device_memcpy_meta(const torch::Tensor &src, const torch::Tensor &dst)
{
    device_memcpy_check(src, dst);
}

void device_memcpy_npu(const torch::Tensor &src, const torch::Tensor &dst)
{
    c10_npu::NPUGuard guard(src.device().index());
    device_memcpy_check(src, dst);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t totalBytes = src.numel() * static_cast<int64_t>(src.element_size());
    int64_t numBlocks, blockBytes, tileSize;
    std::tie(numBlocks, blockBytes, tileSize) = calc_device_memcpy_tiling_params(totalBytes);

    auto srcPtr = (GM_ADDR)src.data_ptr();
    auto dstPtr = (GM_ADDR)dst.data_ptr();

    auto acl_call = [=]() -> int {
        DEVICE_MEMCPY_KERNEL_LAUNCH(srcPtr, dstPtr, totalBytes,
                                     numBlocks, blockBytes, tileSize, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("CannBenchDeviceMemcpy", acl_call);
}

} // namespace cann_bench_utils
