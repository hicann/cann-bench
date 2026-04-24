/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sqrt_host.cpp
 * \brief Sqrt host code - kernel launch and tiling (compiled with bisheng + -xasc)
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

// For fp16/bf16, AscendC::Sqrt is not directly supported — we cast to fp32
// for compute and cast back. For native fp32, no cast.
template <typename T>
class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
        zGm_.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
        // tileSize is element count (not bytes)
        pipe_.InitBuffer(inQueueX_,  PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, PIPELINE_DEPTH, tileSize * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(zFloatBuf_, tileSize * sizeof(float));
        }
        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.DeQue<T>();
        auto zLocal = outQueueZ_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Sqrt(zLocal, xLocal, count);
        } else {
            auto xF = xFloatBuf_.Get<float>();
            auto zF = zFloatBuf_.Get<float>();
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Sqrt(zF, xF, count);
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(zLocal, zF, roundMode, count);
        }
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto zLocal = outQueueZ_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, zGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, zFloatBuf_;
    int64_t elementNumPerTile_ = 0, tileNum_ = 0, tailTileElementNum_ = 0;
};

template <typename T>
__global__ __aicore__ __vector__ void sqrt_kernel(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    KernelSqrt<T> op;
    op.Init(x, z, totalLength, blockLength, tileSize);
    op.Process();
}

// Returns (numBlocks, blockLength, tileSize) where tileSize is the element
// count per UB tile. For fp32: 2 in/out queues (pipelined) = 4 buffers.
// For fp16/bf16: 2 in/out queues (fp16) + 2 fp32 cast buffers = 4 fp32-equivalent
// buffers. Both cases budget 2048 elements, which fits comfortably in UB on 910b.
std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static uint32_t FIXED_TILE_ELEMS = 2048;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLength, static_cast<int64_t>(FIXED_TILE_ELEMS));
}

extern "C" {

void launch_sqrt_kernel_float(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_half(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

}