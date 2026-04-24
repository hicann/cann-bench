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
 * \file add_host.cpp
 * \brief Add host code - kernel launch and tiling (compiled with bisheng + -xasc)
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

// Add kernel - v2
template <typename T>
__global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementNumPerTile * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX.AllocTensor<T>();
        auto yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        auto zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, elementNumPerTile);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailTileElementNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX.AllocTensor<T>();
        auto yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        auto zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}

// Tiling function
std::tuple<int64_t, int64_t, int64_t> calc_add_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t BUFFER_NUM = 3;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_add_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    add_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

void launch_add_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    add_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

void launch_add_kernel_int32(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    add_kernel<int32_t><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

}