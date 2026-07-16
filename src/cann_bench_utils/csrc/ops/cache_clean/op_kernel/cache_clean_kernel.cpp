/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file cache_clean_kernel.cpp
 * \brief CannBenchCacheClean kernel - ReduceMax for L2 cache flush
 *
 * Fixed shape: (96, 1024, 1024) -> scalar, fp16
 * Simplified - just reads all data to flush L2 cache
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t BATCH = 96;
constexpr static int64_t HEIGHT = 1024;
constexpr static int64_t WIDTH = 1024;
constexpr static int64_t TOTAL_SIZE = BATCH * HEIGHT * WIDTH;
constexpr static int64_t PIPELINE_DEPTH = 2;

// Simplified cache clean kernel - just reads all data to flush L2
// CRITICAL: Kernel name must be "CannBenchCacheClean" (no template, no underscore)
// so the Profiler CSV Type field shows "CannBenchCacheClean" for filtering
template <typename T>
__global__ __aicore__ void CannBenchCacheClean(GM_ADDR x, GM_ADDR out,
                                                int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, outGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;

    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);

    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    outGm.SetGlobalBuffer((__gm__ T *)out);

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;

    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;

    // Just read all tiles to flush cache - don't need actual max value
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementNumPerTile * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);

        xLocal = inQueueX.DeQue<T>();
        inQueueX.FreeTensor(xLocal);
    }

    // Block 0 writes a dummy scalar output
    if (AscendC::GetBlockIdx() == 0) {
        auto dummyOut = inQueueX.AllocTensor<T>();
        AscendC::Duplicate(dummyOut, static_cast<T>(0.0f), 1);

        AscendC::DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(outGm, dummyOut, copyParams);

        inQueueX.FreeTensor(dummyOut);
    }
}

// Tiling calculation
std::tuple<int64_t, int64_t, int64_t> calc_cache_clean_tiling_params()
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 2048;
    constexpr static int64_t BUFFER_NUM = 2;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int64_t numBlocks = std::min(coreNum, (TOTAL_SIZE + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (TOTAL_SIZE + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;

    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// Launch wrapper
extern "C" {

void launch_cache_clean_kernel_half(GM_ADDR x, GM_ADDR out, int64_t totalLength,
                                     int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    CannBenchCacheClean<half><<<numBlocks, nullptr, stream>>>(x, out, totalLength, blockLength, tileSize);
}

} // extern "C"
