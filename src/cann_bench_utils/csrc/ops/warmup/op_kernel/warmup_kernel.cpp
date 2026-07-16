/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file warmup_kernel.cpp
 * \brief CannBenchWarmup kernel - MatMul for NPU freq boost
 *
 * Fixed shape: (10240, 10240) @ (10240, 10240) -> (10240, 10240), fp16
 * Simplified MatMul implementation using element-wise operations
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t M = 10240;
constexpr static int64_t K = 10240;
constexpr static int64_t N = 10240;
constexpr static int64_t PIPELINE_DEPTH = 2;

// Simplified warmup kernel - just reads A and B, writes output
// For warmup purposes, we do element-wise multiply instead of full matmul
// This is sufficient to boost NPU frequency
//
// CRITICAL: Kernel name must be "CannBenchWarmup" (no template, no underscore)
// so the Profiler CSV Type field shows "CannBenchWarmup" for filtering
template <typename T>
__global__ __aicore__ void CannBenchWarmup(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                           int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> aGm, bGm, cGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueC;

    pipe.InitBuffer(inQueueA, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueB, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueC, PIPELINE_DEPTH, tileSize);

    aGm.SetGlobalBuffer((__gm__ T *)a + blockLength * AscendC::GetBlockIdx());
    bGm.SetGlobalBuffer((__gm__ T *)b + blockLength * AscendC::GetBlockIdx());
    cGm.SetGlobalBuffer((__gm__ T *)c + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;

    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;

    // Process full tiles
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementNumPerTile * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto aLocal = inQueueA.AllocTensor<T>();
        auto bLocal = inQueueB.AllocTensor<T>();
        AscendC::DataCopyPad(aLocal, aGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(bLocal, bGm[offset], copyParams, padParams);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);

        aLocal = inQueueA.DeQue<T>();
        bLocal = inQueueB.DeQue<T>();
        auto cLocal = outQueueC.AllocTensor<T>();

        // Simplified: element-wise multiply (enough for warmup)
        AscendC::Mul(cLocal, aLocal, bLocal, elementNumPerTile);

        outQueueC.EnQue(cLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);

        cLocal = outQueueC.DeQue<T>();
        AscendC::DataCopyPad(cGm[offset], cLocal, copyParams);
        outQueueC.FreeTensor(cLocal);
    }
}

// Tiling calculation
std::tuple<int64_t, int64_t, int64_t> calc_warmup_tiling_params()
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 2048;
    constexpr static int64_t BUFFER_NUM = 3;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int64_t totalLength = M * N;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;

    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// Launch wrapper
extern "C" {

void launch_warmup_kernel_half(GM_ADDR a, GM_ADDR b, GM_ADDR c, int64_t totalLength,
                                int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    CannBenchWarmup<half><<<numBlocks, nullptr, stream>>>(a, b, c, totalLength, blockLength, tileSize);
}

} // extern "C"
