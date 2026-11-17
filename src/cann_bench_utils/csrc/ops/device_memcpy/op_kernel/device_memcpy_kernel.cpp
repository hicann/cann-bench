/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file device_memcpy_kernel.cpp
 * \brief CannBenchDeviceMemcpy — raw byte-level d2d copy
 *
 * Pipelined GM→UB→GM via uint8_t.  AscendC::DataCopy (UB↔UB compute) requires
 * 32B-aligned element counts; GM reads/writes via DataCopyPad (DMA) use exact
 * sizes.  Tail tiles read only valid bytes from GM, pad the compute step to 32B,
 * and write back exact bytes.  pipe_barrier(PIPE_ALL) between inQ and outQ
 * ensures correct pipeline synchronization.
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;
constexpr static int64_t BUFFER_NUM = 2;
constexpr static int64_t VEC = 32;

using byte_t = uint8_t;

__global__ __aicore__ void CannBenchDeviceMemcpy(GM_ADDR src, GM_ADDR dst,
                                                  int64_t totalBytes,
                                                  int64_t blockBytes,
                                                  uint32_t tileSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<byte_t> srcGm, dstGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQ;

    pipe.InitBuffer(inQ, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQ, PIPELINE_DEPTH, tileSize);

    int64_t start = blockBytes * AscendC::GetBlockIdx();
    srcGm.SetGlobalBuffer((__gm__ byte_t *)src + start);
    dstGm.SetGlobalBuffer((__gm__ byte_t *)dst + start);

    int64_t cur = totalBytes - start;
    if (cur > blockBytes) cur = blockBytes;

    int64_t tileSz = tileSize;
    AscendC::DataCopyPadExtParams<byte_t> pp{false, 0, 0, 0};

    for (int64_t off = 0; off < cur; off += tileSz) {
        int64_t readLen  = (off + tileSz <= cur) ? tileSz : (cur - off);
        int64_t compLen  = ((readLen + VEC - 1) / VEC) * VEC;
        int64_t writeLen = readLen;

        AscendC::DataCopyExtParams rp{1, static_cast<uint32_t>(readLen),  0, 0, 0};
        AscendC::DataCopyExtParams wp{1, static_cast<uint32_t>(writeLen), 0, 0, 0};

        auto s = inQ.AllocTensor<byte_t>();
        AscendC::DataCopyPad(s, srcGm[off], rp, pp);
        inQ.EnQue(s);
        s = inQ.DeQue<byte_t>();
        pipe_barrier(PIPE_ALL);
        auto d = outQ.AllocTensor<byte_t>();
        AscendC::DataCopy(d, s, compLen);
        outQ.EnQue(d);
        inQ.FreeTensor(s);
        pipe_barrier(PIPE_ALL);
        d = outQ.DeQue<byte_t>();
        AscendC::DataCopyPad(dstGm[off], d, wp);
        outQ.FreeTensor(d);
    }
}

std::tuple<int64_t, int64_t, int64_t> calc_device_memcpy_tiling_params(int64_t totalBytes)
{
    constexpr static int64_t MIN_BYTES_PER_CORE = 8192;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = plat->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t blocks = std::min(coreNum, (totalBytes + MIN_BYTES_PER_CORE - 1) / MIN_BYTES_PER_CORE);
    int64_t blkBytes = (totalBytes + blocks - 1) / blocks;
    int64_t tileSz = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(blocks, blkBytes, tileSz);
}

extern "C" {

void launch_device_memcpy_kernel(GM_ADDR src, GM_ADDR dst, int64_t totalBytes,
                                  int64_t blocks, int64_t blkBytes,
                                  uint32_t tileSz, void *stream) {
    CannBenchDeviceMemcpy<<<blocks, nullptr, stream>>>(src, dst, totalBytes, blkBytes, tileSz);
}

} // extern "C"
