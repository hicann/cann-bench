/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file mish_kernel.cpp
 * \brief Mish kernel - y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
 *
 * Computes in float32 internally for all dtypes; casts half/bfloat16 I/O.
 * Math + data flow match the aclnn NsMish::Mish<T, BUFFER_MODE> reference;
 * direct-launch just wraps the same algorithm so tiling is done in the
 * plugin instead of via a CANN tiling-data struct.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

// Double-buffered I/O queues; matches aclnn's BUFFER_NUM=2 for totalIdx>1024.
constexpr static int64_t BUFFER_NUM = 2;

template <typename T>
class KernelMish {
public:
    __aicore__ inline KernelMish() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 int64_t totalLength, int64_t blockLength,
                                 uint32_t tileSize)
    {
        int64_t remainder = totalLength - blockLength * AscendC::GetBlockIdx();
        blockLength_ = (remainder > blockLength) ? blockLength : remainder;
        if (blockLength_ < 0) blockLength_ = 0;
        ubLength_ = tileSize;

        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx(), blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx(), blockLength_);

        pipe_.InitBuffer(xQueue_,   BUFFER_NUM, ubLength_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        uint32_t fBytes = ubLength_ * sizeof(float);
        pipe_.InitBuffer(expBuf_,      fBytes);
        pipe_.InitBuffer(spBuf_,       fBytes);
        pipe_.InitBuffer(thBuf_,       fBytes);
        pipe_.InitBuffer(xFloatBuf_,   fBytes);
        pipe_.InitBuffer(outFloatBuf_, fBytes);
    }

    __aicore__ inline void Process()
    {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1))
                ? (blockLength_ - ubLength_ * i)
                : static_cast<int64_t>(ubLength_);
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        auto xLocal = xQueue_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[progress * ubLength_], copyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    // Mish in float32: y = x * tanh(ln(1 + exp(x)))
    __aicore__ inline void MishF32(AscendC::LocalTensor<float> xF,
                                    AscendC::LocalTensor<float> outF,
                                    int64_t count)
    {
        auto expLocal = expBuf_.Get<float>();
        auto spLocal  = spBuf_.Get<float>();
        auto thLocal  = thBuf_.Get<float>();
        AscendC::Exp(expLocal, xF, count);
        AscendC::Adds(spLocal, expLocal, 1.0f, count);
        AscendC::Ln(spLocal, spLocal, count);
        AscendC::Tanh(thLocal, spLocal, count);
        AscendC::Mul(outF, xF, thLocal, count);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal   = xQueue_.template DeQue<T>();
        auto outLocal = outQueue_.template AllocTensor<T>();
        auto xF   = xFloatBuf_.Get<float>();
        auto outF = outFloatBuf_.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            MishF32(xLocal, outLocal, count);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            MishF32(xF, outF, count);
            AscendC::Cast(outLocal, outF, AscendC::RoundMode::CAST_RINT, count);
        } else {
            // half
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            MishF32(xF, outF, count);
            AscendC::Cast(outLocal, outF, AscendC::RoundMode::CAST_NONE, count);
        }

        outQueue_.template EnQue<T>(outLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        auto outLocal = outQueue_.template DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[progress * ubLength_], outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_, spBuf_, thBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, outFloatBuf_;
    int64_t blockLength_ = 0;
    uint32_t ubLength_ = 0;
};

template <typename T>
__global__ __aicore__ void mish_kernel(GM_ADDR x, GM_ADDR y,
                                        int64_t totalLength,
                                        int64_t blockLength,
                                        uint32_t tileSize)
{
    KernelMish<T> op;
    op.Init(x, y, totalLength, blockLength, tileSize);
    op.Process();
}

// Tiling: mirrors aclnn's MishTilingFunc in spirit.
//   blockFactor = ceil(totalIdx / coreNum) aligned up to ubBlockSize elements
//   ubFactor    = floor(ubSize/4 / bufferNum) aligned down to ubBlockSize elements
// bufferNum counts the 5 float32 compute buffers + double-buffered I/O (as fp32).
std::tuple<int64_t, int64_t, int64_t> calc_mish_tiling_params(int64_t totalLength, int64_t dtypeSize)
{
    constexpr static int64_t UB_BLOCK_BYTES   = 32;
    constexpr static int64_t BUFFER_COUNT_FP32 = 7;       // 5 compute + 1 in + 1 out
    constexpr static int64_t DOUBLE_BUFFER_ADD = 4;       // +2 for in-queue, +2 for out-queue
    constexpr static int64_t MIN_SPLIT_THRESHOLD = 1024;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = platform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) ubSize = 192 * 1024;                 // safe fallback for 910b

    int64_t alignElems = UB_BLOCK_BYTES / dtypeSize;
    int64_t blockLength = ((totalLength + coreNum - 1) / coreNum + alignElems - 1) / alignElems * alignElems;
    if (blockLength < alignElems) blockLength = alignElems;
    int64_t numBlocks = (totalLength + blockLength - 1) / blockLength;

    int64_t useDoubleBuffer = (totalLength > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = BUFFER_COUNT_FP32 + (useDoubleBuffer ? DOUBLE_BUFFER_ADD : 0);
    int64_t ubFactor = static_cast<int64_t>(ubSize) / 4 / bufferNum;
    ubFactor = (ubFactor / alignElems) * alignElems;
    if (ubFactor < alignElems) ubFactor = alignElems;

    return std::make_tuple(numBlocks, blockLength, static_cast<int64_t>(ubFactor));
}

extern "C" {

void launch_mish_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                               int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mish_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileSize);
}

void launch_mish_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                              int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mish_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileSize);
}

void launch_mish_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                                  int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mish_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileSize);
}

}
