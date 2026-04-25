// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef MISH_H
#define MISH_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_tiling_data.h"
#include "mish_tiling_key.h"

namespace NsMish {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class Mish {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Mish(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void MishF32(LocalTensor<float> xF,
                                    LocalTensor<float> outF,
                                    int64_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // 5 float32 compute buffers (non-queued)
    TBuf<QuePosition::VECCALC> expBuf, spBuf, thBuf, xFloatBuf, outFloatBuf;
    GlobalTensor<T> inputGMX, outputGMY;
    int64_t blockLength_ = 0, ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    uint32_t fBytes = ubLength_ * sizeof(float);
    pipe.InitBuffer(expBuf,      fBytes);
    pipe.InitBuffer(spBuf,       fBytes);
    pipe.InitBuffer(thBuf,       fBytes);
    pipe.InitBuffer(xFloatBuf,   fBytes);
    pipe.InitBuffer(outFloatBuf, fBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::MishF32(LocalTensor<float> xF,
                                                     LocalTensor<float> outF,
                                                     int64_t count)
{
    auto expLocal = expBuf.Get<float>();
    auto spLocal  = spBuf.Get<float>();
    auto thLocal  = thBuf.Get<float>();
    AscendC::Exp(expLocal, xF, count);
    AscendC::Adds(spLocal, expLocal, 1.0f, count);
    AscendC::Ln(spLocal, spLocal, count);
    AscendC::Tanh(thLocal, spLocal, count);
    AscendC::Mul(outF, xF, thLocal, count);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal   = inputQueueX.template DeQue<T>();
    LocalTensor<T> outLocal = outputQueueY.template AllocTensor<T>();
    auto xF   = xFloatBuf.Get<float>();
    auto outF = outFloatBuf.Get<float>();

    if constexpr (std::is_same<T, float>::value) {
        MishF32(xLocal, outLocal, currentNum);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        MishF32(xF, outF, currentNum);
        AscendC::Cast(outLocal, outF, AscendC::RoundMode::CAST_RINT, currentNum);
    } else {
        // half
        AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        MishF32(xF, outF, currentNum);
        AscendC::Cast(outLocal, outF, AscendC::RoundMode::CAST_NONE, currentNum);
    }

    outputQueueY.template EnQue<T>(outLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Mish<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsMish
#endif // MISH_H
