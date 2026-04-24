/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef MISH_LAUNCH_H
#define MISH_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_mish_tiling_params(int64_t totalLength, int64_t dtypeSize);

extern "C" {
void launch_mish_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                               int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_mish_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                              int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_mish_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                                  int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

#endif // MISH_LAUNCH_H
