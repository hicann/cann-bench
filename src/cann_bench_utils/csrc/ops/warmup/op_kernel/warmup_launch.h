/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#pragma once

#include <tuple>
#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling parameters calculation
std::tuple<int64_t, int64_t, int64_t> calc_warmup_tiling_params();

// Launch function declaration
extern "C" {
void launch_warmup_kernel_half(GM_ADDR a, GM_ADDR b, GM_ADDR c, int64_t totalLength,
                                int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

// Convenience macro
#define WARMUP_KERNEL_LAUNCH_HALF(a, b, c, len, blocks, blkLen, tileSz, stream) \
    launch_warmup_kernel_half(a, b, c, len, blocks, blkLen, tileSz, stream)
