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
std::tuple<int64_t, int64_t, int64_t> calc_cache_clean_tiling_params();

// Launch function declaration
extern "C" {
void launch_cache_clean_kernel_half(GM_ADDR x, GM_ADDR out, int64_t totalLength,
                                     int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

// Convenience macro
#define CACHE_CLEAN_KERNEL_LAUNCH_HALF(x, out, len, blocks, blkLen, tileSz, stream) \
    launch_cache_clean_kernel_half(x, out, len, blocks, blkLen, tileSz, stream)
