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
 * \file add_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef ADD_LAUNCH_H
#define ADD_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration
std::tuple<int64_t, int64_t, int64_t> calc_add_tiling_params(int64_t totalLength);

// Launch function declarations
extern "C" {
void launch_add_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_add_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_add_kernel_int32(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

// Convenience macros for API layer
#define ADD_KERNEL_LAUNCH_FLOAT(x, y, z, len, blocks, blkLen, tileSz, stream) \
    launch_add_kernel_float(x, y, z, len, blocks, blkLen, tileSz, stream)

#define ADD_KERNEL_LAUNCH_HALF(x, y, z, len, blocks, blkLen, tileSz, stream) \
    launch_add_kernel_half(x, y, z, len, blocks, blkLen, tileSz, stream)

#define ADD_KERNEL_LAUNCH_INT32(x, y, z, len, blocks, blkLen, tileSz, stream) \
    launch_add_kernel_int32(x, y, z, len, blocks, blkLen, tileSz, stream)

#endif // ADD_LAUNCH_H