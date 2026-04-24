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
 * \file sqrt_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef SQRT_LAUNCH_H
#define SQRT_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength);

extern "C" {
void launch_sqrt_kernel_float   (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_half    (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

#endif // SQRT_LAUNCH_H