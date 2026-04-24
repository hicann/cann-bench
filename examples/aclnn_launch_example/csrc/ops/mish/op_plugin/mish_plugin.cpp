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

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../../_common/aclnn_common.h"

namespace cann_bench {

static torch::Tensor mish_meta(const torch::Tensor& x) {
    return torch::empty_like(x);
}

static torch::Tensor mish_npu(const torch::Tensor& x) {
    auto x_c = x.contiguous();
    auto y = torch::empty_like(x_c);
    ACLNN_CMD(aclnnMish, x_c, y);
    return y;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("mish(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("mish", mish_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("mish", mish_npu);
}

}  // namespace cann_bench
