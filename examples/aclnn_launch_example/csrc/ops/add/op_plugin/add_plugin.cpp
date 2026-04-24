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
 * \file add_plugin.cpp
 * \brief Add plugin layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../../_common/aclnn_common.h"

namespace cann_bench {

static torch::Tensor add_meta(const torch::Tensor& x, const torch::Tensor& y) {
    TORCH_CHECK(x.sizes() == y.sizes(),
                "add: shapes must match, got ", x.sizes(), " vs ", y.sizes());
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "add: dtypes must match, got ", x.scalar_type(), " vs ", y.scalar_type());
    return torch::empty_like(x);
}

static torch::Tensor add_npu(const torch::Tensor& x, const torch::Tensor& y) {
    auto x_c = x.contiguous();
    auto y_c = y.contiguous();
    auto z = torch::empty_like(x_c);
    ACLNN_CMD(aclnnAdd, x_c, y_c, z);
    return z;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("add(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("add", add_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("add", add_npu);
}

}  // namespace cann_bench
