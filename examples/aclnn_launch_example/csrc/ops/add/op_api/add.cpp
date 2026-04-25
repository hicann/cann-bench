/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file add.cpp
 * @brief ACLNN L0 API implementation
 */

#include "add.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Add);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32
};

static bool IsAiCoreSupport(const aclTensor* x1, const aclTensor* x2)
{
    return CheckType(x1->GetDataType(), DTYPE_SUPPORT) &&
           CheckType(x2->GetDataType(), DTYPE_SUPPORT);
}

static bool AddInferShape(const op::Shape& x1Shape, const op::Shape& x2Shape, op::Shape& outShape)
{
    if (!BroadcastInferShape(x1Shape, x2Shape, outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape broadcast failed.");
        return false;
    }
    return true;
}

static const aclTensor* AddAiCore(const aclTensor* x1, const aclTensor* x2,
                                   const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AddAiCore, x1, x2, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Add, OP_INPUT(x1, x2), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddAiCore failed."), return nullptr);
    return out;
}

const aclTensor* Add(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AddInferShape(x1->GetViewShape(), x2->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x1, x2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Add not supported: dtype=%d", static_cast<int>(x1->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, x1->GetDataType());
    return AddAiCore(x1, x2, out, executor);
}

} // namespace l0op