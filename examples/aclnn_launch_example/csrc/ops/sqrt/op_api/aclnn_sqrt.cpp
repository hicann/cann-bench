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

/**
 * @file aclnn_sqrt.cpp
 * @brief ACLNN L2 API implementation
 */

#include "aclnn_sqrt.h"
#include "sqrt.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16
};

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    if (!CheckType(x->GetDataType(), DTYPE_SUPPORT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dtype not supported: %d", static_cast<int>(x->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSqrtGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSqrt, DFX_IN(x), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Sqrt(xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSqrt(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSqrt);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}