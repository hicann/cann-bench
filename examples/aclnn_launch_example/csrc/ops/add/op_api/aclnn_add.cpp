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
 * @file aclnn_add.cpp
 * @brief ACLNN L2 API implementation
 */

#include "aclnn_add.h"
#include "add.h"
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
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, DTYPE_SUPPORT);
}

static bool HasEmptyTensor(const aclTensor* x1, const aclTensor* x2)
{
    return x1->IsEmpty() || x2->IsEmpty();
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_DTYPE_NOT_MATCH(x1, x2->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, x1->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    if (!IsDtypeSupported(x1->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dtype not supported: %d", static_cast<int>(x1->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_CHECK_MAX_DIM(x1, ACLNN_MAX_SHAPE_RANK, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(x2, ACLNN_MAX_SHAPE_RANK, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdd, DFX_IN(x1, x2), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x1, x2)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Add(x1Contiguous, x2Contiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAdd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}