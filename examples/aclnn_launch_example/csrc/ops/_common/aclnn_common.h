/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Based on https://gitcode.com/zhangzijie/aclnnwrap (CANN OSL v2,
 * © 2025 Huawei Technologies Co., Ltd.).
 */

/*!
 * \file aclnn_common.h
 * \brief ACLNN_CMD macro and helpers for invoking aclnn ops from a
 *        torch_npu plugin. Symbols are resolved via dlsym exclusively from
 *        libcust_opapi.so (customize bundle). There is no fallback to the
 *        stock libopapi.so — evaluation must use the custom op package.
 */

#ifndef CANN_BENCH_ACLNN_COMMON_H_
#define CANN_BENCH_ACLNN_COMMON_H_

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <functional>
#include <type_traits>
#include <vector>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
    const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storage_dims, uint64_t storage_dims_num,
    void *tensor_data);
using _aclCreateScalar = aclScalar *(*)(void *value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorList = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor *tensor);
using _aclDestroyScalar = int (*)(const aclScalar *scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray *array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray *array);
using _aclDestroyTensorList = int (*)(const aclTensorList *array);

using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = void (*)(void *, bool);
using ReleaseHugeMem = void (*)(void *, bool);

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
    _(at::ScalarType::Byte, ACL_UINT8)               \
    _(at::ScalarType::Char, ACL_INT8)                \
    _(at::ScalarType::Short, ACL_INT16)              \
    _(at::ScalarType::Int, ACL_INT32)                \
    _(at::ScalarType::Long, ACL_INT64)               \
    _(at::ScalarType::Half, ACL_FLOAT16)             \
    _(at::ScalarType::Float, ACL_FLOAT)              \
    _(at::ScalarType::Double, ACL_DOUBLE)            \
    _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED) \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)   \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL)                \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::BFloat16, ACL_BF16)            \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char *GetCustOpApiLibName(void) { return "libcust_opapi.so"; }
inline const char *GetOpApiLibName(void) { return "libopapi.so"; }

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName, int flags = RTLD_LAZY)
{
    auto handler = dlopen(libName, flags);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    // 先加载 libopapi.so（RTLD_GLOBAL），使内部依赖符号（l0op::Contiguous、
    // l0op::ViewCopy 等）全局可见，供 libcust_opapi.so 解析依赖。
    // 但 API 符号（aclnnMish 等）仅从 libcust_opapi.so 获取，不 fallback 到内置。
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName(), RTLD_LAZY | RTLD_GLOBAL);

    static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (custOpApiHandler == nullptr) {
        ASCEND_LOGE("dlopen %s failed, cannot resolve symbol '%s'. "
                     "Ensure the run package is installed correctly. "
                     "dlerror: %s", GetCustOpApiLibName(), apiName, dlerror());
        return nullptr;
    }
    auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGE("symbol '%s' not found in %s. "
                     "The operator may not be included in the custom op package.",
                     apiName, GetCustOpApiLibName());
    }
    return funcAddr;
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    c10::Scalar expScalar;
    const at::Tensor *aclInput = &tensor;
    if (aclInput->scalar_type() == at::ScalarType::Double) {
        expScalar = c10::Scalar(*reinterpret_cast<double *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::Long) {
        expScalar = c10::Scalar(*reinterpret_cast<int64_t *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::Float) {
        expScalar = c10::Scalar(*reinterpret_cast<float *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::Int) {
        expScalar = c10::Scalar(*reinterpret_cast<int *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::Half) {
        expScalar = c10::Scalar(*reinterpret_cast<c10::Half *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
        expScalar = c10::Scalar(*reinterpret_cast<int8_t *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
        expScalar = c10::Scalar(*reinterpret_cast<c10::complex<double> *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
        expScalar = c10::Scalar(*reinterpret_cast<c10::complex<float> *>(aclInput->data_ptr()));
    } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
        expScalar = c10::Scalar(*reinterpret_cast<c10::BFloat16 *>(aclInput->data_ptr()));
    } else {
        ASCEND_LOGE("unsupported scalar type! ");
    }
    return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(
        c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(scalar_data_type)) + " has not been supported");
    c10::SmallVector<int64_t, 5> storageDims;
    auto itemsize = at_tensor.itemsize();
    if (itemsize == 0) {
        AT_ERROR("When ConvertType, tensor item size cannot be zero.");
        return nullptr;
    }
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    }
    aclFormat format = ACL_FORMAT_ND;
    if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
        at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
            aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(), storageDims.size(),
            const_cast<void *>(aclInput.storage().data()));
    }
    return aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
        at_tensor.strides().data(), at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
        const_cast<void *>(at_tensor.storage().data()));
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(scalar_data_type)) + " has not been supported");
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: { double v = at_scalar.toDouble();   acl_scalar = aclCreateScalar(&v, acl_data_type); break; }
        case at::ScalarType::Long:   { int64_t v = at_scalar.toLong();    acl_scalar = aclCreateScalar(&v, acl_data_type); break; }
        case at::ScalarType::Bool:   { bool v = at_scalar.toBool();       acl_scalar = aclCreateScalar(&v, acl_data_type); break; }
        case at::ScalarType::ComplexDouble: { auto v = at_scalar.toComplexDouble(); acl_scalar = aclCreateScalar(&v, acl_data_type); break; }
        default: acl_scalar = nullptr;
    }
    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) return nullptr;
    return aclCreateIntArray(at_array.data(), at_array.size());
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) return nullptr;
    return aclCreateBoolArray(value.data(), value.size());
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) return nullptr;
    return aclCreateBoolArray(value.data(), value.size());
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) return nullptr;
    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    return aclCreateTensorList(tensor_list.data(), tensor_list.size());
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) return ConvertType(opt_tensor.value());
    return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) return ConvertType(opt_array.value());
    return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) return ConvertType(opt_scalar.value());
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    return reinterpret_cast<OpApiFunc>(opApiAddr);
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &...args) { return std::make_tuple(ConvertType(args)...); }

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) { return f(std::get<I>(t)...); }

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor != nullptr) aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar != nullptr) aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray != nullptr) aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray != nullptr) aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList != nullptr) aclDestroyTensorList(p);
}

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename T> struct is_at_tensor : std::false_type {};
template <> struct is_at_tensor<at::Tensor> : std::true_type {};
template <typename T> struct is_at_tensor_list : std::false_type {};
template <> struct is_at_tensor_list<at::TensorList> : std::true_type {};

template <std::size_t I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res) {}

template <std::size_t I = 0, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type
GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

template <typename... Ts>
auto DecodeDevice(Ts &...args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

#define ACLNN_CMD(aclnn_api, ...)                                                                           \
    do {                                                                                                    \
        auto device = DecodeDevice(__VA_ARGS__);                                                            \
        const c10::OptionalDeviceGuard device_guard(device);                                                \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");       \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                     \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                         \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                     \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                              \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                        \
            #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize",                                              \
            " not found in libcust_opapi.so. ",                                                             \
            "Ensure the run package is installed correctly.");                                  \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                     \
        uint64_t workspace_size = 0;                                                                        \
        uint64_t *workspace_size_addr = &workspace_size;                                                    \
        aclOpExecutor *executor = nullptr;                                                                  \
        aclOpExecutor **executor_addr = &executor;                                                          \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);         \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr); \
        if (initMemFunc) { initMemFunc(nullptr, false); }                                                   \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);              \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);  \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                               \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());    \
        void *workspace_addr = nullptr;                                                                     \
        if (workspace_size != 0) {                                                                          \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
            auto workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));                  \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                         \
        }                                                                                                   \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int { \
            using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);                \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                               \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                 \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());         \
            ReleaseConvertTypes(converted_params);                                                          \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);               \
            if (releaseMemFunc) { releaseMemFunc(nullptr, false); }                                         \
            return api_ret;                                                                                 \
        };                                                                                                  \
        at_npu::native::OpCommand cmd;                                                                      \
        cmd.Name(#aclnn_api);                                                                               \
        cmd.SetCustomHandler(acl_call);                                                                     \
        cmd.Run();                                                                                          \
        if (unInitMemFunc) { unInitMemFunc(nullptr, false); }                                               \
    } while (false)

#endif  // CANN_BENCH_ACLNN_COMMON_H_
