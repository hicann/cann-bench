/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file extension.cpp
 * \brief Python extension entry point for cann_bench_utils
 *
 * Exposes warmup_npu, cache_clean_npu and their meta variants as pybind11
 * functions. Op schemas are defined via torch.library.define in Python,
 * with impls registered through torch.library.impl calling into _C.so.
 */

#include <torch/python.h>
#include <torch/all.h>

namespace cann_bench_utils {
void device_zero_npu(const torch::Tensor &dst);
void device_memcpy_npu(const torch::Tensor &src, const torch::Tensor &dst);
void device_memcpy_meta(const torch::Tensor &src, const torch::Tensor &dst);
torch::Tensor warmup_npu(const torch::Tensor &x, const torch::Tensor &y, const torch::Tensor &z);
torch::Tensor cache_clean_npu(const torch::Tensor &x, const torch::Tensor &out);
torch::Tensor warmup_meta(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor cache_clean_meta(const torch::Tensor &x);
}

PYBIND11_MODULE(_C, m) {
    m.def("device_zero_npu", &cann_bench_utils::device_zero_npu, "Zero NPU tensor bytes asynchronously");
    m.def("warmup_npu", &cann_bench_utils::warmup_npu, "CannBenchWarmup NPU impl");
    m.def("cache_clean_npu", &cann_bench_utils::cache_clean_npu, "CannBenchCacheClean NPU impl");
    m.def("warmup_meta", &cann_bench_utils::warmup_meta, "CannBenchWarmup Meta impl");
    m.def("cache_clean_meta", &cann_bench_utils::cache_clean_meta, "CannBenchCacheClean Meta impl");
    m.def("device_memcpy_npu", &cann_bench_utils::device_memcpy_npu, "CannBenchDeviceMemcpy NPU impl");
    m.def("device_memcpy_meta", &cann_bench_utils::device_memcpy_meta, "CannBenchDeviceMemcpy Meta impl");
}
