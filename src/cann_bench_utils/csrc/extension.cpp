/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file extension.cpp
 * \brief Python extension entry point for cann_bench_utils
 */

#include <Python.h>

extern "C"
{
    // _C 模块本身不导出任何 Python 方法；它的唯一作用是作为一个可被 import 的
    // 扩展入口。Python 首次 import _C 时会 dlopen 本 .so，从而触发链接进来的
    // 各 op_plugin（warmup / cache_clean）中由 TORCH_LIBRARY_FRAGMENT /
    // TORCH_LIBRARY_IMPL 注册的静态初始化器，把自定义算子挂到
    // torch.ops.cann_bench_utils 命名空间下。
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            nullptr,
            -1,
            nullptr,
        };
        return PyModule_Create(&module_def);
    }
}
