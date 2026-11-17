# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path
import shutil
import subprocess

import pytest


def test_workspace_survives_queued_op_command(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler is required for workspace lifetime regression')
    header = (Path(__file__).parents[2] / 'src/ascend_arena_e2e/ascend_arena/runners/validation/pytorch_npu_helper.hpp').read_text()
    macro = header[header.index('#define STRINGIFY(x)'):header.rindex('#endif')]
    program = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
static std::weak_ptr<int> workspace;
static std::function<int32_t()> queued;
namespace at {
    constexpr int kByte = 0;
    struct TensorOptions {
        explicit TensorOptions(int) {}
        TensorOptions& dtype(int) { return *this; }
    };
    struct Storage { std::shared_ptr<int> value; const void* data() { return value.get(); } };
    struct Tensor {
        std::shared_ptr<int> value;
        Storage storage() { return {value}; }
        void reset() { value.reset(); }
    };
}
namespace at {
    Tensor empty(std::initializer_list<int64_t>, const TensorOptions&) {
        auto value = std::make_shared<int>(42);
        workspace = value;
        return {value};
    }
}
namespace torch_npu::utils { int get_npu_device_type() { return 0; } }
using aclrtStream = void*;
struct aclOpExecutor {};
namespace c10_npu {
    struct Stream { aclrtStream stream(bool) { return nullptr; } };
    Stream getCurrentNPUStream() { return {}; }
}
namespace at_npu::native {
    struct OpCommand {
        void Name(const char*) {}
        void SetCustomHandler(std::function<int32_t()> callback) { queued = callback; }
        void Run() {}
    };
}
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check failed"); } while (false)
using InitHugeMemThreadLocal = int32_t (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);
using WorkspaceQuery = int (*)(int, int64_t*, aclOpExecutor**);
int query(int, int64_t* size, aclOpExecutor** executor) {
    *size = 64;
    *executor = nullptr;
    return 0;
}
int32_t execute(void* data, uint64_t size, aclOpExecutor*, const aclrtStream) {
    assert(!workspace.expired());
    assert(data == workspace.lock().get() && size == 64);
    return 0;
}
void* GetOpApiFuncAddr(const char* name) {
    if (std::string(name) == "FakeApiGetWorkspaceSize") return reinterpret_cast<void*>(query);
    if (std::string(name) == "FakeApi") return reinterpret_cast<void*>(execute);
    return nullptr;
}
const char* GetOpApiLibName() { return "fake"; }
const char* aclGetRecentErrMsg() { return "fake"; }
auto ConvertTypes(int value, int64_t* size, aclOpExecutor** executor) { return std::make_tuple(value, size, executor); }
template<typename T> auto ConvertToOpApiFunc(T&, void* address) { return reinterpret_cast<WorkspaceQuery>(address); }
template<typename F, typename T> auto call(F function, T& args) { return std::apply(function, args); }
template<typename T> void ReleaseConvertTypes(T&) {}
''' + macro + r'''
int main() {
    EXEC_NPU_CMD(FakeApi, 1);
    assert(!workspace.expired());
    assert(queued() == 0);
    assert(workspace.expired());
    queued = nullptr;
    assert(workspace.expired());
}
'''
    source = tmp_path / 'workspace_lifetime.cpp'
    source.write_text(program)
    binary = tmp_path / 'workspace_lifetime'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)], check=True, timeout=30, capture_output=True)
    subprocess.run([str(binary)], check=True, timeout=5, cwd=tmp_path, capture_output=True)
