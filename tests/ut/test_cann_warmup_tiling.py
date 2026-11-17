# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_warmup_buffer_alignment_fits_reported_ub(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler is required for native tiling regression')
    root = Path(__file__).parents[2] / 'src/cann_bench_utils/csrc/ops/warmup'
    source = (root / 'op_kernel/warmup_kernel.cpp').read_text()
    plugin = (root / 'op_plugin/warmup_plugin.cpp').read_text()
    constants = '\n'.join(line for line in source.splitlines() if line.startswith('constexpr static int64_t'))
    function = source[source.index('std::tuple<int64_t, int64_t, int64_t> calc_warmup_tiling_params()'):source.index('// Launch wrapper')]
    launch_tiling = plugin[plugin.index('    int64_t numBlocks, blockLength, tileSize;'):plugin.index('    auto x_ptr')]
    program = '''
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <cassert>
static uint64_t reported_ub;
namespace platform_ascendc {
    enum class CoreMemType { UB };
    struct Platform {
        void GetCoreMemSize(CoreMemType, uint64_t& size) { size = reported_ub; }
        int64_t GetCoreNumAiv() { return 40; }
    };
    struct PlatformAscendCManager {
        static Platform* GetInstance() { static Platform p; return &p; }
    };
}
''' + constants + '\n' + function + '''
int main() {
    for (auto bytes : {196352u, 196608u, 65536u}) {
        reported_ub = bytes;
''' + launch_tiling + '''
        auto tile = tileSize;
        assert(tile > 0 && tile % 32 == 0);
        assert(((tile + 31) / 32 * 32) * PIPELINE_DEPTH * 3 <= reported_ub);
    }
}
'''
    path = tmp_path / 'warmup_tiling.cpp'
    path.write_text(program)
    binary = tmp_path / 'warmup_tiling'
    subprocess.run([compiler, '-std=c++17', str(path), '-o', str(binary)], check=True, timeout=30, capture_output=True)
    subprocess.run([str(binary)], check=True, timeout=5, cwd=tmp_path, capture_output=True)
