#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

import os, sys, subprocess, shutil, logging
from setuptools import setup, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PACKAGE_NAME = "cann_bench_ops"
VERSION = "1.0.0"

class CMakeBuild(Command):
    description = "Build CMake extensions"
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        import torch, torch_npu
        cwd = os.getcwd()
        build_dir = os.path.join(cwd, 'build')
        torch_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")
        npu_path = os.path.dirname(torch_npu.__file__)
        log.info(f"Torch: {torch_dir}, NPU: {npu_path}")
        subprocess.check_call(['cmake', '-S', cwd, '-B', build_dir,
            '-DCMAKE_BUILD_TYPE=Release',
            f'-DTorch_DIR={torch_dir}',
            f'-DTORCH_NPU_PATH={npu_path}',
            f'-DPython3_EXECUTABLE={sys.executable}'])
        subprocess.check_call(['cmake', '--build', build_dir, '--parallel', str(os.cpu_count() or 2)])
        # Copy .so to cann_bench
        so_file = os.path.join(build_dir, '_C.so')
        target = os.path.join(cwd, 'cann_bench', '_C.so')
        if os.path.exists(so_file):
            shutil.copy(so_file, target)
            log.info(f"Copied {so_file} to {target}")
        else:
            log.error(f"_C.so not found in {build_dir}")

class Clean(Command):
    description = "Clean build artifacts"
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        for d in ['build', f'{PACKAGE_NAME}.egg-info']:
            if os.path.exists(d): shutil.rmtree(d)
        # Clean wheel in dist
        if os.path.exists('dist'):
            for f in os.listdir('dist'):
                if f.endswith('.whl'):
                    os.remove(os.path.join('dist', f))
        so_file = os.path.join(os.getcwd(), 'cann_bench', '_C.so')
        if os.path.exists(so_file): os.remove(so_file)

class BinaryDistribution(Distribution):
    def is_pure(self): return False
    def has_ext_modules(self): return True

class ABI3Wheel(bdist_wheel):
    def get_tag(self):
        py, abi, plat = super().get_tag()
        return "cp38", "abi3", "linux_aarch64"
    def run(self):
        self.run_command('cmake_build')
        super().run()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=["cann_bench"],
    package_data={"cann_bench": ["*.so"]},
    distclass=BinaryDistribution,
    cmdclass={'cmake_build': CMakeBuild, 'clean': Clean, 'bdist_wheel': ABI3Wheel},
    install_requires=["torch", "torch_npu"],
)