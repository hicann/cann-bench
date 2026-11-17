#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys
import logging
import json
import hashlib
import runpy
from pathlib import Path
from setuptools import setup, find_packages, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PACKAGE_NAME = "cann_bench_utils"
VERSION = "1.0.0"
DESCRIPTION = "CANN Bench framework utilities - warmup and cache clean operators"


class CleanCommand(Command):
    """usage: python setup.py clean"""
    description = "Clean build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        folders = ['build', f'{PACKAGE_NAME}.egg-info']
        for f in folders:
            if os.path.exists(f):
                shutil.rmtree(f)
                logging.info(f"Removed: {f}")
        if os.path.exists('dist'):
            for f in os.listdir('dist'):
                if f.endswith('.whl'):
                    os.remove(os.path.join('dist', f))
        logging.info("Cleaned.")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class CustomWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        return python, abi, plat

    def run(self):
        self.run_command('cmake_build')
        super().run()


class CMakeBuildCommand(Command):
    description = "Build CMake extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import torch
        import torch_npu

        cpu_count = os.cpu_count() or 2
        Torch_DIR = os.path.join(torch.utils.cmake_prefix_path, "Torch")
        TORCH_NPU_PATH = os.path.dirname(torch_npu.__file__)
        NPU_ARCH = os.environ.get('NPU_ARCH', 'ascend910b')

        logging.info(f"Torch: {Torch_DIR}")
        logging.info(f"Torch NPU: {TORCH_NPU_PATH}")
        logging.info(f"NPU_ARCH: {NPU_ARCH}")

        build_dir = os.path.join(os.getcwd(), 'build')
        subprocess.check_call([
            'cmake', '-S', os.getcwd(), '-B', build_dir,
            '-DCMAKE_BUILD_TYPE=Release',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DTorch_DIR={Torch_DIR}',
            f'-DTORCH_NPU_PATH={TORCH_NPU_PATH}',
            f'-DNPU_ARCH={NPU_ARCH}'
        ])
        subprocess.check_call([
            'cmake', '--build', build_dir,
            '--config', 'Release', '--parallel', str(cpu_count)
        ])
        build_info = runpy.run_path(f"{PACKAGE_NAME}/build_info.py")
        info = build_info["current_environment"]()
        info["source_sha256"] = build_info["source_fingerprint"](Path.cwd())
        info["npu_arch"] = NPU_ARCH
        info["binary_sha256"] = hashlib.sha256(Path(build_dir, "_C.so").read_bytes()).hexdigest()
        Path(PACKAGE_NAME, "_build_info.json").write_text(json.dumps(info) + "\n")
        logging.info("CMake build complete.")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': CustomWheel,
    'cmake_build': CMakeBuildCommand,
}

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={PACKAGE_NAME: ['*.so', '_build_info.json']},
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    # torch/torch_npu are runtime prerequisites provided by the CANN image.
    # Listing them here lets `pip install --force-reinstall` (without --no-deps)
    # upgrade torch to latest PyPI, breaking the ABI-matched build.
    install_requires=[],
    python_requires='>=3.8',
)
