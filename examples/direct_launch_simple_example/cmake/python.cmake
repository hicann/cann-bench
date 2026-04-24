# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# find python
# Allow caller to pin Python via -DPython3_EXECUTABLE; otherwise fall back
# to auto-detect. When setup.py invokes cmake we want the same interpreter
# that's running setup.py, so setup.py passes -DPython3_EXECUTABLE=${sys.executable}.
if(NOT Python3_EXECUTABLE AND DEFINED ENV{PYTHON_EXECUTABLE})
    set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
endif()
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3: ${Python3_EXECUTABLE} (found version ${Python3_VERSION})")
message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
