# 算子提交原则与禁止行为

本文档说明 CANN Bench 对提交算子的基本要求，以及哪些实现方式会被视为无效或作弊。这里的代码片段只展示行为类型，不对应任何具体提交。

## 1. 算子编写原则

CANN Bench 评测的是提交者实现 Ascend C / NPU kernel 的能力。候选算子应满足以下原则：

- **真实实现目标算子语义**：计算主体应由提交工程中的 Ascend C kernel、自定义 ACLNN 算子或 direct launch kernel 完成。
- **Python/C++ 包装层只做调度和参数整理**：包装层可以读取 shape、dtype、attrs，做 tiling 参数准备、输出张量分配、调用自定义 kernel；不应把核心计算交给 PyTorch、torch_npu 或 CANN 内置同名算子。
- **输入/输出张量处理也属于算子语义**：如果某个输入布局变换、输出重排、dtype 转换或中间 tensor 处理会改变 tensor 数值、布局或物理存储，它应由提交 kernel 完成；不能先用 PyTorch / torch_npu 处理好再交给 kernel。
- **实现应泛化到所有合法输入**：不能只针对公开 case、固定 shape、固定 data pointer 或固定输入值返回预先准备的结果。
- **输出必须是真实计算得到的 Tensor**：返回值需要是普通 `torch.Tensor`，不能用 FakeTensor、懒求值包装器或伪造对象绕过精度检查。
- **性能数据必须来自提交算子实际执行的 NPU kernel**：功能正确但没有实际执行提交 kernel 的实现，不符合评测目标。

允许使用提交 kernel 内的 Ascend C 原生 API 和 intrinsic，例如 `AscendC::Add`、`AscendC::Mul`、`AscendC::Exp`、`AscendC::Mmad` 等。这些 API 在编译期成为提交 kernel 的一部分，不等同于在包装层直接调用现成的 CANN / torch_npu 算子。

## 2. 禁止行为分类

### 2.1 调用 PyTorch / torch_npu 内置计算 API 代算

候选算子的执行路径中，不应直接调用 PyTorch / torch_npu 内置计算 API 来完成目标算子。即使只把部分计算交给现成 API，也属于无效实现。包装层可以做参数整理，但不能把核心计算交给 matmul、conv、softmax、attention、activation、normalization 等现成实现。

反例：

```python
import torch
import torch.nn.functional as F

def softmax(x, dim=-1):
    return F.softmax(x, dim=dim)
```

```python
def grouped_matmul(x, weight):
    return x @ weight
```

同一类绕过写法也不允许，例如改用 `torch.ops.aten.matmul`、`x.matmul(y)`、`torch.mm`、`torch.nn.functional.conv2d` 等。

### 2.2 使用 PyTorch / torch_npu 处理输入输出 tensor

输入预处理、输出后处理和中间 tensor 变换也是目标算子实现的一部分。不能用 PyTorch / torch_npu 的 tensor API 先完成 transpose、permute、contiguous、reshape-copy、cast、slice、gather、scatter 等实质性数据搬运或布局变换，再把处理后的 tensor 交给提交 kernel。

> 注：此类 I/O 搬运 / 布局变换（transpose/permute/gather/scatter 等）属**人工审查**判定为无效的范畴——为避免误伤 Gather / Transpose 等以其为核心语义的算子，框架**不对这些算子自动拦截**；而 matmul / conv / softmax 等计算类算子的绕过，则由 `TorchOpGuard` / `DeviceResidencyGuard` **自动拦截**（默认 block）。

反例：

```python
def transpose_then_kernel(x):
    # 反例：先用现成 NPU tensor API 完成布局变换，再让 kernel 只处理剩余部分。
    x_t = x.transpose(0, 1).contiguous()
    return launch_custom_kernel(x_t)
```

```python
def permute_output(x):
    y = launch_custom_kernel(x)
    # 反例：输出重排由 PyTorch / torch_npu 完成，而不是提交 kernel 完成。
    return y.permute(0, 2, 1).contiguous()
```

如果算法需要类似 transpose / permute / cast / slice 的数据变换，应在提交 kernel 内用 Ascend C API、数据搬运和索引逻辑实现，而不是在包装层调用现成 tensor 算子。

### 2.3 路由到 CANN 内置同名算子

提交工程不应只是把任务转发给评测环境中已有的 CANN 内置同名算子。无论目标环境是否提供同名内置实现，候选提交都应包含自己的 kernel 实现。

反例：

```cpp
// 反例：包装层只调用环境已有的同名内置算子。
CHECK_ACL(aclnnSoftmaxGetWorkspaceSize(x, dim, y, &workspaceSize, &executor));
CHECK_ACL(aclnnSoftmax(workspace, workspaceSize, executor, stream));
```

```cpp
// 反例：注册/转发到现成同名 AiCore 实现，而不是提交自己的 kernel。
ADD_TO_LAUNCHER_LIST_AICORE(Softmax);
```

正例是提交自己的 kernel，并在 kernel 内使用 Ascend C 原语完成计算：

```cpp
// 正例：在提交的 kernel 内使用 Ascend C API 组成目标计算。
AscendC::Exp(dstLocal, srcLocal, elementCount);
```

### 2.4 CPU fallback 或没有执行提交 NPU kernel

候选算子不能把输入搬回 CPU 计算，再把结果搬回 NPU。即使结果数值正确，只要核心计算不是由提交的 NPU kernel 完成，也属于无效实现。

反例：

```python
import torch

def exp(x):
    y_cpu = torch.exp(x.cpu())
    return y_cpu.to(x.device)
```

```python
def add(x, y):
    # 反例：结果来自 host 侧计算或预处理，而不是提交的 NPU kernel。
    return (x.cpu() + y.cpu()).npu()
```

### 2.5 缓存输出、固定输出或按输入地址命中

实现不能缓存某次执行结果，然后在后续 case 或 repeat 中直接返回。实现也不应依赖固定输入地址、固定 shape、固定 dtype、固定 attrs 或公开 case 特征来返回预设结果。

反例：

```python
_cache = {}

def sigmoid(x):
    key = x.data_ptr()
    if key in _cache:
        return _cache[key]
    out = launch_or_compute_once(x)
    _cache[key] = out
    return out
```

```python
_last_output = None

def relu(x):
    global _last_output
    if _last_output is not None:
        return _last_output
    _last_output = launch_or_compute_once(x)
    return _last_output
```

同样不允许根据公开 case 的固定 shape、dtype 或属性直接返回常量结果。

### 2.6 篡改 profiler 或 timing API

提交代码不应 monkey-patch、替换或删除评测环境的计时、同步、性能采集或运行时 API。任何让评测结果不再反映真实 kernel 执行时间的环境篡改都属于作弊行为。

反例：

```python
import torch
import torch_npu

torch.npu.synchronize = lambda *args, **kwargs: None
torch_npu.profiler.profile = fake_profile
```

这类行为会让评测结果不可信，应视为无效提交。

### 2.7 返回 FakeTensor、懒求值包装器或伪 Tensor 对象

候选算子必须返回真实 `torch.Tensor`。用对象包装真实计算、延迟到比较阶段再求值，或返回 Tensor 子类伪装结果，都会破坏评测边界。

反例：

```python
class LazyTensor:
    def __init__(self, x):
        self.x = x

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(real_compute(self.x), *(args or ()), **(kwargs or {}))

def mish(x):
    return LazyTensor(x)
```

## 3. 判定原则

本规范关注的是行为边界，不依赖具体文件名、提交 ID 或评测实现细节。若某种写法的效果是绕过“提交者实现真实 NPU kernel”这一评测目标，即使没有出现在上面的示例中，也可能被判为无效提交。
