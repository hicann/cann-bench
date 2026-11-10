# 算子提交行为与禁止规则

本文档仅适用于提交至 CANN Bench 参与评测的算子，涵盖手工编写和自动生成的实现，统一说明检测范围、提交约束、禁止行为及具体示例。代码片段只展示行为类型，不对应任何具体提交。

**自动检查发现违规时，相应算子将被判为作弊。通过自动检查不代表最终合规；人工审核确认存在违规行为的算子，同样判为作弊。**

**检测范围：从框架调用用户 Torch 算子入口，到该入口返回结果的用户实现全过程。** 包括 Python/C++ 包装层、输入预处理、输出及 workspace 分配、内存拷贝、全部子 kernel 调度与输出后处理，以及这些调用触发的设备计算。**框架自身生成输入、准备设备输入、搬回结果和比较结果等操作，不计入用户实现的行为检查与拷贝预算。**

解包后的源码目录、构建产物和接口格式见[评测输入与接口规范](../spec/submission_spec.md)。本页只定义代码执行行为边界，规则编号与格式规范保持独立；通过 `cannbench.com` 网站提交时还需满足网站自身的 ZIP 上传约束。

## 1. 算子编写原则

CANN Bench 评测的是提交者实现 Ascend C / NPU kernel 的能力。候选算子应满足以下原则：

- **真实实现目标算子语义**：计算主体应由提交工程中的 Ascend C kernel、自定义 ACLNN 算子或 direct launch kernel 完成。
- **Python/C++ 包装层只做调度和参数整理**：包装层可以读取 shape、dtype、attrs，做 tiling 参数准备、输出张量分配、调用自定义 kernel；不应把核心计算交给 PyTorch、torch_npu 或 CANN 内置同名算子。
- **输入/输出张量处理也属于算子语义**：如果某个输入布局变换、输出重排、dtype 转换或中间 tensor 处理会改变 tensor 数值、布局或物理存储，它应由提交 kernel 完成；不能先用 PyTorch / torch_npu 处理好再交给 kernel。
- **实现应泛化到所有合法输入**：不能只针对公开 case、固定 shape、固定 data pointer 或固定输入值返回预先准备的结果。
- **输出必须是真实计算得到的 Tensor**：返回普通、连续的 `torch.Tensor`，满足题目的输出数量、shape、dtype、精度及可选输出约定，不能用 FakeTensor、懒求值包装器或伪造对象绕过精度检查。
- **禁止 CPU 代算，限制 Host 与 Device 之间的拷贝**：禁止将输入、权重或中间数据传回 Host 计算，再将结果传回 Device。每次完整算子调用，在上述检测范围内，Host 与 Device 间的内存拷贝只允许至多一次，且只能用来传输必要的元数据，不得用于 CPU 代算等违规行为。
- **性能数据必须来自提交算子实际执行的 NPU kernel**：功能正确但没有实际执行提交 kernel 的实现，不符合评测目标。

允许使用提交 kernel 内的 Ascend C 原生 API 和 intrinsic，例如 `AscendC::Add`、`AscendC::Mul`、`AscendC::Exp`、`AscendC::Mmad` 等。这些 API 在编译期成为提交 kernel 的一部分，不等同于在包装层直接调用现成的 CANN / torch_npu 算子。

## 2. 禁止行为分类

### 2.1 [SUB-BEH-001] 调用 PyTorch / torch_npu 内置计算 API 代算

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

### 2.2 [SUB-BEH-002] 使用 PyTorch / torch_npu 处理输入输出 tensor

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

上面第二个反例如果连 `.contiguous()` 都省掉、直接 `return y.permute(0, 2, 1)`, 则由框架**自动拦截**: 精度比对前会校验候选输出 `is_contiguous()`, 非连续输出直接判失败(记为结构性失败, 不是精度不达标)。极端情形是完全不算、直接返回输入的某个视图 -- shape 与数值都能对上, 但 stride 会暴露它。

```python
def fake_by_view(x):
    # 反例: 转置视图数值可能与 golden 完全一致, 但没有任何计算发生;
    # is_contiguous() 校验会直接判失败。
    return x.t()
```

补 `.contiguous()` 把数据真复制一遍不在这条自动校验的范围内, 仍属人工审查判定的无效实现。

### 2.3 [SUB-BEH-003] 路由到 CANN 内置同名算子

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

### 2.4 [SUB-BEH-004] CPU fallback 或没有执行提交 NPU kernel

候选算子不能把输入、权重或中间数据搬回 CPU 计算，再把结果搬回 NPU。即使结果数值正确，只要核心计算不是由提交的 NPU kernel 完成，也属于无效实现。每次调用都必须由提交的 NPU kernel 完成目标计算。禁止仅启动空 kernel 或与结果无关的计算，伪装成已在 NPU 上完成算子计算。

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

### 2.5 [SUB-BEH-005] 缓存输出、固定输出或按输入地址命中

每次调用都必须使用当前输入完成真实计算。实现不能缓存某次执行结果，然后在后续 case 或 repeat 中直接返回或复制为输出，也不能只算首轮或预热轮。实现不应依赖固定输入地址、固定 shape、固定 dtype、固定 attrs 或公开 case 特征来返回预设结果。

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

### 2.6 [SUB-BEH-006] 篡改 profiler 或 timing API

提交代码不应 monkey-patch、替换或删除评测环境的计时、同步、性能采集或运行时 API，也不得修改评测配置或结果文件。禁止冒用框架辅助 kernel 标识（`CannBenchWarmup`、`CannBenchCacheClean`）或调用框架内部能力绕过规则。精度阶段和性能阶段须执行相同算子语义。任何让评测结果不再反映真实 kernel 执行时间的环境篡改都属于作弊行为。

反例：

```python
import torch
import torch_npu

torch.npu.synchronize = lambda *args, **kwargs: None
torch_npu.profiler.profile = fake_profile
```

这类行为会让评测结果不可信，应视为无效提交。

### 2.7 [SUB-BEH-007] 返回 FakeTensor、懒求值包装器或伪 Tensor 对象

候选算子必须返回普通、连续的真实 `torch.Tensor`，或题目约定的 Tensor 多输出结构，并满足输出数量、shape、dtype、精度及可选输出约定。用对象包装真实计算、延迟到比较阶段再求值，或返回 Tensor 子类伪装结果，都会破坏评测边界。不能仅靠视图冒充数据变换，相关反例见 SUB-BEH-002。

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
