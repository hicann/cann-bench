# 防作弊：禁用被评测算子的内置 kernel

为保证评测公平，提交不应通过调用 CANN 内置同名算子（`aclnn<Op>` / `ADD_TO_LAUNCHER_LIST_AICORE(<Op>)`）
来"蹭"官方实现，而应自带 kernel。本目录提供**手动**工具，把评测机上被评测算子的内置
AiCore kernel 二进制用 `mv` 移到备份目录（从 OPP 移除，不用 `rm`），使这条作弊路径在运行时直接失败（找不到 kernel）。

## 设计要点

- **只删二进制，不动实现源码 / 注册**：仅移除 `kernel/<soc>/<cat>/<op>/`（`*.o` + `*.json`），
  保留 TBE/AscendC impl 源码、op_proto、ops-info，torch_npu 不受影响。
- **设备侧 intrinsic 不受影响**：提交 kernel 内的 `AscendC::Add/Mul/Exp/Mmad/...` 在编译期展开进
  自己的 kernel，与内置算子二进制无关，删除后照常工作。
- **保护项**：`MatMul*` / `ReduceMax*`（`perf_eval` 每次测量用于升频 + 清 L2 cache）及
  `Add/Mul/Cast/...` 等通用基础算子**不删**。
- **评测仍正常**：golden 走 fp64-CPU，baseline 为预先实测值，故删除内置算子不影响精度/性能评测。

## 文件

| 文件 | 说明 |
|------|------|
| `benchmarked_kernels.txt` | 待禁用 kernel 目录清单（相对 `kernel/<soc>/`），由 `tasks/**/cases.csv` 的 `baseline_kernels` 推导 |
| `kernel_map.json` | 每个被评测算子 → kernel 目录的映射（审阅用） |
| `disable_builtin_kernels.sh` | 用 `mv` 把清单中的 kernel 二进制移到备份目录（= 备份 + 从 OPP 移除，不用 `rm`，**可逆**） |
| `restore_builtin_kernels.sh` | 从备份一键还原 |

## 用法

```bash
# 预览（不删除、不提示）
bash scripts/anti_cheat/disable_builtin_kernels.sh --dry-run

# 执行（交互式需输入 DELETE 确认；脚本/容器中需显式 --yes）
bash scripts/anti_cheat/disable_builtin_kernels.sh            # 交互确认
bash scripts/anti_cheat/disable_builtin_kernels.sh --yes      # 跳过确认

# 还原
bash scripts/anti_cheat/restore_builtin_kernels.sh
```

可选参数：

- `--soc=<dir>`：内置 kernel 树下的 SOC 子目录名，取值如 `ascend910b`（Ascend 910B 系列）、
  `ascend910_93`（Ascend 910_93xx，即 910C）、`ascend910_95`（950）、`ascend310p` 等。
  **省略时由 `acl.get_soc_name()` 自动检测**（避免在 910C 机器上沿用默认 `ascend910b`
  而静默无操作）；若无法从映射中识别芯片名，脚本会**失败而非回退**到旧默认值。
- `--list=<file>`：自定义 kernel 清单。
- `--backup-dir=<dir>`：默认 `$HOME/.cann_bench_kernel_backup`。

## ⚠️ 风险与安全保护

- 修改的是**全局共享**的 CANN 安装，影响本机所有进程/用户/项目，且持续存在。
- **强烈建议在一次性 docker 容器 / 专用评测机中执行**，不要在共享开发机直接运行。
- 安全保护：脚本**绝不会被任何评测流程自动调用**（`run_evaluation.sh`、`kernel_eval`
  均不引用本目录）；非交互环境下若未显式 `--yes` 会**直接中止**，避免误删。
- 完全可逆且不删数据：禁用用 `mv` 把目录移到备份目录（不用 `rm`），`restore` 脚本用 `mv` 一键移回。

## 残留风险：kernel-disable 无法覆盖的算子

kernel-disable 只能移除**预编译的 AiCore kernel 二进制目录**。对于以下算子，
即使在 `cases.csv` 补 `baseline_kernels` 标注也无处可指（OPP 中不存在可移除的 AiCore
kernel），因此**本机制结构性地无法防护**，需依赖 host 侧 `TorchOpGuard`
（`src/kernel_eval/security/torch_op_guard.py`）或其他手段。已知残留：

| Task | 可达的内置路径 | 为何 kernel-disable 无效 |
|------|----------------|--------------------------|
| `tasks/level3/unique` | `at::unique` / `aclnnUnique` | Unique 走 **AICPU**，OPP 无 AiCore kernel 目录可移除 |
| `tasks/level4/gru` | `torch_npu.npu_gru` / `at::gru` | NPU 上 GRU **分解为 matmul + 逐元素**，无单一 GRU kernel；matmul kernel 属 PROTECTED 不可禁 |
| `tasks/level3/strided_slice` | 原生 `x[b:e:s]` 索引 + DMA | 原生切片**不经过任何算子 kernel**（纯 view + DMA 物化），禁 `strided_slice` 目录只挡显式 op 路径 |

> 说明：`TorchOpGuard` 是 Python `TorchFunctionMode`，无法拦截 C++ 自定义算子
> （PrivateUse1 plugin）内部直接发起的 `at::xxx` 调用。**该缺口现已由 dispatch 层守卫
> `DeviceResidencyGuard`（`TorchDispatchMode`，`src/kernel_eval/security/device_residency_guard.py`）
> 补上**：`topk` / `sort` / `matmul` / `addmm` 等已纳入 `ATEN_COMPUTE_LEAVES` 并默认 block
> （此前实证的 C++ `at::topk` 作弊、以及 GRU 经 matmul/addmm 分解的绕过均已覆盖，见
> `tests/ut/test_dispatch_compute_guard.py`）。上表中 `unique`（走 AICPU）与 `strided_slice`
> （纯 view + DMA、不经任何 compute kernel）仍为真实残留——dispatch 层亦无对应 compute leaf 可拦，
> 需依赖人工审查。

> `tasks/level3/engram_gate_fusion` **不在**残留清单：它是全新融合算子（DeepSeek Engram
> 7 步），OPP 无对应内置实现，无法通过单一内置算子绕过，故其缺少 `baseline_kernels` 标注是**正确**的。
