# 编译依赖

`ops-skeleton/` 是固定提交的官方 [multimodal-benchmark](https://gitcode.com/cann/multimodal-benchmark)
子模块，构建入口位于 `ops_optimization_skeleton/`，包含 CMake、构建脚本、公共头文件及许可证。
`ops-base/` 是固定提交的官方 [opbase](https://gitcode.com/cann/opbase) 子模块。
`ops-nn/` 与 `ops-transformer/` 是固定提交的官方算子源码子模块。
`baselines.json` 登记必需文件及 SHA256，`baselines.patch` 保存兼容适配，
`baseline_overrides/` 保存 7 个 CMake 入口及两个接口头文件。

在主仓根目录执行：

```bash
python3 scripts/setup_ascend_arena_dependencies.py
git submodule status
```

脚本读取主仓 Git 索引中的固定提交，以部分克隆和稀疏检出获取骨架和所需算子文件。
算子源码直接获取清单中的固定提交，只检出必需的实现、公共依赖文件和完整 LICENSE。
提交、目录树及 Git 必需的忽略和属性元数据仍会下载。GitCode 服务端须支持 `filter`；若服务器提示过滤不受支持，
稀疏工作区仍生效，但不能保证节省对象下载量。已有完整克隆中的对象不会被删除。
脚本只初始化这四个依赖，不递归初始化其他子模块，并拒绝覆盖本地修改。

骨架精确提交与子目录记录在 `ops-skeleton.json`；其中 SHA256 仅供旧版离线压缩包校验。
opbase 固定为 `0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc`，与骨架的版本检查一致。
ops-nn 固定为 `8e051c393f3a7bbcb67ac40e1c42bb0c55abe1ef`，
ops-transformer 固定为 `0d380e3bb0959d306a3c33d8090adfc32f5279c2`。

初始化脚本校验原始文件哈希，在临时目录中复制源码、统一文本换行并应用兼容补丁，
再校验全部输出哈希，成功后发布到 `baselines/`。基线入口仍是
`bench_lab/ascend_arena_bench/<算子>/<算子>/`，通过相对链接访问该生成目录。
基线保留原有接口、参数检查和计算逻辑。Chamfer 梯度的两个方向使用 `>=` 处理恰好一个 tile 的工作量；该修正记录在清单的 `fixes` 字段。
工作区复制会生成独立文件，编辑候选实现不会修改基线或官方检出。

评测器只复制骨架源码到 `kernel_eval/.sandboxes/<运行ID>/dependency/ops-skeleton/`，
不复制多模态仓其他目录。opbase 通过指向原检出目录的链接共享，保留固定版本检查所需的
Git 元数据。主仓可只读挂载，运行时无需网络，结果保存在 `kernel_eval/`。
`--dependency-root` 也接受原有的平铺骨架目录；外部 opbase 可通过 `OPBASE_SOURCE_PATH` 指定。
离线搬运需保留主仓 `.git`、四个子模块和生成的 `baselines/`，避免破坏相对链接及 Git 元数据路径。

## 更新版本

从官方仓库选择已验证提交，将骨架 submodule 指针和 `ops-skeleton.json` 同步更新，
再执行初始化脚本。opbase 版本应与新骨架要求一致，并同步更新其 submodule 指针。
更新算子源码时同步修改 submodule 指针、`baselines.json` 和兼容补丁；先将旧的
`baselines/` 移到其他位置，再初始化新版本。脚本会拒绝覆盖与清单不同的既有基线文件。
提交前运行 `tests/ut/test_arena_dependency_setup.py`、`tests/ut/test_arena_runtime_reuse.py`、
`tests/ut/test_ascend_arena_auto_pipeline.py`，以及真实 NPU 编译与正确性验证。

## 验证范围

CANN 9.1.0、Ascend 910B4，逻辑卡 4–7：当前 7 个算子的原始基线均通过编译、
smoke、complete、profile correctness 和性能采集。完整批次包含 37 个 smoke 用例、
242 个 complete 用例与 70 个 profile 用例；每个性能用例预热 10 次、采样 30 次。
主时延为逐轮全部目标 kernel 总时间的中位数，完整调用延迟单独保留。
实测结果、Agent 优化轮次与日志见[测试报告样例](../../../docs/examples/ascend_arena_user_evaluation_report.md)。
完整单元测试通过 1611 项、跳过 6 项。

构建时将 [骨架补丁](ops-skeleton.patch) 应用于运行目录的副本，使用 Bash 数组传递 CMake 参数。
依赖仓路径含空格时，将 opbase 复制到独立构建目录，供 CANN 生成的编译命令使用。
