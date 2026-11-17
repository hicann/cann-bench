# 算子基线来源与许可证

7 个评测基线使用固定版本的官方 CANN 算子源码。实现文件由 Git submodule 获取，
初始化脚本只检出需要的文件和完整许可证，并在独立目录组装可离线使用的基线。

| 组件 | 固定提交 | 许可证 |
| --- | --- | --- |
| [ops-nn](https://gitcode.com/cann/ops-nn) | `8e051c393f3a7bbcb67ac40e1c42bb0c55abe1ef` | [CANN 2.0](../../src/ascend_arena_e2e/dependency/ops-nn/LICENSE) |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | `0d380e3bb0959d306a3c33d8090adfc32f5279c2` | [CANN 2.0](../../src/ascend_arena_e2e/dependency/ops-transformer/LICENSE) |

逐文件来源、原始 SHA256、组装结果 SHA256 和基线参考版本记录在
[baselines.json](../../src/ascend_arena_e2e/dependency/baselines.json)。其中的 `paths`
同时作为稀疏检出清单，覆盖所需算子的实现、配置和公共依赖文件。

[兼容补丁](../../src/ascend_arena_e2e/dependency/baselines.patch) 保留评测基线的接口、
参数检查和计算逻辑。Chamfer 梯度在两个方向包含恰好一个 tile 的完整块；
清单的 `fixes` 字段记录该边界修正。
[构建适配文件](../../src/ascend_arena_e2e/dependency/baseline_overrides/) 包含 7 个 CMake
入口及两个接口头文件。任务接口、Python 参考实现、binding 和用例保存在主仓。

官方源码、补丁涉及的代码和接口头文件保留原有版权及 CANN 2.0 声明。
完整协议见对应子模块的 `LICENSE` 及主仓 [LICENSE](../../LICENSE)；
使用、修改和分发须遵循协议条款。

初始化和维护命令见 [编译依赖说明](../../src/ascend_arena_e2e/dependency/README.md)。
