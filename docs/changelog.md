# 版本变更记录

## V0.2.0 (2026-05-07)

**评分体系切换为 SOL-anchored 公式 (对齐 bench.tex)**

- 评分公式改版：单用例性能得分由原始 `SpeedUp = baseline / candidate` 改为 `score_i = (T_baseline − T_HW) / ((T_cand − T_HW) + (T_baseline − T_HW))`（bench.tex Eq. 3）
- 单算子综合评分改版：`EachOperatorScore = [w_c·δ_pass + Σ δ_acc,i (w_f + w_p·score_i) / N] · 100`，归一化到 [0, 100]（bench.tex Eq. 4）
- 权重调整：`(w_c, w_f, w_p) = (0.2, 0.3, 0.5)`（原 `(2, 3, 5)`）
- 新增字段 `t_hw_us`：每个用例新增硬件下界 `T_HW`，写入 cases.yaml 与 cases.csv，加载链路 (case_loader / EvalCaseResult / report_generator) 同步打通
- 工具更新：`inner/generate_cases_csv.py`、`scripts/utils/yaml_to_csv.py`、`inner/validate_cases.py` 均加入新字段；`src/kernel_eval/report/scoring.py` 与 `summary_generator.py` 重写为新公式
- 几何平均加速比保留为诊断字段

## V0.1.1 (2026-04-29)

**文档重组与内容完善**

- 文档目录重组：建立 spec/、design/、guide/ 分层结构
- 文档职责分离：benchmark_spec.md 定义规范，evaluator_design.md 定义实现
- 精度标准完善：新增小值域通过标准（ErrorCount 计算公式）
- 性能评测完善：更新 Trace 解析逻辑（`cat="dequeue"` 事件）、Warmup Kernel 过滤机制、InputPool 防缓存攻击
- 设备同步优化：目标设备同步而非默认设备
- 安全防护：Timing API 防护、返回值类型检查、二次验证机制
- Golden 计算：CPU fp64 Golden 计算流程
- 多硬件支持：多硬件 baseline 解析
- 报告生成：几何平均加速比计算、JSON/Markdown/Summary 多格式输出

---

## V0.1.0 (2026-04-25)

**初版发布**

- 建立基础评测框架
- 定义 L1-L4 四级难度体系
- 完成 55 个算子规格定义和用例设计
- 建立编译正确性、功能正确性、性能优化性三大评测维度
- 定义 MERE/MARE 精度标准和阈值表
- 基础评测架构：编译、功能、性能三维度评测
- JSON + Markdown 报告生成
- Profiler kernel-only 测量
- 目录结构：src/kernel_eval 评测工程