# 评测报告生成系统设计

> **当前实现与模板状态**：正文是报告系统的设计快照。当前 NPU 脚本会输出 correctness/performance/final 分阶段报告，版本、评分与参考失效字段见[快速入门](../guide/quick_start.md)及[评测规范](../spec/benchmark_spec.md)。
>
> 主线 `tasks/description.html` 已迁移，现存文件为 `archives/season1/tasks/description.html`。当前生成器/重建脚本仍尝试旧默认路径，不能把正文中的旧路径视为已修复；重建时应显式传入现存模板，并保留原 JSON。脚本未完整透传新增诊断字段，重建 HTML 不应被当作新的评分依据。

**文档版本：参见 [changelog](../changelog.md)**

## 目录
- [1. Context](#1-context)
- [2. 方案设计](#2-方案设计)
- [3. 报告输出格式](#3-报告输出格式)
- [4. 核心模块设计](#4-核心模块设计)
- [5. 数据流](#5-数据流)
- [6. 实施步骤](#6-实施步骤)

---

## 1. Context

### 1.1 背景

评测框架运行后生成原始评测数据（`EvalOperatorResult`），需要将其转化为结构化、可读的报告，
供人工审核和归档。报告需同时满足机器可解析（JSON）、人类可读（Markdown）和
正式呈现（HTML）三种场景。

### 1.2 报告生成架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           评测数据层                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  EvalOperatorResult (per-operator)                                  │    │
│  │  ├── rel_path, operator                                             │    │
│  │  ├── total_cases, passed_cases, failed_cases                        │    │
│  │  ├── pass_rate, avg_speedup                                         │    │
│  │  └── results: List[EvalCaseResult]                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           报告生成层 (src/kernel_eval/report/)                │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ scoring.py       │  │setup_info.py     │  │ report_generator.py      │  │
│  │ ScoringCalculator│  │ collect_setup()  │  │ ReportGenerator          │  │
│  │ (Eq.3/4/5 评分)  │  │ (metadata + env) │  │ ├── generate()           │  │
│  │                  │  │                  │  │ ├── save_json()          │  │
│  └──────────────────┘  └──────────────────┘  │ ├── save_markdown()      │  │
│                                              │ └── save_html()          │  │
│  ┌──────────────────┐  ┌──────────────────┐  └──────────────────────────┘  │
│  │summary_generator │  │html_generator.py │                                 │
│  │ EvaluationSummary│  │ render_html_report│                                │
│  │ (几何平均加速比)  │  │ (模板拼接渲染)    │                                │
│  └──────────────────┘  └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出产物 (reports/cann/)                           │
│                                                                             │
│  eval_YYYYMMDD_HHMMSS.json   结构化评测数据 + setup_info                     │
│  eval_YYYYMMDD_HHMMSS.md     Markdown 文本报告                              │
│  eval_YYYYMMDD_HHMMSS.html   完整 HTML 报告                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 方案设计

### 2.1 HTML 报告拼接策略

HTML 报告由静态模板片段与动态数据拼接生成，避免整个报告在代码中硬编码 HTML 结构：

```
tasks/description.html        ← 静态模板 (CSS + Header + Abstract + Section 1)
    │
    │  render_html_report()
    │  ├── 1. 读取 description.html
    │  ├── 2. 正则替换摘要中的硬编码值为实际评测数据
    │  │      ├── Agent/Skill、BaseModel（有值则替换，无值则删除）
    │  │      ├── 通过率、得分（从 report.summary 获取）
    │  │      └── 算子数、用例数、等级（从 report.operators 统计）
    │  ├── 3. 渲染 Section 2: Experiment Setup（setup_info）
    │  ├── 4. 渲染 Section 3: Results Analysis（KPI / Level Table / Bar Charts）
    │  ├── 5. 渲染 Section 4: Operator Details（算子详情表）
    │  └── 6. 追加认证印章
    │
    ▼
eval_YYYYMMDD_HHMMSS.html    ← 完整 HTML 报告
```

**设计原则**：
- `description.html` 作为评测集描述的单一事实来源，描述 53 算子/4 等级/1060 用例的基准范围
- 报告生成时通过正则替换将模板中的示例值替换为实际评测数据
- Section 2 环境信息通过 `setup_info.py` 运行时采集，不写死在模板中
- Section 3/4 由 `html_generator.py` 根据 `EvalReport` 数据动态渲染

### 2.2 报告生成入口

两种方式生成报告：

**方式一：评测运行时自动生成（`run_evaluation.sh`）**
```
./scripts/run_evaluation.sh --source-dir examples/aclnn_launch_example
→ reports/cann/eval_xxx.{json,md,html}（三份同时输出）
```

**方式二：从已有 JSON 重新生成（`gen_report.sh`）**
```
./scripts/utils/gen_report.sh reports/cann/eval_xxx.json
→ reports/cann/eval_xxx.html（独立生成，不重新评测）

# 指定自定义摘要/第一章模板
./scripts/utils/gen_report.sh --json eval.json --template custom/template.html
```

### 2.3 摘要数据替换

`description.html` 中的摘要段落包含示例数据，`html_generator.py` 通过正则表达式
将以下字段替换为实际评测数据：

| 模板字段 | 替换来源 | 示例（替换前 → 替换后） |
|----------|----------|--------------------------|
| `Agent/Skill为CANNBot-skill` | `setup_info.metadata.agent_skill` | 有值则替换，无值则删除整段 |
| `BaseModel为DeepSeek V4 Pro` | `setup_info.metadata.base_model` | 同上 |
| `针对 53 个算子…1,060 个评测用例` | `report.total_operators / total_cases` | → `本次对 1 个算子…20 个评测用例` |
| `整体通过率为 48.9%（518/1,060）` | `report.summary.pass_rate / passed_cases` | → `整体通过率为 85.0%（17/20）` |
| `总得分为 131（满分 5,300）` | `report.overall_score` | → `总得分为 79（满分 100）` |

---

## 3. 报告输出格式

### 3.1 JSON 报告 (`eval_xxx.json`)

```json
{
  "version": "1.0",
  "eval_code": "eval_20260601_114121",
  "timestamp": "2026-06-01T11:41:21",
  "device": "npu:0",
  "total_operators": 1,
  "total_cases": 20,
  "passed_cases": 17,
  "failed_cases": 3,
  "overall_score": 78.81,
  "summary": {
    "pass_rate": 0.85,
    "overall_score": 78.81
  },
  "setup_info": {
    "metadata": {
      "framework": "CANN-Bench V0.1.0",
      "date": "2026-06-01 11:41:21",
      "agent_skill": "",
      "base_model": "",
      "benchmark": "CANN-Bench tasks",
      "license": "CANN Open Software License v2.0"
    },
    "environment": {
      "npu": "Ascend950PR_9579 × 1",
      "cpu": "x86_64",
      "cann": "9.0.0-beta.2",
      "driver": "cann-9.0.0-beta.2",
      "pytorch": "2.7.1+cpu",
      "pytorch_npu": "2.7.1.post5",
      "torchvision": "0.22.1+cpu",
      "python": "3.12.9",
      "os": "Linux-6.6.0-132.0.0.111.oe2403sp3.x86_64",
      "docker": "cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone / CANN 9.1.0"
    }
  },
  "operators": [
    {
      "rel_path": "level1/mish",
      "operator": "Mish",
      "total_cases": 20,
      "passed_cases": 17,
      "pass_rate": 0.85,
      "avg_speedup": 2.69,
      "score": 78.81,
      "cases": [...]
    }
  ]
}
```

### 3.2 Markdown 报告 (`eval_xxx.md`)

基于固定模版生成，包含：
- 概览表（算子数、用例数、通过率、综合得分）
- 每个算子的详情表（通过数、加速比、得分）
- 每个用例的执行状态与精度误差

### 3.3 HTML 报告 (`eval_xxx.html`)

完整的独立 HTML 文件，包含：
- **Header Bar**：报告标题与版本
- **Abstract**：摘要（动态数据替换）
- **Section 1**：评测集概述（静态，来自 `description.html`）
- **Section 2**：评测配置（动态，来自 `setup_info`）
- **Section 3**：结果分析（KPI 条 / 等级汇总表 / 柱状图 / Top 算子表）
- **Section 4**：算子详情表（所有评测算子列表）
- **认证印章**：CANN-Bench 认证章

---

## 4. 核心模块设计

### 4.1 `setup_info.py` — 配置信息采集

```
┌─────────────────────────────────────────┐
│  collect_setup_info(config) → Dict      │
├─────────────────────────────────────────┤
│                                         │
│  metadata:                              │
│    framework  ← "CANN-Bench V0.1.0"     │
│    date       ← datetime.now()          │
│    agent_skill← config.agent_skill      │
│    base_model ← config.base_model       │
│    benchmark  ← "CANN-Bench tasks"      │
│    license    ← 固定                    │
│                                         │
│  environment:                           │
│    npu        ← torch_npu.get_device_   │
│                  name(0) × device_count │
│    cpu        ← platform.machine()      │
│    cann       ← ASCEND_TOOLKIT_HOME/    │
│                  compiler/version.info  │
│    driver     ← "cann-{cann_version}"   │
│    pytorch    ← torch.__version__       │
│    pytorch_npu← torch_npu.__version__   │
│    torchvision← torchvision.__version__ │
│    python     ← sys.version             │
│    os         ← platform.platform()     │
│    docker     ← $CANN_BENCH_IMAGE       │
│                  + 实测 CANN 版本       │
└─────────────────────────────────────────┘
```

**CANN 版本读取策略**：
1. 优先从 `ASCEND_TOOLKIT_HOME/compiler/version.info` 读取 `Version=` 行
2. 回退到 `ASCEND_TOOLKIT_HOME/version.info`
3. 回退到 `/usr/local/Ascend/ascend-toolkit/version.cfg`
4. 最后从 `ASCEND_TOOLKIT_HOME` 路径中正则提取 `cann-<version>`

### 4.2 `html_generator.py` — HTML 渲染

核心函数 `render_html_report(report, setup_info, index_path)`：

```
render_html_report()
│
├── 1. 读取 tasks/description.html（CSS + Header + Abstract + Section 1）
│
├── 2. 正则替换摘要动态字段
│   ├── Agent/Skill: re.sub(r'Agent/Skill为[...]', ...)
│   ├── BaseModel:   re.sub(r'BaseModel为[...]', ...)
│   ├── 通过率/得分: re.sub(r'整体通过率为...', ...)
│   └── 算子/用例数: re.sub(r'针对 53 个算子...', ...)
│
├── 3. 渲染 Section 2: _render_section2(setup_info)
│   ├── Metadata 表: framework/date/agent/base_model/benchmark/license
│   └── Environment 表: npu/cpu/cann/driver/pytorch/pytorch_npu/...
│
├── 4. 渲染 Section 3: Results Analysis
│   ├── KPI strip (_render_kpi)
│   ├── Level 汇总表 (_render_level_table)
│   ├── 柱状图: 得分 / 精度 / 加速比 (_render_bars)
│   └── Top 算子表: 通过率 Top / 加速比 Top (_render_top_tables)
│
├── 5. 渲染 Section 4: Operator Details
│   └── 单表列出所有评测算子 (_render_operator_table)
│
└── 6. 追加认证印章 (SEAL_HTML 常量)
```

### 4.3 `report_generator.py` — 报告生成器（扩展）

原有功能保留，新增 `save_html()` 方法：

```python
class ReportGenerator:
    def save_all(self, report: EvalReport) -> Dict[str, Path]:
        return {
            'json': self.save_json(report),
            'markdown': self.save_markdown(report),
            'html': self.save_html(report),     # 新增
        }

    def save_html(self, report, filename=None) -> Path:
        # 读取 tasks/description.html
        # 调用 render_html_report()
        # 写入 reports/cann/{eval_code}.html
```

### 4.4 `Config` 扩展

```python
@dataclass
class Config:
    # 新增字段
    agent_skill: str = ""   # AI Agent/Skill 标识，为空时摘要不显示
    base_model: str = ""    # AI BaseModel 标识，为空时摘要不显示
```

### 4.5 `description.html` — 评测集描述模板

位于 `tasks/description.html`，作为报告的前缀模板，包含：
- HTML5 文档声明与完整 CSS 样式
- Header Bar（报告标题栏）
- Title Area（评测报告主标题与元信息）
- Abstract（摘要，含示例数据，运行时由 html_generator 替换）
- Section 1（评测集概述，Table 1 难度分级，1.2 评分体系）

模板末端有 `<!-- INSERTION POINT -->` 注释标记，**Section 2 起由代码动态生成**。

---

## 5. 数据流

```
评测运行
    │
    ├── Evaluator 执行算子评测
    │   └── EvalOperatorResult (per operator)
    │
    ▼
ReportGenerator.generate()
    │
    ├── ScoringCalculator → OperatorScoreInfo (per operator)
    ├── collect_setup_info(config) → setup_info dict
    │
    ▼
EvalReport (dataclass)
    ├── summary
    ├── setup_info        ← 来自 collect_setup_info()
    └── operators: List[OperatorReport]
        │
        ▼
ReportGenerator.save_all(report)
    │
    ├── save_json(report)
    │   └── reports/cann/eval_xxx.json
    │       ├── summary
    │       ├── setup_info        ← 嵌入 JSON
    │       └── operators[]
    │
    ├── save_markdown(report)
    │   └── reports/cann/eval_xxx.md
    │
    └── save_html(report)
        └── reports/cann/eval_xxx.html
            └── render_html_report(report, setup_info, "tasks/description.html")
                ├── description.html (CSS + Header + Abstract + Section 1)
                ├── 正则替换摘要动态字段
                ├── Section 2 (setup_info)
                ├── Section 3 (KPI + 表格 + 柱状图)
                ├── Section 4 (算子详情)
                └── 认证印章

# 或通过 gen_report.sh 从已有 JSON 重新生成 HTML：
gen_report.sh
    │
    └── 读取 eval_xxx.json
        ├── 反序列化为 EvalReport + OperatorReport
        └── render_html_report(report, setup_info, template)
            └── reports/cann/eval_xxx.html
```

---

## 6. 实施步骤

### 6.1 已完成

- [x] 创建 `tasks/description.html` 评测集描述模板
- [x] 实现 `setup_info.py`：metadata + environment 采集
- [x] 实现 `html_generator.py`：正则替换 + HTML 拼接渲染
- [x] 扩展 `report_generator.py`：新增 `save_html()` 输出
- [x] 扩展 `config.py`：新增 `agent_skill` / `base_model` 字段
- [x] 创建 `scripts/utils/gen_report.sh`：从 JSON 独立生成评测报告，支持 `--template` 自定义模板
- [x] 集成测试通过（`./scripts/run_evaluation.sh --source-dir examples/aclnn_launch_example`）

### 6.2 待完成

- [ ] Docker 环境检测改为软编码（从环境变量或 Dockerfile 注入）
- [ ] HTML 报告标题栏日期动态替换（`<title>` 中的 2026-05-25 → 实际时间戳）
- [ ] 认证印章日期动态替换
- [ ] 算子 Category 列从 `proto.yaml` 动态获取
- [ ] 支持 `--bench-name stanford` 等多 bench 的报告生成
- [ ] 报告增加评测趋势图（历史多次评测对比）
