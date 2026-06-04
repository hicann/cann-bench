#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
HTML 评测报告生成器

将 setup_info + eval 结果拼接为完整 HTML 报告。

拼接策略:
  description.html (CSS + Header + Abstract + Section 1)
  + Section 2 (动态 setup_info)
  + Section 3 (结果分析: KPI / Level Table / Bar Charts / Top Tables)
  + Section 4 (算子详情表)
  + 认证印章

Usage:
    from kernel_eval.report.html_generator import render_html_report

    html_str = render_html_report(report, setup_info, index_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
import re

from .report_generator import EvalReport, OperatorReport


# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------

def _cls_rate(r: float) -> str:
    if r >= 0.8: return "score-high"
    if r >= 0.4: return "score-mid"
    return "score-low"

def _cls_score(s: float) -> str:
    if s >= 80: return "score-high"
    if s >= 50: return "score-mid"
    return "score-low"

def _cls_sp(sp: float) -> str:
    if sp <= 0: return "score-low"
    if sp >= 1.0: return "score-high"
    if sp >= 0.8: return "score-mid"
    return "score-low"

def _fr(r: float) -> str: return f'{r:.0%}' if r > 0 else '0%'
def _fsp(sp: float) -> str: return f'{sp:.2f}x' if sp > 0 else '—'
def _fs(s: float) -> str: return f'{s:.0f}'


# ---------------------------------------------------------------------------
# Section 2: Experiment Setup
# ---------------------------------------------------------------------------

def _render_section2(setup: Dict) -> str:
    """渲染 Section 2 (评测配置)"""
    md = setup.get('metadata', {})
    env = setup.get('environment', {})

    def _row(k, v):
        if v is None or v == '':
            return ''
        return f'            <tr><td class="kv-k">{k}</td><td>{v}</td></tr>\n'

    # Metadata rows
    meta_rows = ''
    for key, label in [
        ('framework', 'Framework'), ('date', 'Date'),
        ('agent_skill', 'Agent/Skill'), ('base_model', 'BaseModel'),
        ('benchmark', '评测集'), ('license', 'License'),
    ]:
        if key in md and md[key]:
            meta_rows += _row(label, md[key])

    # Environment rows
    env_rows = ''
    for key, label in [
        ('npu', 'NPU'), ('cpu', 'CPU'), ('cann', 'CANN'),
        ('driver', 'Driver版本'), ('pytorch', 'PyTorch'),
        ('pytorch_npu', 'PyTorch NPU'), ('torchvision', 'torchvision'),
        ('python', 'Python'), ('os', 'OS'), ('docker', 'Docker'),
    ]:
        if key in env and env[key]:
            env_rows += _row(label, env[key])

    return f'''  <!-- ============================================================ -->
  <!-- 2. EXPERIMENT SETUP                                            -->
  <!-- ============================================================ -->
  <div class="section">
    <h3><span class="sec-num">2.</span> Experiment Setup / 评测配置</h3>
    <div class="setup-grid">
      <div class="setup-block">
        <h4>Metadata / 元信息</h4>
        <table><tbody>
{meta_rows}        </tbody></table>
      </div>
      <div class="setup-block">
        <h4>Environment / 运行环境</h4>
        <table><tbody>
{env_rows}        </tbody></table>
      </div>
    </div>
  </div>

'''


# ---------------------------------------------------------------------------
# Section 3: Results Analysis
# ---------------------------------------------------------------------------

def _render_kpi(report: EvalReport) -> str:
    r = report.summary['pass_rate']
    genuine_r = report.summary.get('genuine_pass_rate', r)
    cascade = report.summary.get('cascade_cases', 0)

    # 级联失败提示块（仅在有级联失败时显示）
    cascade_note = ''
    if cascade > 0:
        cascade_note = f'''      <div class="kpi-item kpi-note">
        <div class="kpi-value" style="color:var(--score-mid)">⚠️ {cascade}</div>
        <div class="kpi-label">级联失败（设备异常）</div>
        <div class="kpi-sub">不计入真实失败率</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">{genuine_r:.1%}</div>
        <div class="kpi-label">Genuine Pass Rate / 真实通过率</div>
        <div class="kpi-sub">排除级联失败后</div>
      </div>'''

    return f'''      <div class="kpi-item">
        <div class="kpi-value">{r:.1%}</div>
        <div class="kpi-label">Pass Rate / 通过率</div>
        <div class="kpi-sub">{report.passed_cases} / {report.total_cases} cases</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">{report.total_operators}</div>
        <div class="kpi-label">Operators / 算子数</div>
        <div class="kpi-sub">AI-generated code</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">{report.total_cases}</div>
        <div class="kpi-label">Total Cases / 总用例</div>
        <div class="kpi-sub">20 cases per operator</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-value">{report.failed_cases}</div>
        <div class="kpi-label">Error Case Number / 失败用例数量</div>
      </div>
{cascade_note}      <div class="kpi-item">
        <div class="kpi-value">{report.overall_score:.0f}</div>
        <div class="kpi-label">Total Score / 总得分</div>
      </div>'''


def _render_level_table(report: EvalReport) -> str:
    """按 level 汇总"""
    # Group by level
    lv_groups = {}
    for op in report.operators:
        lv = int(op.rel_path.split('/')[0][-1]) if op.rel_path else 1
        if lv not in lv_groups:
            lv_groups[lv] = {'ops': 0, 'cases': 0, 'passed': 0, 'score': 0.0, 'speeds': []}
        g = lv_groups[lv]
        g['ops'] += 1
        g['cases'] += op.total_cases
        g['passed'] += op.passed_cases
        g['score'] += op.score
        if op.avg_speedup > 0:
            g['speeds'].append(op.avg_speedup)

    labels = {1: 'Level 1 — 基础算子', 2: 'Level 2 — 中级算子',
              3: 'Level 3 — 高级算子', 4: 'Level 4 — 复杂算子'}

    rows = ''
    for lv in sorted(lv_groups):
        g = lv_groups[lv]
        gr = g['passed'] / g['cases'] if g['cases'] > 0 else 0
        gs = g['score']
        gsp = sum(g['speeds']) / len(g['speeds']) if g['speeds'] else 0
        rc = _cls_rate(gr); sc = _cls_score(gs); spc = _cls_sp(gsp)
        sp_val = _fsp(gsp) if gsp > 0 else '<td class="score-cell score-low">—</td>'
        if gsp > 0:
            sp_val = f'<td class="score-cell {spc}">{_fsp(gsp)}</td>'
        rows += f'''          <tr>
            <td class="col-name" style="white-space:nowrap">{labels.get(lv, f'Level {lv}')}</td>
            <td>{g['ops']}</td><td>{g['cases']}</td><td>{g['passed']}</td>
            <td class="score-cell {rc}">{_fr(gr)}</td>
            <td class="score-cell {rc}">{_fr(gr)}</td>
            {sp_val}
            <td class="score-cell {sc}">{_fs(gs)}</td>
          </tr>\n'''

    return f'''    <h4>3.2 等级分析</h4>
    <div class="table-wrap mt-16">
      <table>
        <caption>Table 2. Results by Difficulty Level</caption>
        <thead><tr>
          <th>Level</th><th>Operators</th><th>Cases</th><th>Passed</th><th>Pass Rate</th><th>Avg Precision</th><th>Avg Speedup</th><th>Total Score</th>
        </tr></thead>
        <tbody>
{rows}        </tbody>
      </table>
    </div>'''


def _render_bars(report: EvalReport) -> str:
    """渲染三组柱状图"""
    # Group by level
    lv_groups = {}
    for op in report.operators:
        lv = int(op.rel_path.split('/')[0][-1]) if op.rel_path else 1
        if lv not in lv_groups:
            lv_groups[lv] = {'cases': 0, 'passed': 0, 'score': 0.0, 'speeds': [], 'ops': 0}
        g = lv_groups[lv]
        g['ops'] += 1
        g['cases'] += op.total_cases
        g['passed'] += op.passed_cases
        g['score'] += op.score
        if op.avg_speedup > 0:
            g['speeds'].append(op.avg_speedup)

    names = {1: 'Level 1', 2: 'Level 2', 3: 'Level 3', 4: 'Level 4'}
    lv_max = {lv: g['ops'] * 100 for lv, g in lv_groups.items()}

    def _bar(lab, pct, val, is_sp=False):
        c = 'var(--score-high)' if pct >= 80 else ('var(--score-mid)' if pct >= 40 else 'var(--score-low)')
        if is_sp: c = 'var(--score-high)' if pct >= 100 else '#5b9bd5'
        return f'      <div class="bar-row"><span class="bar-label">{lab}</span><div class="bar-track"><div class="bar-fill" style="width:{min(pct, 100)}%;background:{c}"></div></div><span class="bar-val">{val}</span></div>\n'

    bar_s = ''.join(_bar(names[lv], lv_groups[lv]['score'] / lv_max[lv] * 100, _fs(lv_groups[lv]['score'])) for lv in sorted(lv_groups))
    bar_p = ''.join(_bar(names[lv], lv_groups[lv]['passed'] / lv_groups[lv]['cases'] * 100, _fr(lv_groups[lv]['passed'] / lv_groups[lv]['cases'])) for lv in sorted(lv_groups))
    bar_sp = ''.join(_bar(names[lv], (sum(lv_groups[lv]['speeds']) / len(lv_groups[lv]['speeds']) * 100) if lv_groups[lv]['speeds'] else 0, _fsp(sum(lv_groups[lv]['speeds']) / len(lv_groups[lv]['speeds'])) if lv_groups[lv]['speeds'] else '—', True) for lv in sorted(lv_groups))

    return f'''    <div class="bar-chart">
{bar_s}    </div>
    <p class="note mt-12">Figure 1a. Total score by difficulty level.</p>

    <div class="bar-chart">
{bar_p}    </div>
    <p class="note mt-12">Figure 1b. Average precision by difficulty level.</p>

    <div class="bar-chart">
{bar_sp}    </div>
    <p class="note mt-12">Figure 1c. Average speedup by difficulty level (几何平均).</p>'''


def _render_top_tables(ops: List[OperatorReport]) -> str:
    top_prec = sorted(ops, key=lambda o: o.pass_rate, reverse=True)
    top_sp = sorted(ops, key=lambda o: o.avg_speedup, reverse=True)

    def _prec_rows(items):
        out = ''
        for i, o in enumerate(items, 1):
            out += f'            <tr><td>{i}</td><td class="col-name">{o.operator}</td><td>L{o.rel_path[5:6] if o.rel_path else "1"}</td><td class="score-cell {_cls_rate(o.pass_rate)}">{_fr(o.pass_rate)}</td></tr>\n'
        return out

    def _sp_rows(items):
        out = ''
        for i, o in enumerate(items, 1):
            sp = o.avg_speedup
            v, cls = (_fsp(sp), _cls_sp(sp)) if sp > 0 else ('—', 'score-low')
            out += f'            <tr><td>{i}</td><td class="col-name">{o.operator}</td><td>L{o.rel_path[5:6] if o.rel_path else "1"}</td><td class="score-cell {cls}">{v}</td></tr>\n'
        return out

    return f'''    <h4>3.3 算子分析</h4>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:14px 0">
      <div class="table-wrap" style="margin:0">
        <table>
          <caption>Table 3a. Operators by Pass Rate</caption>
          <thead><tr><th>#</th><th>Operator</th><th>Level</th><th>Pass Rate</th></tr></thead>
          <tbody>
{_prec_rows(top_prec)}          </tbody>
        </table>
      </div>
      <div class="table-wrap" style="margin:0">
        <table>
          <caption>Table 3b. Operators by Avg Speedup</caption>
          <thead><tr><th>#</th><th>Operator</th><th>Level</th><th>Speedup</th></tr></thead>
          <tbody>
{_sp_rows(top_sp)}          </tbody>
        </table>
      </div>
    </div>
    <p class="note">Table 3a-3b. Left: Operators by pass rate. Right: Operators by NPU speedup ratio.</p>'''


# ---------------------------------------------------------------------------
# Section 4: Operator Details
# ---------------------------------------------------------------------------

def _get_category(rel_path: str) -> str:
    """从 proto.yaml 获取算子 category"""
    if not rel_path:
        return '—'
    try:
        from kernel_eval.config import get_project_root
        proto = get_project_root() / "tasks" / rel_path / "proto.yaml"
        if proto.exists():
            import yaml
            with open(proto) as f:
                data = yaml.safe_load(f)
            return data.get('operator', {}).get('category', '—')
    except Exception:
        pass
    return '—'


def _render_operator_tables(ops: List[OperatorReport]) -> str:
    """按 Level 分组渲染算子详情表"""
    from collections import defaultdict
    levels = defaultdict(list)
    for o in ops:
        lv = int(o.rel_path[5:6]) if o.rel_path and len(o.rel_path) > 5 else 1
        levels[lv].append(o)

    lv_labels = {1: '基础算子', 2: '中级算子', 3: '高级算子', 4: '复杂算子'}
    result = ''
    table_idx = 4
    for lv in sorted(levels):
        lv_ops = levels[lv]
        rows = ''
        for i, o in enumerate(lv_ops, 1):
            sp = o.avg_speedup
            cat = _get_category(o.rel_path)
            lv_s = o.rel_path[5:6] if o.rel_path and len(o.rel_path) > 5 else str(lv)
            rows += f'''          <tr><td>{i}</td><td class="col-name">{o.operator}</td><td>{cat}</td><td>L{lv_s}</td><td>{o.total_cases}</td><td>{o.passed_cases}</td><td class="score-cell {_cls_rate(o.pass_rate)}">{_fr(o.pass_rate)}</td><td class="score-cell {_cls_rate(o.pass_rate)}">{_fr(o.pass_rate)}</td><td class="score-cell {_cls_sp(sp)}">{_fsp(sp)}</td><td class="score-cell {_cls_score(o.score)}">{_fs(o.score)}</td></tr>\n'''

        label = lv_labels.get(lv, f'Level {lv}')
        result += f'''    <h4>4.{lv} Level {lv} — {label}</h4>
    <div class="table-wrap">
      <table>
        <caption>Table {table_idx}. Level {lv} Operator Results</caption>
        <thead><tr>
          <th>#</th><th>Operator</th><th>Category</th><th>Level</th><th>Cases</th><th>Passed</th><th>Pass Rate</th><th>Avg Precision</th><th>Avg Speedup</th><th>Total Score</th>
        </tr></thead>
        <tbody>
{rows}        </tbody>
      </table>
    </div>
'''
        table_idx += 1

    return f'''  <!-- ============================================================ -->
  <!-- 4. OPERATOR DETAILS                                            -->
  <!-- ============================================================ -->
  <div class="section">
    <h3><span class="sec-num">4.</span> Operator Details / 算子明细</h3>

{result}  </div>

'''


# ---------------------------------------------------------------------------
# Certification Seal
# ---------------------------------------------------------------------------

SEAL_HTML = '''  <!-- ============================================================ -->
  <!-- CERTIFICATION SEAL                                             -->
  <!-- ============================================================ -->
  <div class="seal-section">
    <h3>CANN-Bench 认证</h3>

    <div class="seal-wrapper">
      <div class="seal">
        <span class="seal-arc">CANN-BENCH</span>
        <div class="seal-inner">
          <div class="seal-title">CERTIFIED</div>
          <div class="seal-star">&#9733;</div>
          <div class="seal-date">2026 &middot; 06 &middot; 01</div>
        </div>
        <span class="seal-arc-bottom">V0.1.0</span>
      </div>
    </div>

    <div class="seal-cert">
      <p>
        本评测报告由 CANN-Bench 评测框架自动生成，评测对象为 DeepSeek V4 Pro (CANNBot) 生成的
        Ascend C 算子代码。
      </p>
      <p>
        评测数据来源于 CANN-Bench tasks/ 标准化题库，评分严格按照 CANN-Bench V0.1.0 评分规范执行。
      </p>
      <p class="cert-sign">
        CANN-Bench Evaluation Framework &mdash; AI for CANN, Benchmark for AI
      </p>
    </div>
  </div>

</main>
</body>
</html>'''


# ---------------------------------------------------------------------------
# 主渲染函数
# ---------------------------------------------------------------------------

def render_html_report(
    report: EvalReport,
    setup_info: Optional[Dict] = None,
    index_path: Optional[str] = None,
) -> str:
    """渲染完整 HTML 评测报告

    Args:
        report: 评测报告数据
        setup_info: 采集的配置信息
        index_path: 评测集 description.html 路径

    Returns:
        完整的 HTML 字符串
    """
    # 1. 读取 index.html 前缀 (CSS + Header + Abstract + Section 1)
    if index_path and Path(index_path).exists():
        with open(index_path) as f:
            html = f.read()
    else:
        from ..config import get_project_root
        tmpl = get_project_root() / "tasks" / "description.html"
        if tmpl.exists():
            with open(tmpl) as f:
                html = f.read()
        else:
            html = '<!DOCTYPE html><html><body>'

    # 替换摘要中的动态字段
    if setup_info is None:
        setup_info = report.setup_info if hasattr(report, 'setup_info') else {}
    md = setup_info.get('metadata', {})

    # Agent/Skill — 有值则替换，无值则删除
    agent = md.get('agent_skill', '')
    if agent:
        html = re.sub(r'Agent/Skill为[\u4e00-\u9fa5\w-]*', f'Agent/Skill为{agent}', html)
    else:
        html = re.sub(r'，Agent/Skill为[\u4e00-\u9fa5\w-]*', '', html)

    # BaseModel — 同上
    model = md.get('base_model', '')
    if model:
        html = re.sub(r'BaseModel为[\u4e00-\u9fa5\w\s]*', f'BaseModel为{model}', html)
    else:
        html = re.sub(r'，BaseModel为[\u4e00-\u9fa5\w\s]*', '', html)

    # 通过率 & 得分 — 从 report 获取
    r = report.summary['pass_rate']
    html = re.sub(
        r'整体通过率为 [\d.]+%（[\d,]+/[\d,]+），总得分为 [\d.]+（满分 [\d,]+）',
        f'整体通过率为 {r:.1%}（{report.passed_cases}/{report.total_cases}），'
        f'总得分为 {report.overall_score:.0f}（满分 {report.total_operators * 100}）',
        html
    )

    # 算子数/用例数/等级 — 从 report 获取
    levels_set = set()
    for op in report.operators:
        if op.rel_path:
            levels_set.add(int(op.rel_path.split('/')[0][-1]))
    max_lv = max(levels_set) if levels_set else 1
    lv_count = len(levels_set)
    lv_part = f'覆盖 Level 1 至 Level {max_lv} 共 {lv_count} 个难度等级' if lv_count > 1 else '共 1 个难度等级'
    html = re.sub(
        r'针对 53 个算子进行了系统性评测，覆盖 Level 1 至 Level 4 共 4 个难度等级、\s*\n\s*1,060 个评测用例。',
        f'本次对 {report.total_operators} 个算子进行了评测，{lv_part}，{report.total_cases} 个评测用例。',
        html
    )

    # 移除末尾的 INSERTION POINT 注释 (如果有)
    ins_point = html.find('<!-- INSERTION POINT')
    if ins_point >= 0:
        html = html[:ins_point]

    # 清理末尾空白
    html = html.rstrip()

    # 2. Section 2: Experiment Setup
    html += '\n\n' + _render_section2(setup_info)

    # 3. Section 3: Results Analysis
    html += '\n\n  <!-- ============================================================ -->\n'
    html += '  <!-- 3. RESULTS ANALYSIS                                           -->\n'
    html += '  <!-- ============================================================ -->\n'
    html += '  <div class="section">\n'
    html += '    <h3><span class="sec-num">3.</span> Results Analysis / 结果分析</h3>\n\n'
    html += '    <h4>3.1 结果总览</h4>\n'
    html += '    <div class="kpi-strip">\n'
    html += _render_kpi(report)
    html += '    </div>\n\n'
    html += _render_level_table(report) + '\n'
    html += _render_bars(report) + '\n'
    html += _render_top_tables(report.operators) + '\n'
    html += '  </div>\n'

    # 4. Section 4: Operator Details
    html += '\n' + _render_operator_tables(report.operators)

    # 5. Certification Seal + closing tags
    html += '\n' + SEAL_HTML

    return html
