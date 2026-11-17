#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


# flake8: noqa: E501

"""Build one self-contained report for a batch of operator workspaces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict

from kernel_eval.optimization_trajectory import build_trajectory_data, render_optimization_trajectory

REPORT_NAME = "report.html"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_batch_report_data(run_root: Path) -> Dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    config = _load_json(run_root / "config.json")
    progress = _load_json(run_root / "progress.json").get("operators", {})
    if not isinstance(progress, dict):
        progress = {}

    workspaces = {}
    for item in config.get("workspaces", []):
        if isinstance(item, dict) and item.get("operator") and item.get("workspace"):
            workspaces[str(item["operator"])] = Path(str(item["workspace"]))
    configured = config.get("parameters", {}).get("operators", [])
    names = [str(name) for name in configured] if isinstance(configured, list) else []
    for path in sorted(run_root.glob("*/work/workspace.json")):
        if path.is_file():
            name = str(_load_json(path).get("operator") or path.parent.parent.name)
            workspaces[name] = path.parent
            if name not in names:
                names.append(name)

    operators = []
    for name in names:
        local_workspace = run_root / name / "work"
        workspace = local_workspace if local_workspace.is_dir() else workspaces.get(name, local_workspace)
        trajectory = build_trajectory_data(workspace / "kernel_eval")
        if not trajectory.get("operator"):
            trajectory["operator"] = name
        rounds = trajectory.get("rounds", [])
        best = None
        best_rounds = []
        for item in rounds:
            try:
                value = float(item.get("speedup"))
            except (TypeError, ValueError):
                value = 0.0
            if item.get("success") and math.isfinite(value) and value > 0:
                best = value if best is None else max(best, value)
            best_rounds.append({
                "round": item.get("round"),
                "speedup": best if item.get("success") else 0.0,
                "success": item.get("success"),
            })
        state = progress.get(name, {}) if isinstance(progress.get(name), dict) else {}
        status = str(state.get("status") or ("reported" if rounds else "prepared"))
        if rounds and status == "prepared":
            status = "reported"
        operators.append(
            {
                "name": name,
                "status": status,
                "device_id": state.get("device_id"),
                "workspace": str(workspace),
                "rounds": rounds,
                "best_rounds": best_rounds,
                "baselines": trajectory.get("baselines", []),
                "trajectory_html": render_optimization_trajectory(trajectory),
                "best_speedup": best,
            }
        )

    best_values = [item["best_speedup"] for item in operators if item["best_speedup"] is not None]
    parameters = config.get("parameters", {}) if isinstance(config.get("parameters"), dict) else {}
    return {
        "run_root": str(run_root),
        "agent": str(parameters.get("agent") or "manual"),
        "model": str(parameters.get("model") or ""),
        "operators": operators,
        "summary": {
            "total": len(operators),
            "completed": sum(item["status"] in {"completed", "reported"} for item in operators),
            "failed_operators": sum(item["status"] in {"failed", "interrupted"} for item in operators),
            "timed_out": sum(item["status"] == "timeout" for item in operators),
            "failed_rounds": sum(
                round_.get("success") is not True
                for item in operators
                for round_ in item["rounds"]
            ),
            "mean_best_speedup": mean(best_values) if best_values else None,
        },
    }


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_batch_report(data: Dict[str, Any]) -> str:
    return _HTML_TEMPLATE.replace("__BATCH_DATA__", _json_for_script(data))


def write_batch_report(run_root: Path, output: Path | None = None) -> Path:
    run_root = Path(run_root).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else run_root / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_batch_report(build_batch_report_data(run_root)), encoding="utf-8")
    return output


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ascend Arena Batch Optimization</title>
  <style>
    :root { color-scheme: light; --page:#f4f6f5; --surface:#fff; --ink:#18201e; --muted:#62706c; --line:#d8dfdc; --primary:#007f73; --ok:#087f5b; --bad:#bd3448; --pending:#9a6700; font-family:Inter,system-ui,sans-serif; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--page); color:var(--ink); line-height:1.5; }
    header, main { width:min(1200px,calc(100% - 32px)); margin:auto; }
    header { padding:28px 0 20px; }
    h1,h2,h3,p { margin-top:0; } h1 { margin-bottom:4px; } h2 { margin-bottom:12px; }
    .muted { color:var(--muted); } .mono { font-family:ui-monospace,monospace; overflow-wrap:anywhere; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px; overflow:hidden; margin:16px 0 24px; border:1px solid var(--line); border-radius:8px; background:var(--line); }
    .metric { padding:15px; background:var(--surface); } .metric span { display:block; color:var(--muted); font-size:.8rem; } .metric strong { font-size:1.35rem; }
    .panel { margin-bottom:22px; padding:20px; border:1px solid var(--line); border-radius:8px; background:var(--surface); }
    .operator-list { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }
    button.operator { min-width:0; min-height:96px; padding:12px; border:1px solid var(--line); border-radius:6px; background:#fff; color:inherit; font:inherit; text-align:left; cursor:pointer; touch-action:manipulation; }
    button.operator strong { display:block; line-height:1.35; overflow-wrap:anywhere; word-break:break-word; }
    button.operator:hover, button.operator:focus-visible, button.operator[aria-pressed="true"] { border-color:var(--primary); box-shadow:0 0 0 2px #d9f0ed; }
    .status { display:inline-block; margin-top:6px; padding:2px 7px; border-radius:999px; background:#eef2f1; color:var(--muted); font-size:.75rem; }
    .status.completed,.status.reported { background:#e2f4ec; color:var(--ok); } .status.failed { background:#fae8eb; color:var(--bad); } .status.running,.status.initializing,.status.timeout { background:#fff3cd; color:var(--pending); }
    svg { display:block; width:100%; min-height:300px; } .grid { stroke:#e7ecea; } .axis { stroke:#9aa8a4; } .line { fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; } .point { fill:#fff; stroke-width:2.5; } text { fill:var(--muted); font:12px system-ui,sans-serif; }
    .chart-wrap { position:relative; } .chart-hit { fill:transparent; cursor:crosshair; } .hover-guide { stroke:#667570; stroke-width:1.5; stroke-dasharray:5 5; pointer-events:none; }
    .chart-tooltip { position:absolute; z-index:2; min-width:250px; max-width:min(360px,calc(100% - 16px)); padding:10px 12px; border:1px solid var(--line); border-radius:6px; background:rgba(255,255,255,.97); box-shadow:0 8px 24px rgba(24,32,30,.16); visibility:hidden; opacity:0; pointer-events:none; font-size:.78rem; font-variant-numeric:tabular-nums; }
    .chart-tooltip.visible { visibility:visible; opacity:1; } .tooltip-title { margin-bottom:6px; font-weight:700; color:var(--ink); } .tooltip-row { display:grid; grid-template-columns:16px minmax(0,1fr) auto; align-items:center; gap:7px; padding:2px 0; } .tooltip-swatch { width:14px; border-top-width:3px; } .tooltip-name { overflow-wrap:anywhere; } .tooltip-value { color:var(--ink); font-weight:600; }
    .chart-legend { display:flex; flex-wrap:wrap; gap:8px 18px; margin:0 0 12px; }
    .legend-item { display:inline-flex; align-items:center; gap:7px; min-width:0; color:var(--muted); font-size:.8rem; }
    .legend-line { width:24px; flex:0 0 24px; border-top-width:3px; }
    .table-wrap { overflow:auto; } table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; } th,td { padding:9px 10px; border-bottom:1px solid #e9eeec; text-align:left; } th { color:var(--muted); font-size:.78rem; } td.number,th.number { text-align:right; }
    details { margin-top:10px; border:1px solid var(--line); border-radius:6px; } summary { padding:10px 12px; cursor:pointer; font-weight:600; } pre { max-height:460px; margin:0; overflow:auto; padding:12px; background:#171b1a; color:#e8eeec; font:12px/1.55 ui-monospace,monospace; white-space:pre; }
    .empty { padding:24px; color:var(--muted); text-align:center; }
    .trajectory-section { margin-bottom:24px; scroll-margin-top:16px; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:var(--surface); }
    .trajectory-frame { display:block; width:100%; min-height:900px; overflow:hidden; border:0; background:var(--page); }
    @media(max-width:700px){ .metrics{grid-template-columns:1fr 1fr;} .panel{padding:14px;} }
  </style>
</head>
<body>
  <header><h1>Ascend Arena 批量优化报告</h1></header>
  <main>
    <section id="metrics" class="metrics" aria-label="批次汇总"></section>
    <section class="panel"><h2>整体优化情况（最优性能）</h2><p class="muted">横轴为优化轮数，纵轴为截至当前轮每个算子的最优逐 case 平均加速比。失败轮次显示为 0x，以 × 标记；曲线覆盖各算子的实际优化轮次。</p><div id="overall-legend" class="chart-legend" aria-label="算子图例"></div><div id="overall-wrap" class="chart-wrap"><svg id="overall" role="img" aria-label="各算子截至当前轮次的最优平均加速比"></svg><div id="overall-tooltip" class="chart-tooltip" aria-hidden="true"></div></div></section>
    <section class="panel"><h2>算子列表</h2><div id="operators" class="operator-list"></div></section>
    <div id="trajectories" aria-live="polite"></div>
  </main>
  <script id="batch-data" type="application/json">__BATCH_DATA__</script>
  <script>
  (() => {
    const data=JSON.parse(document.getElementById('batch-data').textContent), ops=data.operators||[];
    const el=id=>document.getElementById(id), fmt=n=>n==null?'N/A':new Intl.NumberFormat('zh-CN',{maximumFractionDigits:4}).format(n);
    const metrics=[['算子',data.summary.total],['已完成',data.summary.completed],['会话到时',data.summary.timed_out],['失败算子',data.summary.failed_operators],['失败轮次',data.summary.failed_rounds],['平均最佳加速比',data.summary.mean_best_speedup==null?'N/A':`${fmt(data.summary.mean_best_speedup)}x`]];
    metrics.forEach(([label,value])=>{const d=document.createElement('div');d.className='metric';const s=document.createElement('span'),b=document.createElement('strong');s.textContent=label;b.textContent=value;d.append(s,b);el('metrics').append(d);});
    const chartColors=['#007f73','#b45309','#2563eb','#7c3aed','#be123c','#4d7c0f','#0e7490','#9f1239'],chartDashes=['','9 4','2 4','12 4 2 4'];
    function chart(svg,series){
      svg.replaceChildren();
      const ns='http://www.w3.org/2000/svg',node=name=>document.createElementNS(ns,name),w=900,h=300,m={l:68,r:24,t:22,b:62},iw=w-m.l-m.r,ih=h-m.t-m.b,maxRound=Math.max(0,...series.map(op=>Math.max(0,...op.rounds.map(item=>Number(item.round)||0))));
      svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
      if(!series.length||!maxRound){const empty=node('text');empty.setAttribute('x',w/2);empty.setAttribute('y',h/2);empty.setAttribute('text-anchor','middle');empty.textContent='暂无优化轨迹数据';svg.append(empty);return;}
      const values=series.flatMap(op=>op.rounds.map(item=>Number(item.speedup)).filter(value=>Number.isFinite(value)&&value>0)),maxValue=Math.max(1,...values),ceiling=Math.ceil(maxValue*4)/4,x=round=>m.l+(maxRound===1?iw/2:(round-1)*iw/(maxRound-1)),y=value=>m.t+ih-value/ceiling*ih;
      for(let tick=0;tick<=4;tick++){const value=ceiling*tick/4,py=y(value),line=node('line');line.setAttribute('x1',m.l);line.setAttribute('x2',w-m.r);line.setAttribute('y1',py);line.setAttribute('y2',py);line.setAttribute('class','grid');svg.append(line);const label=node('text');label.setAttribute('x',m.l-10);label.setAttribute('y',py+4);label.setAttribute('text-anchor','end');label.textContent=value.toFixed(2);svg.append(label);}
      const tickStep=Math.max(1,Math.ceil(maxRound/8)),tickRounds=[];
      for(let round=1;round<=maxRound;round+=tickStep)tickRounds.push(round);
      if(tickRounds[tickRounds.length-1]!==maxRound)tickRounds.push(maxRound);
      tickRounds.forEach(round=>{const label=node('text');label.setAttribute('x',x(round));label.setAttribute('y',h-31);label.setAttribute('text-anchor','middle');label.textContent=`R${round}`;svg.append(label);});
      series.forEach((op,index)=>{const color=chartColors[index%chartColors.length],dash=chartDashes[index%chartDashes.length],points=op.rounds.filter(item=>item.speedup!=null&&Number.isFinite(Number(item.speedup))&&Number(item.speedup)>=0);if(points.length){const line=node('polyline');line.setAttribute('points',points.map(item=>`${x(Number(item.round)||1)},${y(Number(item.speedup))}`).join(' '));line.setAttribute('class','line');line.setAttribute('stroke',color);if(dash)line.setAttribute('stroke-dasharray',dash);svg.append(line);}points.forEach(item=>{const value=Number(item.speedup),px=x(Number(item.round)||1),py=y(value),point=node(item.success===false?'path':'circle'),title=node('title');if(item.success===false){point.setAttribute('d',`M ${px-5} ${py-5} L ${px+5} ${py+5} M ${px-5} ${py+5} L ${px+5} ${py-5}`);}else{point.setAttribute('cx',px);point.setAttribute('cy',py);point.setAttribute('r',4);}point.setAttribute('class','point');point.setAttribute('stroke',color);title.textContent=`${op.name} · R${item.round} · ${value.toFixed(2)}x${item.success===false?' · 失败':''}`;point.append(title);svg.append(point);});});
      const yLabel=node('text');yLabel.setAttribute('x',18);yLabel.setAttribute('y',m.t+ih/2);yLabel.setAttribute('text-anchor','middle');yLabel.setAttribute('transform',`rotate(-90 18 ${m.t+ih/2})`);yLabel.textContent='最优平均加速比';svg.append(yLabel);
      const xLabel=node('text');xLabel.setAttribute('x',m.l+iw/2);xLabel.setAttribute('y',h-6);xLabel.setAttribute('text-anchor','middle');xLabel.textContent='优化轮数';svg.append(xLabel);
      const guide=node('line');guide.setAttribute('y1',m.t);guide.setAttribute('y2',m.t+ih);guide.setAttribute('class','hover-guide');guide.style.display='none';svg.append(guide);
      const hit=node('rect');hit.setAttribute('x',m.l);hit.setAttribute('y',m.t);hit.setAttribute('width',iw);hit.setAttribute('height',ih);hit.setAttribute('class','chart-hit');svg.append(hit);
      const wrap=el('overall-wrap'),tip=el('overall-tooltip');
      hit.addEventListener('pointermove',event=>{const cursor=svg.createSVGPoint();cursor.x=event.clientX;cursor.y=event.clientY;const point=cursor.matrixTransform(svg.getScreenCTM().inverse()),px=Math.max(m.l,Math.min(w-m.r,point.x)),round=maxRound===1?1:Math.max(1,Math.min(maxRound,Math.round(1+(px-m.l)/iw*(maxRound-1))));guide.setAttribute('x1',px);guide.setAttribute('x2',px);guide.style.display='';if(tip.dataset.round!==String(round)){const title=document.createElement('div');title.className='tooltip-title';title.textContent=`第 ${round} 轮`;tip.replaceChildren(title,...series.map((op,index)=>{const row=document.createElement('div'),swatch=document.createElement('span'),name=document.createElement('span'),value=document.createElement('span'),result=op.rounds.filter(item=>Number(item.round)<=round).pop(),speed=result&&result.speedup!=null?Number(result.speedup):NaN;row.className='tooltip-row';swatch.className='tooltip-swatch';swatch.style.borderTopColor=chartColors[index%chartColors.length];swatch.style.borderTopStyle=['solid','dashed','dotted','dashed'][index%4];name.className='tooltip-name';name.textContent=op.name;value.className='tooltip-value';value.textContent=result&&result.success===false?'0x · 失败':Number.isFinite(speed)&&speed>0?`${speed.toFixed(2)}x`:'N/A';row.append(swatch,name,value);return row;}));tip.dataset.round=String(round);}tip.classList.add('visible');tip.setAttribute('aria-hidden','false');const box=wrap.getBoundingClientRect(),pointerX=event.clientX-box.left,pointerY=event.clientY-box.top,tipWidth=tip.offsetWidth,tipHeight=tip.offsetHeight,left=pointerX+12+tipWidth>box.width-8?pointerX-tipWidth-12:pointerX+12;tip.style.left=`${Math.max(8,left)}px`;tip.style.top=`${Math.max(8,Math.min(pointerY+12,box.height-tipHeight-8))}px`;});
      hit.addEventListener('pointerleave',()=>{guide.style.display='none';tip.classList.remove('visible');tip.setAttribute('aria-hidden','true');});
    }
    function renderLegend(series){el('overall-legend').replaceChildren(...series.map((op,index)=>{const item=document.createElement('span'),line=document.createElement('span'),label=document.createElement('span');item.className='legend-item';line.className='legend-line';line.style.borderTopColor=chartColors[index%chartColors.length];line.style.borderTopStyle=['solid','dashed','dotted','dashed'][index%4];label.textContent=op.name;item.append(line,label);return item;}));}
    const measured=ops.filter(op=>(op.best_rounds||[]).length).map(op=>({...op,rounds:op.best_rounds}));renderLegend(measured);chart(el('overall'),measured);
    function statusNode(status){const s=document.createElement('span');s.className=`status ${status}`;s.textContent=status==='timeout'?'会话到时':status;return s;}
    function renderTrajectory(op,index){document.querySelectorAll('button.operator').forEach((button,itemIndex)=>button.setAttribute('aria-pressed',itemIndex===index));const section=document.createElement('section');section.id='selected-trajectory';section.className='trajectory-section';const frame=document.createElement('iframe');frame.className='trajectory-frame';frame.title=`${op.name} 优化轨迹`;frame.setAttribute('scrolling','no');frame.addEventListener('load',()=>{const doc=frame.contentDocument;if(!doc)return;doc.documentElement.style.overflow='hidden';doc.body.style.overflow='hidden';const resize=()=>{frame.style.height='0px';frame.style.height=`${Math.max(doc.body.scrollHeight,doc.documentElement.scrollHeight)}px`;};requestAnimationFrame(resize);if(window.ResizeObserver){const observer=new ResizeObserver(resize);observer.observe(doc.documentElement);observer.observe(doc.body);}});frame.srcdoc=op.trajectory_html;section.append(frame);el('trajectories').replaceChildren(section);}
    ops.forEach((op,index)=>{const button=document.createElement('button');button.type='button';button.className='operator';button.setAttribute('aria-pressed','false');button.setAttribute('aria-controls','trajectories');const strong=document.createElement('strong');strong.textContent=op.name;const speed=document.createElement('div');speed.className='muted';speed.textContent=op.best_speedup==null?'最佳 N/A':`最佳 ${fmt(op.best_speedup)}x`;button.append(strong,speed,statusNode(op.status));button.addEventListener('click',()=>renderTrajectory(op,index));el('operators').append(button);});
    if(!ops.length){const empty=document.createElement('div');empty.className='panel empty';empty.textContent='未发现算子 workspace。';el('trajectories').append(empty);}
  })();
  </script>
</body>
</html>"""
