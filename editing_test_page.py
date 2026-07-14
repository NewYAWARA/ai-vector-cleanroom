# -*- coding: utf-8 -*-
"""Build a self-contained Stage 2 timed designer handoff test page."""

from __future__ import annotations

import hashlib
from html import escape as html_escape
import json
from pathlib import Path
from typing import Mapping, Sequence


TASKS = (
    ("global_recolour", "全域換色", "把主要綠色系改成指定新色，含填色、筆畫與漸層色標。"),
    ("gradient_adjust", "調整漸層", "調整主要漸層的方向或色標，保留原本物件範圍。"),
    ("main_ring_adjust", "修改主圓環", "調整主外環半徑或粗細，不破壞其餘圖稿。"),
    ("move_local_component", "移動局部元件", "選取一組左右側局部圖文元件並移動或縮放。"),
    ("hide_decorations", "隱藏裝飾", "隱藏網點與平行線裝飾，保留主要標誌。"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_rows(item_index: int) -> str:
    rows = []
    for task_id, label, instruction in TASKS:
        rows.append(f'''<tr data-task="{task_id}">
 <td><b>{html_escape(label)}</b><small>{html_escape(instruction)}</small></td>
 <td><button class="timer" data-kind="vector">開始</button>
     <input class="seconds vector" type="number" min="0" step="1" inputmode="numeric" aria-label="向量接手秒數"> 秒</td>
 <td><button class="timer" data-kind="baseline">開始</button>
     <input class="seconds baseline" type="number" min="0" step="1" inputmode="numeric" aria-label="從頭重畫秒數"> 秒
     <select class="baseline-kind" aria-label="基準時間來源">
       <option value="actual">實際計時</option><option value="estimated">設計師估算</option><option value="none" selected>未提供</option>
     </select></td>
 <td><select class="task-status"><option value="completed">完成</option><option value="partial">部分完成</option>
     <option value="unable">無法完成</option><option value="not_applicable">不適用</option></select></td>
 <td><input class="task-note" type="text" placeholder="卡在哪裡／需修什麼"></td>
</tr>''')
    return "\n".join(rows)


def build_editing_test_page(output_dir: str | Path,
                            results: Sequence[Mapping[str, object]], *,
                            tool_version: str) -> Path:
    """Write an offline-capable timed editing page and return its path."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_root = output.resolve()
    cards = []
    for index, result in enumerate(results):
        directory_name = str(result.get("dir") or "")
        directory = (output / directory_name).resolve()
        svg_rel = str(result.get("svg") or "")
        svg = (output / Path(svg_rel)).resolve()
        if (output_root not in directory.parents
                or output_root not in svg.parents):
            continue
        source = directory / "source_reference.png"
        if not (directory.is_dir() and svg.is_file() and source.is_file()):
            continue
        report_file = directory / "report.json"
        report = {}
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        meta = {
            "name": str(result.get("base") or directory_name),
            "input": str(result.get("input") or report.get("input") or ""),
            "svg": svg_rel.replace("\\", "/"),
            "svg_sha256": _file_sha256(svg),
            "tool_version": str(report.get("tool_version") or tool_version),
            "visual_acceptance_status": str(
                result.get("visual_acceptance_status") or
                report.get("visual_acceptance_status") or "not_audited"),
            "editability_status": str(
                result.get("editability_status") or
                report.get("editability_status") or "not_audited"),
        }
        meta_attr = html_escape(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            quote=True)
        source_rel = f"{directory_name}/source_reference.png".replace("\\", "/")
        default_order = "vector_first" if index % 2 == 0 else "redraw_first"
        cards.append(f'''<section class="case" data-meta="{meta_attr}">
 <h2>案例 {index + 1}：{html_escape(meta['name'])}</h2>
 <div class="links"><a href="{html_escape(svg_rel.replace(chr(92), '/'), quote=True)}" target="_blank">開啟／下載 SVG</a>
 · <a href="{html_escape(source_rel, quote=True)}" target="_blank">開啟原始參考圖</a>
 · SVG SHA-256 <code>{meta['svg_sha256']}</code></div>
 <label>條件順序 <select class="condition-order">
   <option value="vector_first"{' selected' if default_order == 'vector_first' else ''}>先改向量、再重畫</option>
   <option value="redraw_first"{' selected' if default_order == 'redraw_first' else ''}>先重畫、再改向量</option>
 </select></label>
 <table><thead><tr><th>任務</th><th>向量接手</th><th>從頭重畫基準</th><th>結果</th><th>備註</th></tr></thead>
 <tbody>{_task_rows(index)}</tbody></table>
 <label>整體接手判定 <select class="handoff"><option value="yes">會接手使用</option><option value="partial">部分可用</option>
   <option value="no">寧願重畫</option></select></label>
 <label>整體備註 <textarea class="case-note" placeholder="最省時間與最難修的地方"></textarea></label>
 <div class="case-summary"></div>
</section>''')

    empty = "<p>output 裡尚無同時具備 SVG 與來源參考圖的結果。</p>"
    page = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>向量清稿 Stage 2 實作計時</title>
<style>
body{{font-family:system-ui,'Microsoft JhengHei',sans-serif;max-width:1400px;margin:20px auto;padding:0 16px;color:#222}}
.intro{{background:#f5f7fb;border:1px solid #d7deea;border-radius:10px;padding:14px 18px;line-height:1.65}}
.case{{border:1px solid #ccc;border-radius:10px;padding:14px;margin:20px 0}}h2{{font-size:18px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ddd;padding:7px;vertical-align:top}}
th{{background:#f4f4f4}}td small{{display:block;color:#666;max-width:300px}}button{{padding:5px 10px}}
input.seconds{{width:75px}}input.task-note{{width:100%;min-width:170px}}select{{padding:4px}}textarea{{display:block;width:100%;min-height:54px}}
.links{{font-size:12px;margin:6px 0 12px;overflow-wrap:anywhere}}code{{font-size:10px}}.case-summary{{font-weight:700;margin-top:8px}}
.global{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}#export{{position:sticky;bottom:12px;background:#1769d3;color:#fff;border:0;border-radius:8px;padding:11px 18px;font-weight:700}}
@media(max-width:900px){{table,thead,tbody,tr,th,td{{display:block}}thead{{display:none}}td{{border-top:0}}tr{{margin-bottom:12px;border-top:1px solid #bbb}}}}
</style></head><body>
<h1>向量清稿驗收（Stage 2：設計師實際編輯計時）</h1>
<div class="intro"><b>目的：</b>檢驗這份 SVG 是否真的節省收尾工時，而不是只看起來像向量。每項任務都要實際操作。<br>
「向量接手」記錄用本工具 SVG 完成任務的秒數；「從頭重畫基準」最好也實際重畫並計時。若只填估算，資料會保留，但<b>不計入 80% 省工驗收</b>。<br>
為降低先做一次造成的熟悉偏差，案例會交錯建議順序。請使用平常工作的向量軟體，開始前先複製一份檔案。</div>
<div class="global"><label>設計師代碼 <input id="designer" placeholder="匿名代碼"></label>
<label>軟體／版本 <input id="editor" placeholder="例如 Illustrator 2026"></label>
<label>經驗年資 <input id="experience" type="number" min="0" step="0.5"> 年</label></div>
{''.join(cards) if cards else empty}
<button id="export">匯出 Stage 2 結果</button>
<script>
let active=null;
function seconds(input){{const value=Number(input.value);return Number.isFinite(value)&&value>=0?value:null;}}
function stopActive(){{if(!active)return;const elapsed=Math.max(1,Math.round((Date.now()-active.started)/1000));
 const old=seconds(active.input)||0;active.input.value=old+elapsed;active.button.textContent='開始';active=null;refresh();}}
document.querySelectorAll('.timer').forEach(button=>button.onclick=()=>{{
 if(active&&active.button===button){{stopActive();return;}}stopActive();
 const row=button.closest('tr'),kind=button.dataset.kind,input=row.querySelector('.seconds.'+kind);
 if(kind==='baseline')row.querySelector('.baseline-kind').value='actual';
 active={{button,input,started:Date.now()}};button.textContent='停止';
}});
function casePayload(card){{let actualVector=0,actualBaseline=0,comparable=0;
 const tasks=[...card.querySelectorAll('tbody tr')].map(row=>{{
  const vector=seconds(row.querySelector('.vector')),baseline=seconds(row.querySelector('.baseline'));
  const baselineKind=row.querySelector('.baseline-kind').value;
  let saving=null;if(baselineKind==='actual'&&vector!=null&&baseline>0){{saving=(baseline-vector)/baseline*100;actualVector+=vector;actualBaseline+=baseline;comparable++;}}
  return {{id:row.dataset.task,status:row.querySelector('.task-status').value,vector_seconds:vector,
   redraw_seconds:baseline,redraw_evidence:baselineKind,time_saving_percent:saving==null?null:+saving.toFixed(2),note:row.querySelector('.task-note').value}};
 }});
 let meta={{}};try{{meta=JSON.parse(card.dataset.meta||'{{}}')}}catch(_e){{}}
 const weighted=comparable&&actualBaseline>0?(actualBaseline-actualVector)/actualBaseline*100:null;
 return {{...meta,condition_order:card.querySelector('.condition-order').value,handoff:card.querySelector('.handoff').value,
  note:card.querySelector('.case-note').value,tasks,summary:{{actual_timed_comparable_tasks:comparable,
  vector_seconds_sum:actualVector,redraw_seconds_sum:actualBaseline,actual_timed_weighted_saving_percent:weighted==null?null:+weighted.toFixed(2),
  observed_ge_80_percent_for_this_session:weighted!=null&&weighted>=80,
  product_claim_validated:false}}}};
}}
function refresh(){{document.querySelectorAll('.case').forEach(card=>{{const s=casePayload(card).summary;
 card.querySelector('.case-summary').textContent=s.actual_timed_comparable_tasks?
  '實際可比較任務 '+s.actual_timed_comparable_tasks+' 項；本次加權省工 '+s.actual_timed_weighted_saving_percent.toFixed(1)+'%':'尚無實際雙條件計時；不能計算省工率';}});}}
document.querySelectorAll('input,select,textarea').forEach(x=>x.addEventListener('change',refresh));refresh();
document.getElementById('export').onclick=()=>{{stopActive();const cases=[...document.querySelectorAll('.case')].map(casePayload);
 const payload={{tool:'ai-vector-cleanroom',version:{json.dumps(tool_version)},generated:new Date().toISOString(),
  kind:'timed-designer-editing-stage2',designer_code:document.getElementById('designer').value,
  editor:document.getElementById('editor').value,experience_years:seconds(document.getElementById('experience')),
  validation_scope:'one session; product-level 80% claim requires multiple designers and representative logos',cases}};
 const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='實作計時結果_Stage2.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}};
</script></body></html>'''
    destination = output / "editing_test_stage2.html"
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["TASKS", "build_editing_test_page"]
