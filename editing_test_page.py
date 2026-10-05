# -*- coding: utf-8 -*-
"""Build a self-contained Stage 2 timed designer handoff test page."""

from __future__ import annotations

import hashlib
from html import escape as html_escape
import json
from pathlib import Path
from typing import Mapping, Sequence


TASKS = (
    ("complete_deliverable", "整張做到可交付（主要比較）", "以同一交付規格完成整張圖，計入檢查、選取、清除碎點、修形、重畫、匯出與返工。"),
    ("global_recolour", "換色", "依案例需要修改指定物件或色彩角色；保留不應改動的色彩。"),
    ("gradient_adjust", "調整漸層", "案例有漸層時，修改指定漸層的方向或色標。"),
    ("contour_cleanup", "輪廓與碎點整理", "修整指定輪廓、清除碎點，保留所需尖角與細節。"),
    ("stroke_adjust", "修改線寬或形狀", "案例適用時修改指定筆畫或幾何形狀，不破壞其他部分。"),
    ("move_local_component", "移動局部元件", "選取指定完整元件並移動或縮放。"),
    ("text_replace", "更換文字", "案例有文字時，依交付規格替換文字或重新排字。"),
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
        checked = " checked" if task_id == "complete_deliverable" else ""
        measurements = []
        for kind, name in (("vector", "向量接手"), ("baseline", "從頭重畫")):
            measurements.append(f'''<td><button class="timer" data-kind="{kind}">開始</button>
 <input class="seconds {kind}" type="number" min="0" step="0.1" inputmode="decimal" aria-label="{name}秒數"> 秒
 <label>時間來源 <select class="{kind}-kind" aria-label="{name}時間來源">
   <option value="none" selected>未提供</option><option value="actual">實際計時</option><option value="estimated">估算</option>
 </select></label>
 <label>結果 <select class="{kind}-status" aria-label="{name}結果">
   <option value="not_started" selected>未開始</option><option value="in_progress">進行中</option>
   <option value="completed">完成且符合交付規格</option><option value="partial">部分完成</option><option value="unable">無法完成</option>
 </select></label></td>''')
        rows.append(f'''<tr data-task="{task_id}">
 <td><label><input class="applicable" type="checkbox"{checked}> 適用</label>
 <b>{html_escape(label)}</b><small>{html_escape(instruction)}</small></td>
 {''.join(measurements)}
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
    case_fingerprints = []
    for index, result in enumerate(results):
        directory_name = str(result.get("dir") or "")
        directory = (output / directory_name).resolve()
        svg_rel = str(result.get("svg") or "")
        svg = (output / Path(svg_rel)).resolve()
        if (output_root not in directory.parents
                or output_root not in svg.parents):
            continue
        source = directory / "source_original.png"
        reference_kind = "original"
        if not source.is_file():
            source = directory / "source_reference.png"
            reference_kind = "processed_reference"
        if not (directory.is_dir() and svg.is_file() and source.is_file()):
            continue
        if source.resolve().parent != directory:
            continue
        source_rel = f"{directory_name}/{source.name}".replace("\\", "/")
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
            "source_reference": source_rel,
            "source_sha256": _file_sha256(source),
            "reference_kind": reference_kind,
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
        case_fingerprints.append({key: meta[key] for key in
                                  ("svg", "svg_sha256", "source_reference", "source_sha256", "reference_kind")})
        source_label = ("開啟未處理原圖" if reference_kind == "original"
                        else "開啟清理後參考圖（非原圖）")
        reference_note = ("" if reference_kind == "original" else
                          '<p class="reference-note">本案例只有清理後參考圖，可能已移除背景或細節；'
                          '計時結果僅代表此參考條件，不能當作未處理原圖的省工證據。</p>')
        default_order = "vector_first" if index % 2 == 0 else "redraw_first"
        cards.append(f'''<section class="case" data-meta="{meta_attr}">
 <h2>案例 {index + 1}：{html_escape(meta['name'])}</h2>
 <div class="links"><a href="{html_escape(svg_rel.replace(chr(92), '/'), quote=True)}" target="_blank">開啟／下載 SVG</a>
 · <a href="{html_escape(source_rel, quote=True)}" target="_blank">{source_label}</a>
 · SVG SHA-256 <code>{meta['svg_sha256']}</code></div>
 {reference_note}
 <label>條件順序 <select class="condition-order">
   <option value="vector_first"{' selected' if default_order == 'vector_first' else ''}>先改向量、再重畫</option>
   <option value="redraw_first"{' selected' if default_order == 'redraw_first' else ''}>先重畫、再改向量</option>
 </select></label>
 <label>共同交付規格 <textarea class="delivery-criteria" placeholder="例如：尺寸、必須保留的細節、允許的外觀偏差、物件可編輯要求；兩邊使用同一份規格"></textarea></label>
 <table><thead><tr><th>適用任務</th><th>向量接手</th><th>從頭重畫基準</th><th>備註</th></tr></thead>
 <tbody>{_task_rows(index)}</tbody></table>
 <label>整體接手判定 <select class="handoff"><option value="not_tested" selected>尚未判定</option><option value="yes">會接手使用</option><option value="partial">部分可用</option>
   <option value="no">寧願重畫</option></select></label>
 <label>整體備註 <textarea class="case-note" placeholder="最省時間與最難修的地方"></textarea></label>
 <div class="case-summary"></div>
</section>''')

    empty = "<p>output 裡尚無同時具備 SVG 與來源參考圖的結果。</p>"
    metrics_script = Path(__file__).with_name("editing_metrics.js").read_text(encoding="utf-8")
    draft_identity = hashlib.sha256(json.dumps({
        "cases": case_fingerprints, "tasks": [task[0] for task in TASKS],
        "tool_version": tool_version, "draft_schema": 1,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    page = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>向量清稿 Stage 2 實作計時</title>
<style>
body{{font-family:system-ui,'Microsoft JhengHei',sans-serif;max-width:1400px;margin:20px auto;padding:0 16px;color:#222}}
.intro{{background:#f5f7fb;border:1px solid #d7deea;border-radius:10px;padding:14px 18px;line-height:1.65}}
.case{{border:1px solid #ccc;border-radius:10px;padding:14px;margin:20px 0}}h2{{font-size:18px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ddd;padding:7px;vertical-align:top}}
th{{background:#f4f4f4}}td small{{display:block;color:#666;max-width:300px}}button{{padding:5px 10px}}
input.seconds{{width:75px}}input.task-note{{width:100%;min-width:170px}}select{{padding:4px}}textarea{{display:block;width:100%;min-height:54px}}
td label{{display:block;margin:5px 0}}tr[data-task="complete_deliverable"]{{background:#f0f6ff}}td b{{display:block}}
.links{{font-size:12px;margin:6px 0 12px;overflow-wrap:anywhere}}code{{font-size:10px}}.case-summary{{font-weight:700;margin-top:8px}}
.global{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}#export{{position:sticky;bottom:12px;background:#1769d3;color:#fff;border:0;border-radius:8px;padding:11px 18px;font-weight:700}}
@media(max-width:900px){{table,thead,tbody,tr,th,td{{display:block}}thead{{display:none}}td{{border-top:0}}tr{{margin-bottom:12px;border-top:1px solid #bbb}}}}
</style></head><body>
<h1>向量清稿驗收（Stage 2：設計師實際編輯計時）</h1>
<div class="intro"><b>主要比較：</b>兩邊把整張圖做到同一交付規格，記錄完整人工時間（含檢查、刪碎點與返工）；先寫下共同規格。<br>
雙邊都必須明確標記完成，且各有大於 0 秒的實際計時，才計算省工率。空白、0、估算、未完成與失敗會保留紀錄，但不當作省工成功。<br>
其他任務按案例勾選，請用獨立副本測試；細項只供診斷，不與整張時間相加，也不代替整張省工率。計時停止後請自行確認完成狀態。<br>
為降低先做一次造成的熟悉偏差，案例會交錯建議順序。請使用平常工作的向量軟體，開始前先複製一份檔案。</div>
<div class="global"><label>設計師代碼 <input id="designer" placeholder="匿名代碼"></label>
<label>軟體／版本 <input id="editor" placeholder="例如 Illustrator 2026"></label>
<label>經驗年資 <input id="experience" type="number" min="0" step="0.5"> 年</label></div>
<p id="draft-status" role="status">開始輸入後會自動儲存瀏覽器草稿；重新開啟一律停表，不計離頁時間。</p>
{''.join(cards) if cards else empty}
<button id="export">匯出 Stage 2 結果</button>
<script>
{metrics_script}
let active=null,dirty=false;
const DRAFT_IDENTITY={json.dumps(draft_identity)};
const DRAFT_KEY='aivc.stage2.draft.v1.'+DRAFT_IDENTITY;
const draftFields=[...document.querySelectorAll('input,select,textarea')];
function draftStatus(text){{document.getElementById('draft-status').textContent=text;}}
function saveDraft(){{
 if(!dirty)return;
 const fields=draftFields.map(field=>({{type:field.type||field.tagName,value:field.value,checked:field.type==='checkbox'?field.checked:null}}));
 if(active){{const index=draftFields.indexOf(active.input);if(index>=0){{
  const elapsed=Math.max(0,(performance.now()-active.started)/1000);
  fields[index].value=String(Math.round(((seconds(active.input)||0)+elapsed)*10)/10);
 }}}}
 try{{localStorage.setItem(DRAFT_KEY,JSON.stringify({{schema:1,identity:DRAFT_IDENTITY,fields}}));
  draftStatus('未匯出：草稿已儲存於此瀏覽器；重開後保持停表。');
 }}catch(_e){{draftStatus('瀏覽器無法儲存草稿，請在離頁前匯出結果。');}}
}}
function markDirty(){{dirty=true;saveDraft();}}
function restoreDraft(){{
 let draft;try{{draft=JSON.parse(localStorage.getItem(DRAFT_KEY)||'null');}}catch(_e){{return;}}
 if(!draft||draft.schema!==1||draft.identity!==DRAFT_IDENTITY||!Array.isArray(draft.fields)||draft.fields.length!==draftFields.length)return;
 if(draft.fields.some((saved,index)=>!saved||saved.type!==(draftFields[index].type||draftFields[index].tagName)||typeof saved.value!=='string'||
   (draftFields[index].type==='checkbox'&&typeof saved.checked!=='boolean')))return;
 draft.fields.forEach((saved,index)=>{{const field=draftFields[index];field.value=saved.value;if(field.type==='checkbox')field.checked=saved.checked;}});
 // Only accumulated seconds are restored. No wall-clock timestamp can resume
 // a timer or add time spent away from the page.
 active=null;dirty=true;draftStatus('已恢復未匯出草稿；全部計時器保持停止，離頁時間未計入。');
}}
window.addEventListener('beforeunload',event=>{{if(!dirty)return;saveDraft();event.preventDefault();event.returnValue='';}});
window.addEventListener('pagehide',()=>{{stopActive();saveDraft();}});
setInterval(()=>{{if(active&&dirty)saveDraft();}},5000);
function seconds(input){{return EditingMetrics.seconds(input.value);}}
function stopActive(){{if(!active)return;const elapsed=Math.max(0,(performance.now()-active.started)/1000);
 const old=seconds(active.input)||0;active.input.value=String(Math.round((old+elapsed)*10)/10);active.button.textContent='開始';active=null;markDirty();refresh();}}
document.querySelectorAll('.timer').forEach(button=>button.onclick=()=>{{
 if(active&&active.button===button){{stopActive();return;}}stopActive();
 const row=button.closest('tr'),kind=button.dataset.kind,input=row.querySelector('.seconds.'+kind);
 row.querySelector('.applicable').checked=true;
 const evidence=row.querySelector('.'+kind+'-kind');
 if(evidence.value!=='actual')input.value='';
 evidence.value='actual';row.querySelector('.'+kind+'-status').value='in_progress';
 active={{button,input,started:performance.now()}};button.textContent='停止';markDirty();refresh();
}});
function casePayload(card){{
 const rawTasks=[...card.querySelectorAll('tbody tr')].map(row=>({{
  id:row.dataset.task,applicable:row.querySelector('.applicable').checked,
  vector_seconds:row.querySelector('.vector').value,redraw_seconds:row.querySelector('.baseline').value,
  vector_evidence:row.querySelector('.vector-kind').value,redraw_evidence:row.querySelector('.baseline-kind').value,
  vector_status:row.querySelector('.vector-status').value,redraw_status:row.querySelector('.baseline-status').value,
  note:row.querySelector('.task-note').value
 }}));
 const measured=EditingMetrics.summarize(rawTasks);
 let meta={{}};try{{meta=JSON.parse(card.dataset.meta||'{{}}')}}catch(_e){{}}
 return {{...meta,condition_order:card.querySelector('.condition-order').value,handoff:card.querySelector('.handoff').value,
  delivery_criteria:card.querySelector('.delivery-criteria').value,
  note:card.querySelector('.case-note').value,...measured}};
}}
function refresh(){{document.querySelectorAll('.case').forEach(card=>{{const s=casePayload(card).summary;
 const primary=s.primary_comparison_eligible?'整張交付省工 '+s.primary_time_saving_percent.toFixed(1)+'%':'整張交付尚無有效雙邊完稿計時；不能計算省工率';
 card.querySelector('.case-summary').textContent=primary+'；細項可比較 '+s.diagnostic_comparable_tasks+' 項；已嘗試 '+s.attempted_tasks+' 項（無法完成 '+s.unable_tasks+' 項、尚不可比較 '+s.uncomparable_attempts+' 項）。';}});}}
draftFields.forEach(field=>{{for(const event of ['input','change'])field.addEventListener(event,()=>{{markDirty();refresh();}});}});
restoreDraft();refresh();
document.getElementById('export').onclick=()=>{{stopActive();const cases=[...document.querySelectorAll('.case')].map(casePayload);
 const payload={{tool:'ai-vector-cleanroom',version:{json.dumps(tool_version)},generated:new Date().toISOString(),
  kind:'timed-designer-editing-stage2',metrics_schema:'ai-vector-cleanroom.editing-metrics/v2',designer_code:document.getElementById('designer').value,
  editor:document.getElementById('editor').value,experience_years:seconds(document.getElementById('experience')),
  validation_scope:'one session; product-level 80% claim requires multiple designers and representative logos',cases}};
 const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='實作計時結果_Stage2.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 dirty=false;try{{localStorage.removeItem(DRAFT_KEY);}}catch(_e){{}}
 draftStatus('已要求下載匯出結果；請確認檔案已保存。繼續修改會建立新草稿。');
}};
</script></body></html>'''
    destination = output / "editing_test_stage2.html"
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["TASKS", "build_editing_test_page"]
