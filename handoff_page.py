# -*- coding: utf-8 -*-
"""Self-contained, local designer review UI for an Illustrator handoff.

The page changes review decisions only; the server owns export and all geometry.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET


_ACTIONS = {"keep", "review", "redraw"}
_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"


def _safe_json(value: object) -> str:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def _numbers(value: object, *, positive_size: bool = False) -> list[float] | None:
    try:
        parts = re.split(r"[\s,]+", value.strip()) if isinstance(value, str) else list(value)
        result = [float(n) for n in parts]
        if len(result) != 4 or not all(math.isfinite(n) for n in result):
            return None
        if positive_size and (result[2] <= 0 or result[3] <= 0):
            return None
        if not positive_size and (result[2] < 0 or result[3] < 0):
            return None
        return result
    except (ValueError, TypeError, OverflowError):
        return None


def _count(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _fraction(value: object) -> float:
    try:
        number = float(value or 0)
        return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _display_svg(svg_text: str) -> tuple[str, list[float]]:
    """Remove active/external content in the preview, without rewriting geometry.

    Namespace preview IDs so SVG IDs cannot shadow the surrounding HTML UI.
    Only the preview copy is affected; export receives the original fingerprint.
    """
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", svg_text, flags=re.I):
        raise ValueError("SVG 預覽不支援 DOCTYPE 或 ENTITY")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError("無法讀取 SVG 預覽") from exc
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("預覽來源必須為 SVG")
    view_box = _numbers(root.get("viewBox"), positive_size=True)
    if view_box is None:
        def dimension(name: str) -> float:
            match = re.match(r"^\s*(\d+(?:\.\d+)?)", root.get(name, "100"))
            return max(float(match.group(1)), 1) if match else 100
        view_box = [0, 0, dimension("width"), dimension("height")]
    prohibited = {"script", "foreignObject", "iframe", "object", "embed", "audio",
                  "video", "animate", "animateMotion", "animateTransform", "set",
                  "discard", "handler", "listener", "style", "a"}
    # A link may contain useful geometry: unwrap it instead of discarding that geometry.
    def clean_children(parent: ET.Element) -> None:
        for child in list(parent):
            local = child.tag.split("}")[-1] if isinstance(child.tag, str) else ""
            clean_children(child)
            namespace = child.tag.split("}")[0][1:] if "}" in child.tag else ""
            if namespace not in {"", _SVG_NS}:
                parent.remove(child)
                continue
            if local in prohibited:
                position = list(parent).index(child)
                if local == "a":
                    for subchild in list(child):
                        parent.insert(position, subchild)
                        position += 1
                parent.remove(child)
    clean_children(root)
    id_map = {node.attrib["id"]: f"handoff-shape-{i}"
              for i, node in enumerate(root.iter()) if "id" in node.attrib}
    for node in root.iter():
        original_id = node.get("id")
        for key, value in list(node.attrib.items()):
            local = key.split("}")[-1].lower()
            if local.startswith("on") or local in {"src", "tabindex", "autofocus", "base"}:
                del node.attrib[key]
            elif local == "href":
                if value.startswith("#") and value[1:] in id_map:
                    node.set(key, "#" + id_map[value[1:]])
                else:
                    del node.attrib[key]
            elif "url(" in value.lower():
                urls = re.findall(r"url\(\s*['\"]?([^)'\"]+)", value, flags=re.I)
                if any(not url.startswith("#") or url[1:] not in id_map for url in urls):
                    del node.attrib[key]
                else:
                    node.set(key, re.sub(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)",
                                        lambda m: "url(#" + id_map[m.group(1)] + ")", value))
            elif local == "style" and re.search(r"[@\\]|expression\s*\(|javascript:", value, re.I):
                del node.attrib[key]
        if original_id is not None:
            node.set("id", id_map[original_id])
            node.set("data-source-id", original_id)
    root.set("viewBox", " ".join(str(n) for n in view_box))
    root.set("width", "100%")
    root.set("height", "100%")
    root.set("preserveAspectRatio", "xMidYMid meet")
    root.set("aria-label", "向量圖稿預覽")
    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)
    return ET.tostring(root, encoding="unicode"), view_box


def build_handoff_page(manifest: Mapping[str, object], svg_text: str,
                       source_data_url: str, *, result_dir: str | Path,
                       token: str, saved_decisions: Mapping[str, str] | None = None) -> str:
    """Return a standalone review page for ``aivc.designer-handoff/v1`` data."""
    svg, fallback_view_box = _display_svg(svg_text)
    objects = []
    seen = set()
    for raw in manifest.get("objects", []):
        if not isinstance(raw, Mapping):
            continue
        obj_id = str(raw.get("id", ""))
        if not obj_id or obj_id in seen:
            continue
        seen.add(obj_id)
        members = raw.get("member_ids", [])
        if not isinstance(members, (list, tuple)):
            members = []
        reasons = raw.get("reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        elif not isinstance(reasons, (list, tuple)):
            reasons = []
        action = raw.get("suggested_action", "review")
        objects.append({"id": obj_id, "label": str(raw.get("label") or obj_id),
                        "member_ids": [str(item) for item in members],
                        "anchor_count": _count(raw.get("anchor_count")),
                        "path_count": _count(raw.get("path_count")),
                        "bbox": _numbers(raw.get("bbox")),
                        "source_defect_count": _count(raw.get("source_defect_count")),
                        "source_defect_fraction": _fraction(raw.get("source_defect_fraction")),
                        "curve_review_count": _count(raw.get("curve_review_count")),
                        "source_audit_status": str(raw.get("source_audit_status", "unavailable")),
                        "suggested_action": action if action in _ACTIONS else "review",
                        "reasons": [str(item) for item in reasons]})
    proposed = manifest.get("default_decisions")
    proposed = proposed if isinstance(proposed, Mapping) else {}
    defaults = {obj["id"]: proposed.get(obj["id"]) if proposed.get(obj["id"]) in _ACTIONS
                else obj["suggested_action"] for obj in objects}
    # None means there is no server save; an empty server save remains authoritative.
    saved = ({key: value for key, value in saved_decisions.items()
              if key in defaults and value in _ACTIONS}
             if isinstance(saved_decisions, Mapping) else None)
    source = str(source_data_url or "")
    if not re.fullmatch(r"data:image/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\s]+", source):
        source = ""
    source_audit = manifest.get('source_object_audit')
    source_audit = source_audit if isinstance(source_audit, Mapping) else {}
    scene_concerns = []
    for raw in manifest.get('scene_source_concerns', []) or []:
        if not isinstance(raw, Mapping):
            continue
        box = _numbers([raw.get(k) for k in ('x', 'y', 'w', 'h')], positive_size=True)
        if box is not None:
            scene_concerns.append({'label': str(raw.get('label') or '原圖局部結構需要檢查'), 'bbox': box})
    payload = {
        "schema": "aivc.designer-handoff/v1",
        "svg_sha256": str(manifest.get("svg_sha256", "")),
        "result": str(result_dir), "token": str(token), "objects": objects,
        "view_box": _numbers(manifest.get("view_box"), positive_size=True) or fallback_view_box,
        "default_decisions": defaults, "saved_decisions": saved,
        "saved_revision": (manifest.get("saved_revision")
                           if isinstance(manifest.get("saved_revision"), str) else None),
        "source_data_url": source,
        "auto_prepare": (manifest.get("auto_prepare")
                         if isinstance(manifest.get("auto_prepare"), dict) else None),
        "reference_kind": ("original" if manifest.get("reference_kind") == "original"
                           else "processed_reference"),
        "source_audit_status": str(source_audit.get("status", "unavailable")),
        "scene_source_concerns": scene_concerns,
    }
    substitutions = {"PAYLOAD": _safe_json(payload), "SVG": svg, "STATE": _STATE_JS}
    return re.sub(r"@@(PAYLOAD|SVG|STATE)@@", lambda match: substitutions[match.group(1)], _HTML)


_STATE_JS = r"""
class HandoffState {
  constructor(objects, defaults, saved) {
    this.objects = objects;
    this.allowed = new Set(['keep', 'review', 'redraw']);
    this.defaults = Object.fromEntries(objects.map(o => [o.id,
      this.allowed.has(defaults[o.id]) ? defaults[o.id] : 'review']));
    this.decisions = {...this.defaults};
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
      for (const o of objects) if (this.allowed.has(saved[o.id])) this.decisions[o.id] = saved[o.id];
    }
    this.history = []; this.future = []; this.revision = 0;
  }
  apply(ids, action) {
    if (!this.allowed.has(action)) return false;
    const changes = [];
    for (const id of new Set(ids)) {
      if (Object.hasOwn(this.decisions, id) && this.decisions[id] !== action)
        changes.push([id, this.decisions[id], action]);
    }
    if (!changes.length) return false;
    for (const [id, , value] of changes) this.decisions[id] = value;
    this.history.push(changes); this.future = []; this.revision++; return true;
  }
  undo() {
    const changes = this.history.pop(); if (!changes) return false;
    for (const [id, value] of changes) this.decisions[id] = value;
    this.future.push(changes); this.revision++; return true;
  }
  redo() {
    const changes = this.future.pop(); if (!changes) return false;
    for (const [id, , value] of changes) this.decisions[id] = value;
    this.history.push(changes); this.revision++; return true;
  }
  restore(decisions) {
    const changes=[];
    for(const id of Object.keys(this.decisions)) {
      if(Object.hasOwn(decisions,id)&&this.allowed.has(decisions[id])&&this.decisions[id]!==decisions[id])
        changes.push([id,this.decisions[id],decisions[id]]);
    }
    if(!changes.length)return false;
    for(const [id,,value] of changes)this.decisions[id]=value;
    this.history.push(changes);this.future=[];this.revision++;return true;
  }
  counts() {
    const counts = {keep: 0, review: 0, redraw: 0};
    for (const action of Object.values(this.decisions)) counts[action]++;
    return counts;
  }
}
function makeHandoffDraft(svgSha,baseRevision,decisions){
  return {svg_sha256:svgSha,baseRevision,decisions:{...decisions}};
}
function restorableHandoffDraft(data,draft){
  if(!draft||draft.svg_sha256!==data.svg_sha256||!draft.decisions||typeof draft.decisions!=='object'||Array.isArray(draft.decisions))return false;
  const ids=data.objects.map(object=>object.id);
  return Object.keys(draft.decisions).length===ids.length&&ids.every(id=>Object.hasOwn(draft.decisions,id)&&['keep','review','redraw'].includes(draft.decisions[id]));
}
function preserveHandoffDraft(backups,draft){
  if(!draft)return;
  const text=JSON.stringify(draft);
  if(!backups.some(item=>JSON.stringify(item)===text))backups.push(draft);
}
function resolveHandoffDraft(data,local){
  const server=new HandoffState(data.objects,data.default_decisions,data.saved_decisions).decisions;
  const result={initial:server,serverSnapshot:data.saved_decisions===null?null:JSON.stringify(server),recovered:false,
                backups:[],pending:null};
  if(!local||typeof local!=='object'||Array.isArray(local))return result;
  if(Array.isArray(local.backups))result.backups=local.backups.slice();
  if(local.pendingDraft&&typeof local.pendingDraft==='object')result.pending=local.pendingDraft;
  const candidate=Object.hasOwn(local,'baseRevision')?
    makeHandoffDraft(local.svg_sha256,local.baseRevision,local.decisions):
    {svg_sha256:local.svg_sha256,decisions:local.decisions,legacy_revision_unknown:true};
  if(restorableHandoffDraft(data,candidate)&&Object.hasOwn(candidate,'baseRevision')&&candidate.baseRevision===data.saved_revision){
    result.initial={...candidate.decisions};
    result.recovered=JSON.stringify(result.initial)!==JSON.stringify(server)||data.saved_decisions===null;
  }else if(candidate.decisions){
    preserveHandoffDraft(result.backups,candidate);
    result.pending=candidate;
  }
  if(local.unreadable_raw!==undefined)preserveHandoffDraft(result.backups,{unreadable_raw:local.unreadable_raw});
  return result;
}
function mergeOtherHandoffDraft(data,latest,writerId,decisions,backups,pending){
  if(!latest||typeof latest!=='object'||Array.isArray(latest))return pending;
  if(Array.isArray(latest.backups))for(const entry of latest.backups)preserveHandoffDraft(backups,entry);
  if(latest.pendingDraft)preserveHandoffDraft(backups,latest.pendingDraft);
  if(latest.unreadable_raw!==undefined)preserveHandoffDraft(backups,{unreadable_raw:latest.unreadable_raw});
  if(latest.writerId!==writerId&&latest.decisions&&JSON.stringify(latest.decisions)!==JSON.stringify(decisions)){
    const other=Object.hasOwn(latest,'baseRevision')?makeHandoffDraft(latest.svg_sha256,latest.baseRevision,latest.decisions):
      {svg_sha256:latest.svg_sha256,decisions:latest.decisions,legacy_revision_unknown:true};
    preserveHandoffDraft(backups,other);return pending||other;
  }
  return pending;
}
function handoffBoundsContained(box,start,end){
  if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite)||box[2]<0||box[3]<0)return false;
  const x=Math.min(start[0],end[0]),y=Math.min(start[1],end[1]);
  const right=Math.max(start[0],end[0]),bottom=Math.max(start[1],end[1]);
  return box[0]>=x&&box[1]>=y&&box[0]+box[2]<=right&&box[1]+box[3]<=bottom;
}
"""


_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer"><title>清稿接手台 · Illustrator</title>
<style>
:root{--ink:#24302e;--muted:#6f7974;--line:#e0e6df;--paper:#fffefa;--bg:#f1f4ee;--green:#30654a;--green-light:#e8f1e6;--amber:#925e1e;--amber-light:#faf0da;--red:#9b4e3e;--red-light:#f8e9e2;--radius:15px}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;color:var(--ink);background:var(--bg);font-family:"Segoe UI","Microsoft JhengHei",sans-serif;font-size:14px;line-height:1.55}button,input,select{font:inherit}button,select{cursor:pointer}button{border:1px solid var(--line);border-radius:9px;background:var(--paper);color:var(--ink);padding:9px 13px;transition:background .15s,border-color .15s}button:hover:not(:disabled){background:#eff3eb;border-color:#b8c9b7}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #80a78c;outline-offset:2px}button:disabled{opacity:.42;cursor:default}button.primary{background:var(--green);color:white;border-color:var(--green);font-weight:650}button.primary:hover:not(:disabled){background:#254f3a}.small{font-size:12px}.muted{color:var(--muted)}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:23px 30px 20px;background:var(--paper);border-bottom:1px solid var(--line)}.brand{display:flex;gap:14px;align-items:center}.mark{width:42px;height:42px;background:var(--green);color:white;border-radius:13px;display:grid;place-items:center;font-size:23px;font-weight:300}.eyebrow{font-size:10px;letter-spacing:1.7px;font-weight:700;color:var(--green)}h1{font-size:23px;letter-spacing:.4px;margin:1px 0 0;font-weight:650}.header-actions{display:flex;align-items:center;gap:9px}.save-state{font-size:12px;color:var(--muted);max-width:170px;line-height:1.4}.shell{padding:21px 24px 24px;max-width:1920px;margin:0 auto}.intro{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:17px}.intro p{margin:0;font-size:13px}.stats{display:flex;gap:7px;flex-shrink:0}.stat{border-radius:30px;padding:6px 12px;background:#e5eade;color:#52614f;white-space:nowrap}.stat b{margin-left:7px;font-variant-numeric:tabular-nums}.stat.keep{background:var(--green-light);color:var(--green)}.stat.review{background:var(--amber-light);color:var(--amber)}.stat.redraw{background:var(--red-light);color:var(--red)}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:17px;align-items:start}.canvas-card,.inspector{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}.canvas-bar{padding:12px 15px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.segmented{padding:3px;border-radius:10px;background:#edf1e9;display:flex;gap:3px}.segmented button{padding:6px 12px;border:0;background:transparent;font-size:12px;border-radius:7px}.segmented button.active{background:white;box-shadow:0 1px 4px #24302e12;color:var(--green);font-weight:650}.canvas-tools{display:flex;align-items:center;gap:6px}.canvas-tools button{padding:5px 9px;font-size:13px}.zoom-text{min-width:47px;text-align:center;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}.canvas-stage{display:grid;grid-template-columns:1fr 1fr;height:clamp(380px,58vh,740px)}.canvas-stage.single{grid-template-columns:1fr}.canvas-stage.single .source-pane{display:none}.pane{min-width:0;position:relative;overflow:hidden;background-color:#fbfcf9;background-image:linear-gradient(45deg,#edf0e9 25%,transparent 25%),linear-gradient(-45deg,#edf0e9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#edf0e9 75%),linear-gradient(-45deg,transparent 75%,#edf0e9 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}.source-pane{border-right:1px solid var(--line)}.pane-label{position:absolute;z-index:5;top:15px;left:15px;border:1px solid #dfe6da;background:#fffefaeb;border-radius:7px;padding:4px 10px;font-size:11px;color:#61725e;pointer-events:none}.viewport{position:absolute;inset:0;overflow:hidden;touch-action:none;cursor:crosshair}.viewport.pan-mode{cursor:grab}.viewport.dragging{cursor:grabbing}.artboard{position:absolute;transform-origin:0 0}.artboard>svg,.artboard>img{position:absolute;width:100%;height:100%;inset:0;display:block;object-fit:contain;user-select:none;-webkit-user-drag:none}.artboard>.overlay{pointer-events:none;overflow:visible;z-index:2}.overlay .selection-box{fill:none;stroke:#34754e;stroke-width:1.6;vector-effect:non-scaling-stroke;stroke-dasharray:4 3}.overlay .hover-box{fill:none;stroke:#85a88d;stroke-width:1;vector-effect:non-scaling-stroke}.marquee{position:absolute;border:1.5px solid var(--green);background:#43815820;pointer-events:none;z-index:10;display:none}.canvas-empty{position:absolute;inset:50% 20px auto;text-align:center;transform:translateY(-50%);font-size:13px;color:var(--muted);pointer-events:none}.canvas-footer{min-height:48px;border-top:1px solid var(--line);padding:12px 16px;display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:11px}.canvas-footer strong{color:var(--green);font-weight:500}.legend-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:5px}.review-note{padding:14px 2px;font-size:12px;color:var(--muted);line-height:1.75}.review-note b{color:#50634f;font-weight:600}.workflow-note{margin-top:8px;padding:13px 15px;background:#e9eee4;border:1px solid #dce4d5;border-radius:10px;font-size:12px;color:#50634f}
.inspector-head{padding:17px 17px 13px;border-bottom:1px solid var(--line)}.inspector-title{display:flex;justify-content:space-between;align-items:baseline}h2{margin:0;font-size:15px;font-weight:650}.inspector-title span{font-size:11px;color:var(--muted)}.search{width:100%;border:1px solid var(--line);border-radius:8px;background:#f6f8f2;padding:8px 10px;margin:12px 0 9px;font-size:12px}.filter-row{display:flex;gap:7px}.filter-row select{min-width:0;flex:1;width:50%;border:1px solid var(--line);border-radius:7px;background:white;padding:6px;font-size:11px;color:#5b6b58}.selection-bar{padding:10px 17px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);font-size:11px}.text-button{background:none!important;border:0;padding:0 4px;color:var(--green);font-size:11px}.objects{max-height:clamp(210px,35vh,450px);overflow-y:auto;overscroll-behavior:contain}.object-row{display:grid;grid-template-columns:18px 1fr auto;gap:9px;align-items:center;padding:11px 16px;border-bottom:1px solid #edf0e9;cursor:pointer;position:relative}.object-row:hover{background:#f5f7f0}.object-row.selected{background:#eef4e8;box-shadow:inset 3px 0 var(--green)}.object-row input{accent-color:var(--green);margin:0;width:14px;height:14px;cursor:pointer}.object-label{font-size:12px;font-weight:600;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}.object-meta{font-size:10px;color:var(--muted);margin-top:2px}.badge{font-size:10px;padding:3px 7px;border-radius:6px;white-space:nowrap}.badge.keep{background:var(--green-light);color:var(--green)}.badge.review{background:var(--amber-light);color:var(--amber)}.badge.redraw{background:var(--red-light);color:var(--red)}.list-empty{text-align:center;padding:28px 15px;font-size:12px;color:var(--muted)}.decision-panel{padding:15px 17px 17px;border-top:1px solid var(--line)}.decision-title{font-size:12px;font-weight:650;display:flex;justify-content:space-between;align-items:center}.undo-group{display:flex;gap:5px}.undo-group button{padding:2px 6px;font-size:12px}.decisions{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:11px}.decision-button{font-size:11px;padding:10px 3px;display:flex;flex-direction:column;align-items:center;gap:3px;background:white}.decision-button .symbol{font-size:16px;line-height:1.2}.decision-button[data-action=keep]{color:var(--green)}.decision-button[data-action=review]{color:var(--amber)}.decision-button[data-action=redraw]{color:var(--red)}.decision-button.active[data-action=keep]{background:var(--green-light);border-color:#94b29b}.decision-button.active[data-action=review]{background:var(--amber-light);border-color:#d0b989}.decision-button.active[data-action=redraw]{background:var(--red-light);border-color:#d0a89b}.selection-detail{font-size:11px;color:var(--muted);margin-top:11px;line-height:1.7;min-height:36px}.selection-detail ul{padding-left:16px;margin:5px 0 0}.selection-detail p{margin:0 0 4px}.selection-detail .focus-button{margin-top:6px;color:var(--green);font-size:11px;padding:3px 7px}.status-message{display:none;margin-top:13px;padding:12px 15px;background:var(--green-light);border:1px solid #cbdcc9;border-radius:10px;font-size:12px;overflow-wrap:anywhere}.status-message.show{display:block}.status-message.error{background:var(--red-light);border-color:#e5c9bd;color:var(--red)}.status-message a{color:var(--green);margin-right:13px;display:inline-block;margin-top:7px}.pending-export{font-size:11px;color:var(--muted);margin:7px 0 0;max-width:380px;text-align:right}.export-block{display:flex;flex-direction:column;align-items:flex-end}.key-hint{font-family:inherit;border:1px solid #d9e1d3;border-radius:4px;padding:0 4px;font-size:10px;background:white}.history-hint{font-size:10px;margin-top:9px;color:var(--muted)}
.refine-panel{border-top:1px solid var(--line);margin-top:13px;padding-top:12px}.refine-panel label{font-size:11px;color:var(--muted);display:block;margin-bottom:6px}.refine-controls{display:flex;gap:6px}.refine-controls select{width:104px;border:1px solid var(--line);border-radius:7px;padding:5px;font-size:11px;color:var(--ink);background:white}.refine-controls button{flex:1;min-width:0;padding:7px 4px;font-size:11px}.refine-help{font-size:10px;line-height:1.7;color:var(--muted);margin:7px 0 0}
@media(min-width:1600px){.workspace{grid-template-columns:minmax(0,1fr) 380px}.shell{padding:26px 32px}.canvas-stage{height:65vh}.objects{max-height:43vh}}
@media(max-width:1080px){header{padding:18px 20px}.shell{padding:16px}.workspace{grid-template-columns:minmax(0,1fr) 310px;gap:12px}.intro{align-items:flex-start}.intro p{max-width:390px}.stats{gap:4px}.stat{padding:5px 9px}.canvas-footer{flex-direction:column;gap:4px}.header-actions{gap:7px}.save-state{max-width:110px;font-size:11px}.canvas-stage{height:480px}}
@media(max-width:800px){header{align-items:flex-start;flex-wrap:wrap;gap:13px}.header-actions{width:100%;justify-content:flex-end}.brand{flex:1}.export-block{margin-left:auto}.pending-export{max-width:none;text-align:left}.intro{flex-direction:column;gap:10px}.intro p{max-width:none}.workspace{grid-template-columns:1fr}.canvas-stage{height:440px}.objects{max-height:280px}.canvas-footer{flex-direction:row;flex-wrap:wrap}.save-state{margin-right:auto;max-width:140px}.canvas-tools{margin-left:auto}.shell{padding:12px}.inspector{margin-top:0}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<header>
 <div class="brand"><div class="mark" aria-hidden="true">⌁</div><div><div class="eyebrow">VECTOR HANDOFF / ILLUSTRATOR</div><h1>清稿接手台</h1></div></div>
 <div class="header-actions"><span id="saveState" class="save-state" role="status">尚未儲存至工作台</span><button id="saveButton">儲存判斷</button><div class="export-block"><button id="exportButton" class="primary">匯出 Illustrator 接手包 <span aria-hidden="true">↗</span></button></div></div>
</header>
<main class="shell">
 <div id="statusMessage" class="status-message" role="status" aria-live="polite" style="margin:0 0 14px"></div>
 <div class="intro"><p>先讓程式整理，再檢查需要你判斷的地方。<br><span class="muted">簡化會另存新版本；已採用的物件會保留。</span></p><div class="stats" aria-label="目前處理狀態"><span class="stat keep">採用<b id="keepCount">0</b></span><span class="stat review">待確認<b id="reviewCount">0</b></span><span class="stat redraw">交人工<b id="redrawCount">0</b></span></div></div>
 <div class="workflow-note"><button id="prepareButton" class="primary">自動整理整張圖</button> <span id="prepareProgress">一次檢查整張圖，減少多餘節點，保留孔洞與漸層；無法安全簡化的地方保持原狀。</span></div>
 <div id="prepareSummary" class="workflow-note" hidden><b id="prepareSummaryText"></b><p>節點較少不代表設計已完成。可直接在圖上確認，或打開接手包繼續修改。</p><details><summary>查看本次處理的區域</summary><div id="prepareUnits"></div></details></div>
 <div id="draftNotice" class="workflow-note" hidden><p id="draftNoticeText"></p><button id="recoverDraftButton">取回本機草稿，重新核對</button> <button id="downloadDraftButton">下載保留草稿 JSON</button></div>
 <div class="workspace">
  <section aria-label="圖稿比對">
   <div class="canvas-card"><div class="canvas-bar"><div class="segmented" role="group" aria-label="預覽模式"><button data-view="compare" class="active" aria-pressed="true">雙欄比對</button><button data-view="vector" aria-pressed="false">向量細看</button><button data-view="adopted" aria-pressed="false">只看已採用</button></div><div class="canvas-tools"><button id="panButton" aria-pressed="false" title="平移模式；也可按住空白鍵拖曳">✥ 平移</button><button id="zoomOut" title="縮小" aria-label="縮小">−</button><span id="zoomText" class="zoom-text">100%</span><button id="zoomIn" title="放大" aria-label="放大">＋</button><button id="fitButton" title="顯示完整圖稿">全圖</button></div></div>
    <div id="canvasStage" class="canvas-stage">
     <div class="pane source-pane"><span id="sourceLabel" class="pane-label">參考圖片</span><div id="sourceViewport" class="viewport"><div id="sourceBoard" class="artboard"><img id="sourceImage" alt="參考圖片" draggable="false"><svg class="overlay" aria-hidden="true"></svg></div><div class="marquee"></div><div id="sourceMissing" class="canvas-empty" hidden>這份結果未提供參考圖片預覽</div></div></div>
     <div class="pane vector-pane"><span id="vectorLabel" class="pane-label">向量結果 · 點選物件</span><div id="vectorViewport" class="viewport"><div id="vectorBoard" class="artboard">@@SVG@@<svg class="overlay" aria-hidden="true"></svg></div><div class="marquee"></div><div id="adoptedEmpty" class="canvas-empty" hidden>還沒有採用的物件<br>切回「雙欄比對」即可選取與採用物件</div></div></div>
    </div><div class="canvas-footer"><span><span class="legend-dot"></span><strong id="selectedCaption">尚未選取物件</strong></span><span>完整框住才選取 · <span class="key-hint">Shift</span> 多選 · 滾輪縮放 · 空白鍵拖曳平移</span></div>
   </div>
   <div id="workflowNote" class="workflow-note"><b>不必逐一標記，也能直接匯出接著修。</b>先開 working.svg，完整候選向量可直接編輯；draft.svg 用來照圖重描，<b id="referencePackageLabel">草稿含參考圖片</b>。accepted.svg 僅保留你已採用的部分。<span id="processedReferenceNote" hidden>此結果只有清理後參考圖，並非原始圖片；判斷細節時請另對照原檔。</span></div>
   <div class="review-note"><b id="pendingNote">正在載入區域判斷…</b><br>系統建議是檢查起點。請留意文字、細線、遮擋與小孔洞，再決定是否採用。</div>
   <div class="review-note"><span id="sourceAuditNote"></span><details id="sceneSourceConcerns" hidden><summary id="sceneSourceSummary"></summary><p>這些是全圖的連通、孔洞或留白疑點，尚未指定由哪個物件造成。放大後請並排查看原圖；不會自動採用或刪除物件。</p><div id="sceneSourceRegions"></div></details></div>
  </section>
  <aside class="inspector" aria-label="物件與處理判斷">
   <div class="inspector-head"><div class="inspector-title"><h2>區域與物件</h2><span id="listCount">0 個物件</span></div><label class="sr-only" for="objectSearch">搜尋區域或物件</label><input id="objectSearch" class="search" type="search" placeholder="搜尋名稱或處理原因…" autocomplete="off"><div class="filter-row"><select id="statusFilter" aria-label="依處理狀態篩選"><option value="all">全部狀態</option><option value="review">只看待確認</option><option value="redraw">只看交人工</option><option value="keep">只看已採用</option><option value="suggest-keep">建議採用（需確認）</option><option value="suggest-redraw">建議重畫（需確認）</option></select><select id="sortOrder" aria-label="物件排序"><option value="priority">原圖差異優先</option><option value="anchors">節點多的優先</option><option value="original">原始順序</option></select></div></div>
   <div class="selection-bar"><span id="selectionCount">未選取</span><div><button id="selectVisible" class="text-button">選取目前列表</button><button id="clearSelection" class="text-button">清除</button></div></div>
   <div id="objectList" class="objects" role="group" aria-label="物件列表"></div>
   <div class="decision-panel"><div class="decision-title"><span id="decisionTitle">選取物件後決定</span><div class="undo-group"><button id="undoButton" disabled title="復原 Ctrl+Z" aria-label="復原判斷">↶</button><button id="redoButton" disabled title="重做 Ctrl+Shift+Z" aria-label="重做判斷">↷</button></div></div>
    <div class="decisions" role="group" aria-label="設定所選物件的處理方式"><button class="decision-button" data-action="keep" disabled><span class="symbol" aria-hidden="true">✓</span>採用 <span class="small">1</span></button><button class="decision-button" data-action="review" disabled><span class="symbol" aria-hidden="true">◷</span>待確認 <span class="small">2</span></button><button class="decision-button" data-action="redraw" disabled><span class="symbol" aria-hidden="true">✎</span>交人工 <span class="small">3</span></button></div>
    <div id="selectionDetail" class="selection-detail">點選圖稿或右側物件。框選需完整包住物件範圍，才會選取。</div><div class="history-hint">快捷鍵 1 / 2 / 3 切換判斷；Ctrl + Z 復原。</div>
    <div class="refine-panel"><label for="refineBudget">簡化誤差預算 · 整張／局部共用</label><div class="refine-controls"><select id="refineBudget" aria-label="簡化誤差預算"><option value="0.1">保守 0.1%</option><option value="0.25" selected>標準 0.25%</option><option value="0.5">較大 0.5%</option></select><button id="refineButton" disabled>僅簡化所選輪廓</button></div><p id="refineHelp" class="refine-help">可用上方「自動整理整張圖」，也可直接匯出後自行編輯。只有想指定小區域時，才用局部簡化。先用標準；保守會保留更多節點。重建規則幾何時，會另外對照原圖檢查，並標出結果。</p><details class="refine-help"><summary>誤差數字怎麼看</summary><p>百分比依輪廓對角線換算。95% 的採樣偏差須在設定值內，最大採樣偏差不得超過 3 倍；小孔、尖角與透明裂縫另有檢查。這不是完稿率，也不是所有曲線位置的數學保證。</p></details></div>
   </div>
  </aside>
 </div>
</main>
<script id="handoff-data" type="application/json">@@PAYLOAD@@</script>
<script>
'use strict';
@@STATE@@
(() => {
const data = JSON.parse(document.getElementById('handoff-data').textContent);
const $ = id => document.getElementById(id);
const labels = {keep:'採用',review:'待確認',redraw:'交人工'};
const objects = data.objects, byId = new Map(objects.map(o => [o.id,o]));
const storageKey = 'aivc.handoff.v1:' + data.svg_sha256 + ':' + data.result;
const draftWriterId=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random().toString(36).slice(2);
let localRecord=null, localReadFailed=false;
try { const raw=localStorage.getItem(storageKey);
  if(raw!==null){try{localRecord=JSON.parse(raw);}catch(_){localRecord={unreadable_raw:raw};}}
}catch(_){localReadFailed=true;}
const recovered=resolveHandoffDraft(data,localRecord);
const state = new HandoffState(objects,data.default_decisions,recovered.initial);
let recoveredLocal=recovered.recovered, savedSnapshot=recovered.serverSnapshot;
let draftBackups=recovered.backups, pendingDraft=recovered.pending;
let currentRevision = data.saved_revision;
let localStorageFailed = localReadFailed;
const selected = new Set();
let hovered = null, visibleObjects = [], mode = 'compare', lastClicked = null, allowNavigation = false;
const memberToObjects = new Map();
for (const obj of objects) for (const member of obj.member_ids) {
  if (!memberToObjects.has(member)) memberToObjects.set(member,[]);
  memberToObjects.get(member).push(obj.id);
}
const viewBox = data.view_box, [vx,vy,vw,vh] = viewBox;
const sourceBoard = $('sourceBoard'), vectorBoard = $('vectorBoard');
const vectorSvg = vectorBoard.querySelector('svg:not(.overlay)');
vectorSvg.setAttribute('viewBox',viewBox.join(' '));
const viewports = [$('sourceViewport'),$('vectorViewport')];
const boards = [sourceBoard,vectorBoard];
const overlays = boards.map(board=>board.querySelector('.overlay'));
overlays.forEach(overlay=>overlay.setAttribute('viewBox',viewBox.join(' ')));
const isOriginalReference=data.reference_kind==='original';
$('sourceLabel').textContent=isOriginalReference?'原始圖片 · 參考底圖':'清理後參考圖（非原始圖片）';
$('sourceImage').alt=isOriginalReference?'原始圖片參考':'清理後參考圖（非原始圖片）';
$('referencePackageLabel').textContent=isOriginalReference?'草稿含原圖參考':'草稿含清理後參考圖';
$('processedReferenceNote').hidden=isOriginalReference;
if (data.source_data_url) $('sourceImage').src=data.source_data_url;
else { $('sourceImage').hidden=true; $('sourceMissing').hidden=false; }
let scale=1, centerX=vx+vw/2, centerY=vy+vh/2, fitScale=1, panMode=false, spaceHeld=false, drag=null;
const ns='http://www.w3.org/2000/svg';
function element(tag,cls,text) { const node=document.createElement(tag); if(cls)node.className=cls; if(text!==undefined)node.textContent=text; return node; }
function svgRect(box,cls) { const rect=document.createElementNS(ns,'rect');const pad=5/Math.max(.000001,scale),outline=[box[0]-pad,box[1]-pad,box[2]+2*pad,box[3]+2*pad]; ['x','y','width','height'].forEach((key,i)=>rect.setAttribute(key,outline[i])); rect.setAttribute('class',cls);rect.setAttribute('fill','none'); return rect; }
function storageSave() { try {
  const raw=localStorage.getItem(storageKey);let latest=null;
  if(raw!==null){try{latest=JSON.parse(raw);}catch(_){latest={unreadable_raw:raw};}}
  pendingDraft=mergeOtherHandoffDraft(data,latest,draftWriterId,state.decisions,draftBackups,pendingDraft);
  localStorage.setItem(storageKey,JSON.stringify({schema:'aivc.handoff-draft/v2',writerId:draftWriterId,
   ...makeHandoffDraft(data.svg_sha256,currentRevision,state.decisions),backups:draftBackups,pendingDraft}));localStorageFailed=false;
 }catch(_){localStorageFailed=true;}
 renderDraftNotice();
}
function renderDraftNotice(){
 const hasBackup=draftBackups.length>0||pendingDraft;
 $('draftNotice').hidden=!hasBackup;
 $('recoverDraftButton').hidden=!pendingDraft;
 $('recoverDraftButton').disabled=busy||!restorableHandoffDraft(data,pendingDraft);
 $('downloadDraftButton').disabled=busy;
 $('draftNoticeText').textContent=pendingDraft?
  '另有其他分頁的本機草稿，或草稿所依據的工作台版本不同。它未自動覆寫目前判斷。可下載備份，或取回後逐區核對，再自行儲存。':
  `已保留 ${draftBackups.length} 份先前的本機草稿，可下載 JSON；不會自動覆寫目前判斷。`;
}
function recoverDraft(){
 if(busy||!restorableHandoffDraft(data,pendingDraft))return;
 preserveHandoffDraft(draftBackups,makeHandoffDraft(data.svg_sha256,currentRevision,state.decisions));
 const candidate=pendingDraft;preserveHandoffDraft(draftBackups,candidate);pendingDraft=null;
 state.restore(candidate.decisions);recoveredLocal=true;afterChange();renderDraftNotice();
 message('已取回本機草稿，尚未寫入工作台。請逐區核對；可按復原回到取回前，再決定是否儲存。');
}
function downloadDrafts(){
 const payload={schema:'aivc.handoff-draft-backup/v1',result:data.result,
  current:makeHandoffDraft(data.svg_sha256,currentRevision,state.decisions),pendingDraft,backups:draftBackups};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='接手判斷草稿備份.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function updateSaveState() {
  const same=JSON.stringify(state.decisions)===savedSnapshot;
  $('saveState').textContent=same ? '已儲存至工作台' : localStorageFailed ? '尚未儲存；瀏覽器暫存不可用' : '判斷已暫存於此瀏覽器';
}
function updateCounts() {
 const c=state.counts();
 for(const action of Object.keys(c)) $(action+'Count').textContent=c[action];
 $('pendingNote').textContent=c.review ? `還有 ${c.review} 個待確認、${c.redraw} 個交人工。可以先匯出草稿繼續處理。` : c.redraw ? `區域判斷已完成；仍有 ${c.redraw} 個區域需要人工接手。` : objects.length ? '所有區域都已標記採用；匯出前仍請目視確認圖稿。' : '這份結果尚無可供判斷的物件。';
 $('adoptedEmpty').hidden=mode!=='adopted'||c.keep>0;
 $('undoButton').disabled=!state.history.length; $('redoButton').disabled=!state.future.length;
}
function listObjects() {
 const search=$('objectSearch').value.trim().toLocaleLowerCase(),filter=$('statusFilter').value;
 visibleObjects=objects.filter(o=>selectableObject(o.id)&&(filter==='all'||state.decisions[o.id]===filter||(filter.startsWith('suggest-')&&o.suggested_action===filter.slice(8)))&&(!search||(o.label+' '+o.reasons.join(' ')).toLocaleLowerCase().includes(search)));
 if($('sortOrder').value==='anchors') visibleObjects.sort((a,b)=>b.anchor_count-a.anchor_count);
 else if($('sortOrder').value==='priority') {const rank={redraw:0,review:1,keep:2};visibleObjects.sort((a,b)=>Number(b.source_defect_count>0)-Number(a.source_defect_count>0)||b.source_defect_fraction-a.source_defect_fraction||Number(b.curve_review_count>0)-Number(a.curve_review_count>0)||rank[a.suggested_action]-rank[b.suggested_action]||b.anchor_count-a.anchor_count);}
 const list=$('objectList'); list.replaceChildren();
 $('listCount').textContent=`${visibleObjects.length} / ${objects.length} 個`;
 if(!visibleObjects.length)list.append(element('div','list-empty',objects.length?'沒有符合條件的物件':'這份結果尚無可檢查物件'));
 for(const o of visibleObjects) {
  const row=element('label','object-row'+(selected.has(o.id)?' selected':''));row.dataset.objectId=o.id;
  const check=document.createElement('input');check.type='checkbox';check.checked=selected.has(o.id);check.setAttribute('aria-label','選取 '+o.label);
  const content=element('div');content.style.minWidth='0';content.append(element('div','object-label',o.label),element('div','object-meta',`${o.anchor_count.toLocaleString()} 節點 · ${o.path_count.toLocaleString()} 路徑${o.bbox?'':' · 無區域範圍'}`));
  if(o.source_defect_count) content.append(element('div','object-meta',`原圖比對：${o.source_defect_count} 項待查`));
  row.append(check,content,element('span','badge '+state.decisions[o.id],labels[state.decisions[o.id]]));
  row.addEventListener('click',event=>{if(event.target!==check)event.preventDefault();
   if(event.shiftKey&&lastClicked&&visibleObjects.some(obj=>obj.id===lastClicked)){
    const a=visibleObjects.findIndex(obj=>obj.id===lastClicked),b=visibleObjects.findIndex(obj=>obj.id===o.id);
    for(const obj of visibleObjects.slice(Math.min(a,b),Math.max(a,b)+1))selected.add(obj.id);
   } else if(event.target===check||event.shiftKey||event.ctrlKey||event.metaKey){if(selected.has(o.id))selected.delete(o.id);else selected.add(o.id);}
   else {selected.clear();selected.add(o.id);}
   lastClicked=o.id;renderSelection();
  });
  row.addEventListener('dblclick',()=>focusObjects([o.id]));
  row.addEventListener('mouseenter',()=>{hovered=o.id;drawOverlays();});
  row.addEventListener('mouseleave',()=>{hovered=null;drawOverlays();});
  list.append(row);
 }
 $('selectVisible').disabled=!visibleObjects.length;
}
function drawOverlays() {
 reconcileVisibleSelection();
 for(const overlay of overlays){overlay.replaceChildren();
  if(hovered&&!selected.has(hovered)){const obj=byId.get(hovered);if(obj&&obj.bbox)overlay.append(svgRect(obj.bbox,'hover-box'));}
  for(const id of selected){const obj=byId.get(id);if(obj&&obj.bbox)overlay.append(svgRect(obj.bbox,'selection-box'));}
 }
}
function renderSelection() {
 reconcileVisibleSelection();
 for(const row of $('objectList').querySelectorAll('.object-row')){const has=selected.has(row.dataset.objectId);row.classList.toggle('selected',has);row.querySelector('input').checked=has;}
 const count=selected.size;
 $('selectionCount').textContent=count?`已選 ${count} 個`:'未選取';
 $('selectedCaption').textContent=count?`已選 ${count} 個區域`:'尚未選取物件';
 $('decisionTitle').textContent=count?`這 ${count} 個物件怎麼處理？`:'選取物件後決定';
 $('clearSelection').disabled=!count;
 const actions=new Set([...selected].map(id=>state.decisions[id]));
 for(const button of document.querySelectorAll('[data-action]')){button.disabled=!count;const active=count>0&&actions.size===1&&actions.has(button.dataset.action);button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}
 const detail=$('selectionDetail');detail.replaceChildren();
 if(!count)detail.textContent='點選圖稿或右側物件。框選需完整包住物件範圍，才會選取。';
 else if(count===1){const obj=byId.get([...selected][0]);detail.append(element('p','',`系統建議：${labels[obj.suggested_action]}`));
  if(obj.reasons.length){const ul=element('ul');for(const reason of obj.reasons)ul.append(element('li','',reason));detail.append(ul);}
  if(!obj.reasons.length)detail.append(element('p','',obj.source_audit_status==='checked'?'目前未找到具體的顏色／覆蓋差異提示；仍待你確認設計意圖。':obj.source_audit_status==='no_measurable_visible_contribution'?'此物件目前被遮住或對畫面沒有可量測影響；改動遮擋後需再確認。':'這個物件的原圖像素比對尚未完成；沒有提示不代表通過。'));
  else if(!['checked','no_measurable_visible_contribution'].includes(obj.source_audit_status))detail.append(element('p','', '原圖像素比對尚未完成；以下結構建議不能當作外觀驗收。'));
  if(!obj.bbox)detail.append(element('p','', '這個物件沒有區域範圍；請由物件列表檢查。'));
 } else detail.append(element('p','',`共 ${[...selected].reduce((sum,id)=>sum+byId.get(id).anchor_count,0).toLocaleString()} 個節點。按下處理方式會套用至全部 ${count} 個物件。`));
 if(count&&[...selected].some(id=>byId.get(id).bbox)){const focus=element('button','focus-button','放大所選範圍');focus.onclick=()=>focusObjects([...selected]);detail.append(focus);}
 updateRefineButton();
 drawOverlays();
}
function updateVisibility() {
 const shapes=vectorSvg.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon,text,image,use');
 for(const node of shapes){if(node.closest('defs,clipPath,mask,pattern,marker'))continue;
  let ids=[],current=node;
  while(current&&current!==vectorBoard){if(current.hasAttribute('data-source-id'))ids.push(...(memberToObjects.get(current.getAttribute('data-source-id'))||[]));current=current.parentElement;}
  const visible=mode!=='adopted'||ids.some(id=>state.decisions[id]==='keep');
  if(visible) { if(node.hasAttribute('data-handoff-hidden')){node.style.display=node.getAttribute('data-handoff-display')||'';node.removeAttribute('data-handoff-hidden');} }
  else { if(!node.hasAttribute('data-handoff-hidden'))node.setAttribute('data-handoff-display',node.style.display||'');node.setAttribute('data-handoff-hidden','true');node.style.display='none'; }
 }
 $('vectorLabel').textContent=mode==='adopted'?'採用預覽 · 僅保留已採用部分':'向量結果 · 點選物件';
 updateCounts();
}
function afterChange(){storageSave();updateSaveState();updateCounts();listObjects();renderSelection();updateVisibility();}
function decision(action){if(state.apply([...selected],action))afterChange();}
function undo(){if(state.undo())afterChange();}
function redo(){if(state.redo())afterChange();}
function activeViewport(){return $('vectorViewport');}
function layout(){
 boards.forEach((board,index)=>{const vp=viewports[index];board.style.width=(vw*scale)+'px';board.style.height=(vh*scale)+'px';board.style.left=(vp.clientWidth/2-(centerX-vx)*scale)+'px';board.style.top=(vp.clientHeight/2-(centerY-vy)*scale)+'px';});
 $('zoomText').textContent=Math.round(scale/fitScale*100)+'%';
 drawOverlays();
}
function fit(){const vp=activeViewport();fitScale=Math.max(.001,Math.min((vp.clientWidth-70)/vw,(vp.clientHeight-85)/vh));scale=fitScale;centerX=vx+vw/2;centerY=vy+vh/2;layout();}
function screenToWorld(vp,x,y){const r=vp.getBoundingClientRect();return [centerX+(x-r.left-vp.clientWidth/2)/scale,centerY+(y-r.top-vp.clientHeight/2)/scale];}
function zoom(factor,vp=activeViewport(),x=null,y=null){
 const r=vp.getBoundingClientRect(),px=x===null?r.left+vp.clientWidth/2:x,py=y===null?r.top+vp.clientHeight/2:y;
 const before=screenToWorld(vp,px,py);scale=Math.min(fitScale*40,Math.max(fitScale*.2,scale*factor));const after=screenToWorld(vp,px,py);centerX+=before[0]-after[0];centerY+=before[1]-after[1];layout();
}
function focusObjects(ids){const boxes=ids.map(id=>byId.get(id)?.bbox).filter(Boolean);if(!boxes.length)return;
 const x=Math.min(...boxes.map(b=>b[0])),y=Math.min(...boxes.map(b=>b[1])),r=Math.max(...boxes.map(b=>b[0]+b[2])),b=Math.max(...boxes.map(b=>b[1]+b[3]));
 const vp=activeViewport();centerX=(x+r)/2;centerY=(y+b)/2;scale=Math.min(fitScale*40,Math.min((vp.clientWidth-100)/Math.max(r-x,vw*.01),(vp.clientHeight-100)/Math.max(b-y,vh*.01)));layout();
}
function showSourceHints(){
 const count=objects.filter(o=>o.source_defect_count>0).length;
 $('sourceAuditNote').textContent=data.source_audit_status==='completed'?`已對照原圖可見像素，${count} 個物件有具體差異提示。沒有提示仍需你確認；這不是完稿驗收。`:'逐物件原圖比對尚未完整完成；沒有提示不代表外觀通過。';
 const concerns=data.scene_source_concerns||[];
 $('sceneSourceConcerns').hidden=!concerns.length;
 $('sceneSourceSummary').textContent=`另有 ${concerns.length} 處全圖結構疑點，點開定位`;
 for(const concern of concerns){const button=element('button','focus-button',concern.label+' · 放大位置');
  button.onclick=()=>{document.querySelector('[data-view="compare"]').click();
   const [x,y,w,h]=concern.bbox,vp=activeViewport();centerX=x+w/2;centerY=y+h/2;scale=Math.min(fitScale*40,Math.min((vp.clientWidth-100)/Math.max(w,vw*.01),(vp.clientHeight-100)/Math.max(h,vh*.01)));updateVisibility();layout();};
  $('sceneSourceRegions').append(button);
 }
}
function setPan(){for(const vp of viewports)vp.classList.toggle('pan-mode',panMode||spaceHeld);$('panButton').setAttribute('aria-pressed',String(panMode));}
function selectableObject(id){return mode!=='adopted'||state.decisions[id]==='keep';}
function reconcileVisibleSelection(){
 for(const id of selected)if(!selectableObject(id))selected.delete(id);
 if(hovered&&!selectableObject(hovered))hovered=null;
}
function hitObjects(event,vp){
 if(vp===$('vectorViewport')){let node=event.target;while(node&&node!==vp){const sourceId=node.getAttribute?.('data-source-id');if(sourceId&&memberToObjects.has(sourceId))return memberToObjects.get(sourceId).filter(selectableObject);node=node.parentElement;}}
 const [x,y]=screenToWorld(vp,event.clientX,event.clientY);
 const candidates=objects.filter(o=>selectableObject(o.id)&&o.bbox&&x>=o.bbox[0]&&x<=o.bbox[0]+o.bbox[2]&&y>=o.bbox[1]&&y<=o.bbox[1]+o.bbox[3]);
 candidates.sort((a,b)=>a.bbox[2]*a.bbox[3]-b.bbox[2]*b.bbox[3]);return candidates.length?[candidates[0].id]:[];
}
for(const vp of viewports){
 vp.addEventListener('wheel',event=>{event.preventDefault();zoom(Math.exp(-event.deltaY*.0015),vp,event.clientX,event.clientY);},{passive:false});
 vp.addEventListener('pointerdown',event=>{if(event.button!==0&&event.button!==1)return;event.preventDefault();
  drag={vp,id:event.pointerId,x:event.clientX,y:event.clientY,cx:centerX,cy:centerY,world:screenToWorld(vp,event.clientX,event.clientY),pan:panMode||spaceHeld||event.button===1,extend:event.shiftKey||event.ctrlKey||event.metaKey,hit:hitObjects(event,vp),moved:false};
  vp.setPointerCapture(event.pointerId);vp.classList.toggle('dragging',drag.pan);
 });
 vp.addEventListener('pointermove',event=>{
  if(!drag||drag.vp!==vp){const hit=hitObjects(event,vp);if(hovered!==(hit[0]||null)){hovered=hit[0]||null;drawOverlays();}return;}
  const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.moved=drag.moved||Math.hypot(dx,dy)>4;
  if(drag.pan){centerX=drag.cx-dx/scale;centerY=drag.cy-dy/scale;layout();}
  else if(drag.moved){const r=vp.getBoundingClientRect(),box=vp.querySelector('.marquee');box.style.display='block';box.style.left=(Math.min(drag.x,event.clientX)-r.left)+'px';box.style.top=(Math.min(drag.y,event.clientY)-r.top)+'px';box.style.width=Math.abs(dx)+'px';box.style.height=Math.abs(dy)+'px';}
 });
 vp.addEventListener('pointerup',event=>{if(!drag||drag.vp!==vp)return;
  if(!drag.pan){if(!drag.extend)selected.clear();
   if(drag.moved){const end=screenToWorld(vp,event.clientX,event.clientY);
    for(const obj of objects)if(selectableObject(obj.id)&&handoffBoundsContained(obj.bbox,drag.world,end))selected.add(obj.id);
   }else{for(const id of drag.hit){if(drag.extend&&selected.has(id))selected.delete(id);else selected.add(id);}}
   renderSelection();
   if(selected.size===1){const id=[...selected][0];for(const row of $('objectList').querySelectorAll('.object-row'))if(row.dataset.objectId===id)row.scrollIntoView({block:'nearest'});}
  }
  vp.querySelector('.marquee').style.display='none';vp.classList.remove('dragging');if(vp.hasPointerCapture(event.pointerId))vp.releasePointerCapture(event.pointerId);drag=null;
 });
 vp.addEventListener('pointercancel',()=>{vp.querySelector('.marquee').style.display='none';vp.classList.remove('dragging');drag=null;});
 vp.addEventListener('pointerleave',()=>{if(!drag){hovered=null;drawOverlays();}});
 vp.addEventListener('dblclick',()=>{if(selected.size)focusObjects([...selected]);});
}
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>{
 mode=button.dataset.view;reconcileVisibleSelection();
 for(const b of document.querySelectorAll('[data-view]')){b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));}
 $('canvasStage').classList.toggle('single',mode!=='compare');listObjects();renderSelection();updateVisibility();fit();
};
for(const button of document.querySelectorAll('[data-action]'))button.onclick=()=>decision(button.dataset.action);
$('panButton').onclick=()=>{panMode=!panMode;setPan();};$('zoomIn').onclick=()=>zoom(1.25);$('zoomOut').onclick=()=>zoom(.8);$('fitButton').onclick=fit;
$('undoButton').onclick=undo;$('redoButton').onclick=redo;
$('objectSearch').oninput=()=>{listObjects();renderSelection();};$('statusFilter').onchange=()=>{listObjects();renderSelection();};$('sortOrder').onchange=()=>{listObjects();renderSelection();};
$('selectVisible').onclick=()=>{selected.clear();for(const obj of visibleObjects)selected.add(obj.id);renderSelection();};$('clearSelection').onclick=()=>{selected.clear();renderSelection();};
document.addEventListener('keydown',event=>{if(event.target.matches('input,textarea,select')||event.target.isContentEditable)return;
 if(event.code==='Space'){event.preventDefault();spaceHeld=true;setPan();}
 if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();}
 else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='y'){event.preventDefault();redo();}
 else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();save();}
 else if(!event.ctrlKey&&!event.metaKey&&['1','2','3'].includes(event.key)){event.preventDefault();decision({1:'keep',2:'review',3:'redraw'}[event.key]);}
 else if(event.key==='Escape'){selected.clear();renderSelection();}
});
document.addEventListener('keyup',event=>{if(event.code==='Space'){spaceHeld=false;setPan();}});
window.addEventListener('blur',()=>{spaceHeld=false;setPan();});
window.addEventListener('beforeunload',event=>{if(!allowNavigation&&JSON.stringify(state.decisions)!==savedSnapshot&&(state.revision>0||recoveredLocal)){event.preventDefault();event.returnValue='';}});
new ResizeObserver(()=>layout()).observe($('canvasStage'));
function message(text,error=false){const box=$('statusMessage');box.replaceChildren(element('div','',text));box.className='status-message show'+(error?' error':'');}
let busy=false;
async function request(kind,extra={}){
 const snapshot={...state.decisions};
 const response=await fetch('/api/handoff/'+kind,{method:'POST',headers:{'Content-Type':'application/json','X-WB-Token':data.token},body:JSON.stringify({result:data.result,svg_sha256:data.svg_sha256,decisions:snapshot,...extra,revision:currentRevision})});
 let result;try{result=await response.json();}catch(_){throw new Error('工作台回應無法讀取，請重新嘗試。');}
 if(!response.ok||result?.error){const detail=typeof result?.error==='string'?result.error:result?.message||'工作台無法完成要求。';throw new Error(detail+(response.status===409?' 目前頁面的判斷仍保留。請先重新載入結果並核對其他視窗的變更，再儲存或匯出。':''));}
 if(kind==='save'||kind==='export'){
  if(typeof result.revision!=='string')throw new Error('工作台未回傳儲存版本，請重新載入頁面後確認儲存結果。');
  currentRevision=result.revision;
  storageSave();
 }
 return {result,snapshot};
}
function updateRefineButton(){const eligible=selected.size>0&&[...selected].every(id=>state.decisions[id]!=='keep');$('refineButton').disabled=busy||!eligible;$('refineBudget').disabled=busy;$('prepareButton').disabled=busy||objects.every(o=>state.decisions[o.id]==='keep');}
function busyButtons(value){busy=value;$('saveButton').disabled=value;$('exportButton').disabled=value;updateRefineButton();renderDraftNotice();}
async function save(){if(busy)return;busyButtons(true);$('saveButton').textContent='儲存中…';
 try{const {snapshot}=await request('save');savedSnapshot=JSON.stringify(snapshot);updateSaveState();message('判斷已儲存至工作台。');}
 catch(error){message('儲存未完成：'+error.message,true);}finally{busyButtons(false);$('saveButton').textContent='儲存判斷';}
}
async function exportPackage(){if(busy)return;busyButtons(true);$('exportButton').textContent='正在建立接手包…';
 try{const {result,snapshot}=await request('export');
  savedSnapshot=JSON.stringify(snapshot);updateSaveState();
  const files=Array.isArray(result.files)?result.files:[];
  const safeFiles=files.map(file=>{try{const url=new URL(file.url,location.href);return url.origin===location.origin&&/^https?:$/.test(url.protocol)?{name:String(file.name||'下載檔案'),url:url.href}:null;}catch(_){return null;}}).filter(Boolean);
  if(!safeFiles.length)throw new Error('接手包未回傳可下載的檔案。');
  const counts={review:0,redraw:0,keep:0};for(const value of Object.values(snapshot))counts[value]++;
  message(`接手包已建立。下載 ZIP 後，開啟 working.svg 就能接著修。採用 ${counts.keep} 個；仍有 ${counts.review} 個待確認、${counts.redraw} 個交人工。`);
  const box=$('statusMessage'),details=element('details');details.append(element('summary','','詳細交接資料'));
  for(const file of safeFiles){const a=element('a','',file.name);a.href=file.url;a.setAttribute('download','');(/\.json$/i.test(file.name)?details:box).append(a);}
  if(details.children.length>1)box.append(details);
  if(JSON.stringify(snapshot)!==JSON.stringify(state.decisions))box.append(element('div','small','這份接手包使用按下匯出時的判斷；之後的變更尚未包含。'));
 }catch(error){message('匯出未完成：'+error.message,true);}finally{busyButtons(false);$('exportButton').textContent='匯出 Illustrator 接手包 ↗';}
}
async function refineSelection(){
 if(busy||!selected.size||[...selected].some(id=>state.decisions[id]==='keep'))return;
 const ids=[...selected];busyButtons(true);$('refineButton').textContent='簡化中…';
 message(`正在簡化 ${ids.length} 個所選區域，最多約 1 分鐘。會產生新版本，保留原版與已採用的物件。`);
 try{const {result,snapshot}=await request('refine',{object_ids:ids,error_budget_percent:Number($('refineBudget').value)});
  const destination=new URL(result.url,location.href);
  if(!result.url||destination.origin!==location.origin||!/^https?:$/.test(destination.protocol))throw new Error('工作台未回傳可開啟的新版本。');
  // Preserve the old page's decisions; the server carries the submitted snapshot into the derivative.
  storageSave();
  if(JSON.stringify(snapshot)!==JSON.stringify(state.decisions)){
   message('新版本已建立。簡化期間你更新了判斷，因此先保留目前頁面；新版本使用開始簡化時的判斷。');
   const link=element('a','','開啟簡化的新版本');link.href=destination.href;link.target='_blank';link.rel='noopener';$('statusMessage').append(link);
  }else {allowNavigation=true;location.assign(destination.href);}
 }catch(error){message('局部簡化未完成：'+error.message,true);}
 finally{busyButtons(false);$('refineButton').textContent='僅簡化所選輪廓';}
}
function preparationReason(unit){
 if(unit.proposal_kind==='source_primitive')return '依原圖重建為規則幾何，節點更少且比原候選更貼近原圖；請確認造型';
 if(unit.status==='improved')return '已減少節點，通過輪廓、孔洞及顏色檢查';
 const r=String(unit.reason||'');
 if(r.includes('keep_locked'))return '你已採用，保持原狀';
 if(/time|budget_exhausted/.test(r))return '本次整理時間已到，保持原狀';
 if(/already_refined|history/.test(r))return '已整理過，避免重複累積誤差';
 if(/stroke|open_path/.test(r))return '筆畫或開放曲線，保持原本的編輯方式';
 if(/native|not_a_path|too_few|low_anchor/.test(r))return '已是簡單形狀，保持原狀';
 if(/whole_alpha_components/.test(r))return '簡化會使物件黏合、分裂或消失，保留原狀';
 if(/whole_alpha_holes/.test(r))return '簡化會改變孔洞或封住留白，保留原狀';
 if(/whole_alpha/.test(r))return '簡化會改變透明邊緣或缺少可靠檢查，保留原狀';
 if(/source|render|topology|geometry|paint|silhouette/.test(r))return '簡化可能影響圖形，保留原狀供你判斷';
 return '未找到安全的減點方式，保持原狀';
}
function showPreparation(record){
 if(!record||!record.summary)return;
 const s=record.summary,n=value=>Number.isFinite(Number(value))?Number(value):0;
 $('prepareSummary').hidden=false;
 $('prepareSummaryText').textContent=s.status==='unchanged'?'本次未找到能安全改善的輪廓，圖稿保持原狀。':`已整理 ${n(s.units_improved)} 個區域，節點由 ${n(s.anchors_before)} 減為 ${n(s.anchors_after)}；${n(s.units_skipped)} 個區域保持原狀。`;
 if(n(s.source_reconstructed_paths)>0)$('prepareSummaryText').textContent+=` 其中 ${n(s.source_reconstructed_paths)} 個依原圖重建為規則幾何，請確認造型是否符合原意。`;
 const list=$('prepareUnits');list.replaceChildren();
 for(const unit of Array.isArray(record.units)?record.units:[]){
  const row=element('div','');row.style.margin='7px 0';const id=String(unit.id||''),obj=byId.get(id);
  if(obj){const button=element('button','text-button',obj.label);button.onclick=()=>{selected.clear();selected.add(id);renderSelection();focusObjects([id]);};row.append(button);}
  else row.append(element('span','',id));
  row.append(document.createTextNode(' — '+preparationReason(unit)+(unit.status==='improved'?`（${n(unit.anchors_before)} → ${n(unit.anchors_after)} 節點）`:'')));list.append(row);
 }
}
async function prepareWhole(){
 if(busy)return;busyButtons(true);const started=Date.now();$('prepareButton').textContent='正在整理…';
 const progress=()=>{$('prepareProgress').textContent=`正在逐區檢查，已經過 ${Math.floor((Date.now()-started)/1000)} 秒；整理搜尋最多約 3 分鐘，最後檢查與保存另需時間。可以繼續看圖。`;};
 progress();const timer=setInterval(progress,1000);
 try{const {result,snapshot}=await request('prepare',{error_budget_percent:Number($('refineBudget').value)});
  showPreparation(result);storageSave();
  if(result.url===null){message('檢查完成，沒有找到能安全減點的地方；原圖稿保持原狀。');return;}
  const destination=new URL(result.url,location.href);
  if(!result.url||destination.origin!==location.origin||!/^https?:$/.test(destination.protocol))throw new Error('工作台未回傳可開啟的新版本。');
  if(JSON.stringify(snapshot)!==JSON.stringify(state.decisions)){
   message('整理完成。期間你更新了判斷，因此先保留目前頁面；新版本使用開始整理時的判斷。');
   const link=element('a','','開啟整理完成的新版本');link.href=destination.href;link.target='_blank';link.rel='noopener';$('statusMessage').append(link);
  }else{allowNavigation=true;location.assign(destination.href);}
 }catch(error){message('自動整理未完成：'+error.message,true);}
 finally{clearInterval(timer);busyButtons(false);$('prepareButton').textContent='自動整理整張圖';$('prepareProgress').textContent='整理會另存新版本；已採用物件保持原狀。';}
}
$('saveButton').onclick=save;$('exportButton').onclick=exportPackage;
$('refineButton').onclick=refineSelection;
$('prepareButton').onclick=prepareWhole;
$('recoverDraftButton').onclick=recoverDraft;$('downloadDraftButton').onclick=downloadDrafts;
listObjects();renderSelection();updateCounts();updateVisibility();storageSave();updateSaveState();renderDraftNotice();
showPreparation(data.auto_prepare);
showSourceHints();
if(recoveredLocal)message('已恢復這張圖在此瀏覽器暫存的判斷；請按「儲存判斷」寫回工作台。');
requestAnimationFrame(fit);
})();
</script></body></html>
"""
