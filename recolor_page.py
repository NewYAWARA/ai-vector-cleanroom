"""Generate a completely offline paint-role editor for a delivered SVG.

The SVG remains the rendering authority.  This page is only a convenient
front end for the deterministic paint-role manifest produced by
``paint_roles.py``; exporting writes ordinary explicit ``fill``, ``stroke``
and ``stop-color`` attributes, so the result does not depend on this page or
on CSS custom properties once it is opened in a vector editor.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import re
from typing import Mapping
import xml.etree.ElementTree as ET


_ROLE_LABELS = {
    "neutral-dark": "深色中性色",
    "neutral-mid": "中間中性色",
    "neutral-light": "淺色中性色",
}

_SVG_NS = "http://www.w3.org/2000/svg"
_MANIFEST_SCHEMA = "ai-vector-cleanroom.paint-roles/v1"
_ROLE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_HEX_RGB_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FORBIDDEN_SVG_ELEMENTS = {
    "script", "foreignobject", "image", "feimage", "iframe", "audio",
    "video", "animate", "animatemotion", "animatetransform", "set",
    "discard",
}
_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)


def _safe_json(value: object) -> str:
    # A literal </script> inside metadata or a filename must not terminate the
    # embedded JSON block.  Escaping '<' is valid JSON and leaves the parsed
    # value unchanged.
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (encoded.replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _has_external_url(value: str) -> bool:
    return any(not match.group(2).strip().startswith("#")
               for match in _URL_RE.finditer(value))


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _validate_svg(svg_text: str) -> None:
    """Reject active/external content before embedding or exporting it."""

    if (re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", svg_text, re.IGNORECASE)
            or re.search(r"<\?xml-stylesheet\b", svg_text, re.IGNORECASE)):
        raise ValueError(
            "SVG must not contain a doctype, entity, or stylesheet declaration")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"SVG is not well-formed XML: {exc}") from exc
    if root.tag != f"{{{_SVG_NS}}}svg":
        raise ValueError("SVG root must use the standard SVG namespace")
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag.lower() in _FORBIDDEN_SVG_ELEMENTS:
            raise ValueError(f"SVG element <{tag}> is not allowed in the offline editor")
        if tag == "style":
            css = "".join(element.itertext())
            if ("@import" in css.lower() or "javascript:" in css.lower()
                    or _has_external_url(css)):
                raise ValueError("SVG style contains active or external content")
        for qualified_name, raw_value in element.attrib.items():
            name = _local_name(qualified_name).lower()
            value = raw_value.strip()
            lowered = value.lower()
            if name.startswith("on"):
                raise ValueError(f"SVG event attribute {name!r} is not allowed")
            if "javascript:" in lowered or lowered.startswith("data:"):
                raise ValueError("SVG active or embedded URL content is not allowed")
            if name == "href" and value and not value.startswith("#"):
                raise ValueError("SVG references must remain local to the document")
            if _has_external_url(value):
                raise ValueError("SVG paint/filter URLs must remain local")


def _validate_manifest(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    roles = list(manifest.get("roles") or [])
    if not roles:
        raise ValueError("paint-role manifest contains no controls")
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError("unsupported paint-role manifest schema")
    role_ids: set[str] = set()
    member_colours: set[str] = set()
    for role in roles:
        if not isinstance(role, Mapping):
            raise ValueError("paint-role controls must be JSON objects")
        role_id = str(role.get("id") or "")
        if not _ROLE_ID_RE.fullmatch(role_id) or role_id in role_ids:
            raise ValueError(f"invalid or duplicate paint role id: {role_id!r}")
        role_ids.add(role_id)
        if role.get("kind") not in {"neutral", "chromatic"}:
            raise ValueError(f"paint role {role_id!r} has an invalid kind")
        control = role.get("control")
        if not isinstance(control, Mapping):
            raise ValueError(f"paint role {role_id!r} has no control definition")
        default_hex = str(control.get("default_hex") or "")
        if not _HEX_RGB_RE.fullmatch(default_hex):
            raise ValueError(f"paint role {role_id!r} has an invalid default colour")
        anchor_chroma = control.get("anchor_chroma", 0.0)
        if (isinstance(anchor_chroma, bool)
                or not isinstance(anchor_chroma, (int, float))
                or not math.isfinite(float(anchor_chroma))):
            raise ValueError(f"paint role {role_id!r} has invalid control values")
        members = role.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"paint role {role_id!r} has no colour members")
        member_count = _nonnegative_int(
            role.get("member_count"), f"paint role {role_id!r} member count")
        _nonnegative_int(
            role.get("usage_count"), f"paint role {role_id!r} usage count")
        if member_count != len(members):
            raise ValueError(f"paint role {role_id!r} member count is inconsistent")
        for member in members:
            if not isinstance(member, Mapping):
                raise ValueError(f"paint role {role_id!r} has an invalid member")
            colour = str(member.get("hex") or "").lower()
            if not _HEX_RGB_RE.fullmatch(colour) or colour in member_colours:
                raise ValueError(f"invalid or duplicate paint member: {colour!r}")
            member_colours.add(colour)
            relative = member.get("relative")
            if not isinstance(relative, Mapping):
                raise ValueError(f"paint member {colour!r} has no relative transform")
            for key in ("lightness_delta", "chroma_ratio",
                        "hue_delta_degrees"):
                value = relative.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))):
                    raise ValueError(f"paint member {colour!r} has invalid {key}")
    return roles


def _safe_download_filename(value: str) -> str:
    filename = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", str(value)).strip(" .")
    if not filename:
        filename = "recolored.svg"
    if not filename.lower().endswith(".svg"):
        filename += ".svg"
    return filename[:-4][:176] + ".svg"


def _display_label(role: Mapping[str, object], index: int) -> str:
    role_id = str(role.get("id") or f"role-{index}")
    if role_id in _ROLE_LABELS:
        return _ROLE_LABELS[role_id]
    if role_id.startswith("accent-"):
        suffix = role_id.split("-", 1)[-1]
        return f"主色系 {suffix}"
    return str(role.get("label") or role_id)


def make_recolor_html(
    output_path: str | Path,
    svg_path: str | Path,
    manifest: Mapping[str, object],
    *,
    download_filename: str | None = None,
    tool_version: str = "",
) -> Path:
    """Write a self-contained, offline SVG paint-role editor.

    The manifest is embedded as data, not fetched.  The page uses the same
    relative OKLCH transform as :func:`paint_roles.apply_role_recolor` and
    always exports explicit SVG paint attributes.
    """

    output = Path(output_path)
    source = Path(svg_path)
    svg_text = source.read_text(encoding="utf-8")
    _validate_svg(svg_text)
    roles = _validate_manifest(manifest)
    filename = _safe_download_filename(
        download_filename or f"{source.stem}_recolored.svg")
    role_cards = []
    for index, role in enumerate(roles, 1):
        role_id = str(role.get("id") or f"role-{index}")
        control = role.get("control") or {}
        default_hex = str(control.get("default_hex") or "#808080")
        kind = "彩色" if role.get("kind") == "chromatic" else "中性色"
        member_count = int(role.get("member_count") or 0)
        usage_count = int(role.get("usage_count") or 0)
        label = _display_label(role, index)
        role_cards.append(f'''\n        <div class="role" data-role-card="{html.escape(role_id, quote=True)}">
          <label for="role-{index}">{html.escape(label)}</label>
          <input id="role-{index}" type="color" data-role="{html.escape(role_id, quote=True)}"
                 value="{html.escape(default_hex, quote=True)}" aria-label="{html.escape(label, quote=True)}">
          <output>{html.escape(default_hex.lower())}</output>
          <small>{kind} · {member_count} 個色階 · {usage_count} 處</small>
        </div>''')

    page = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SVG 全域換色</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"Segoe UI","Noto Sans TC",sans-serif;color:#172033;background:#eef1f5}
*{box-sizing:border-box}body{margin:0;min-height:100vh}.app{display:grid;grid-template-columns:minmax(270px,340px) 1fr;min-height:100vh}
aside{background:#fff;border-right:1px solid #d7dce5;padding:24px;overflow:auto}h1{font-size:22px;margin:0 0 8px}.sub{font-size:13px;line-height:1.55;color:#5c667a;margin:0 0 20px}
.role{display:grid;grid-template-columns:1fr 48px;gap:5px 10px;padding:13px 0;border-top:1px solid #edf0f4}.role label{font-weight:700;align-self:end}.role input{width:48px;height:38px;padding:2px;border:1px solid #cbd2de;border-radius:8px;background:#fff;grid-row:1/3;grid-column:2}.role output{font:12px ui-monospace,monospace;color:#526077}.role small{grid-column:1/-1;color:#7b8494}
.actions{position:sticky;bottom:0;background:linear-gradient(transparent,#fff 18px);padding-top:30px;display:grid;gap:8px}.actions button{border:0;border-radius:10px;padding:11px 14px;font-weight:750;cursor:pointer}.primary{background:#1868db;color:#fff}.secondary{background:#edf2fa;color:#24405f}.notice{font-size:12px;color:#6c7584;line-height:1.5;margin:15px 0 0}
main{position:relative;display:flex;align-items:center;justify-content:center;min-width:0;padding:32px;background-color:#e8ebf0;background-image:linear-gradient(45deg,#dfe3e9 25%,transparent 25%),linear-gradient(-45deg,#dfe3e9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#dfe3e9 75%),linear-gradient(-45deg,transparent 75%,#dfe3e9 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}
#stage{width:min(100%,1400px);height:calc(100vh - 64px);display:flex;align-items:center;justify-content:center;overflow:hidden}#stage svg{max-width:100%;max-height:100%;width:auto;height:auto;filter:drop-shadow(0 10px 28px rgba(14,25,42,.15))}
.status{position:absolute;right:18px;bottom:15px;background:rgba(25,34,49,.88);color:#fff;padding:8px 11px;border-radius:8px;font-size:12px}
@media(max-width:760px){.app{grid-template-columns:1fr}.app aside{border-right:0;border-bottom:1px solid #d7dce5}.actions{position:static}.app main{min-height:62vh;padding:18px}#stage{height:58vh}}
</style>
</head>
<body>
<div class="app">
  <aside>
    <h1>SVG 全域換色</h1>
    <p class="sub">調一個色票，會同步更新同色系的填色、線條與漸層，同時保留原本明暗層次。下載後仍是一般 SVG，可直接交給 Illustrator、Inkscape 或 Affinity Designer。</p>
    <div id="roles">__ROLE_CARDS__
    </div>
    <div class="actions">
      <button id="download" class="primary">下載換色後 SVG</button>
      <button id="reset" class="secondary">恢復原色</button>
    </div>
    <p class="notice">這是顏色家族控制，不會猜測品牌色名稱，也不會把文字還原成字型。下載檔使用明確 SVG 色彩屬性，不依賴本頁才能顯示。__VERSION__</p>
  </aside>
  <main>
    <div id="stage" aria-live="polite"></div>
    <div class="status" id="status">原色預覽</div>
  </main>
</div>
<script id="svg-source" type="application/json">__SVG_SOURCE__</script>
<script id="paint-manifest" type="application/json">__MANIFEST__</script>
<script>
(()=>{
  'use strict';
  const SVG_NS='http://www.w3.org/2000/svg';
  const XMLNS_NS='http://www.w3.org/2000/xmlns/';
  const manifest=JSON.parse(document.getElementById('paint-manifest').textContent);
  const sourceText=JSON.parse(document.getElementById('svg-source').textContent);
  const stage=document.getElementById('stage');
  const controls=[...document.querySelectorAll('input[data-role]')];
  const defaults=Object.fromEntries(controls.map(c=>[c.dataset.role,c.value.toLowerCase()]));
  const roles=Object.create(null);for(const role of manifest.roles)roles[role.id]=role;
  let renderedSvg=null;

  /* PAINT_MATH_START */
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  const pyRound=v=>{const f=Math.floor(v),d=v-f;return d<.5?f:d>.5?f+1:(f%2===0?f:f+1)};
  const hexRgb=h=>{h=h.replace('#','');if(h.length===3)h=[...h].map(x=>x+x).join('');return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16)/255)};
  const lin=v=>v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4);
  const gam=v=>v<=.0031308?12.92*v:1.055*Math.pow(v,1/2.4)-.055;
  function toOklch(hex){
    const [r,g,b]=hexRgb(hex).map(lin);
    const l=.4122214708*r+.5363325363*g+.0514459929*b;
    const m=.2119034982*r+.6806995451*g+.1073969566*b;
    const s=.0883024619*r+.2817188376*g+.6299787005*b;
    const l_=Math.cbrt(l),m_=Math.cbrt(m),s_=Math.cbrt(s);
    const L=.2104542553*l_+.793617785*m_-.0040720468*s_;
    const A=1.9779984951*l_-2.428592205*m_+.4505937099*s_;
    const B=.0259040371*l_+.7827717662*m_-.808675766*s_;
    let H=Math.atan2(B,A)*180/Math.PI;if(H<0)H+=360;
    return [L,Math.hypot(A,B),H];
  }
  function rawRgb(L,C,H){
    const a=C*Math.cos(H*Math.PI/180),b=C*Math.sin(H*Math.PI/180);
    const l_=L+.3963377774*a+.2158037573*b;
    const m_=L-.1055613458*a-.0638541728*b;
    const s_=L-.0894841775*a-1.291485548*b;
    const l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
    return [4.0767416621*l-3.3077115913*m+.2309699292*s,-1.2684380046*l+2.6097574011*m-.3413193965*s,-.0041960863*l-.7034186147*m+1.707614701*s];
  }
  function fromOklch(L,C,H){
    L=clamp(L,0,1);let lo=0,hi=Math.max(0,C),rgb=rawRgb(L,hi,H);
    const inside=x=>x.every(v=>v>=-1e-7&&v<=1.0000001);
    if(!inside(rgb)){for(let i=0;i<24;i++){const mid=(lo+hi)/2,r=rawRgb(L,mid,H);if(inside(r))lo=mid;else hi=mid}rgb=rawRgb(L,lo,H)}
    return '#'+rgb.map(v=>pyRound(clamp(gam(v),0,1)*255).toString(16).padStart(2,'0')).join('');
  }
  function mappingFor(role,target){
    const [L,C,H]=toOklch(target),out={};
    for(const member of role.members){
      const rel=member.relative||{};
      const nl=clamp(L+(+rel.lightness_delta||0),.03,.99);
      let nc,nh;
      if(role.kind==='chromatic'){
        nc=C>1e-6?C*Number(rel.chroma_ratio??1):0;
        nh=(H+.5*(+rel.hue_delta_degrees||0)+360)%360;
      }else{
        const anchor=+(role.control&&role.control.anchor_chroma)||0;
        nc=C*(anchor>1e-7?(+rel.chroma_ratio||0):1);nh=H;
      }
      out[member.hex.toLowerCase()]=fromOklch(nl,nc,nh);
    }
    return out;
  }
  function parsePaint(value){
    const raw=(value||'').trim().replace(/\s*!important\s*$/i,'');
    let m=raw.match(/^#([0-9a-f]{3,8})$/i);if(m){let d=m[1];if(d.length===3||d.length===4)d=[...d].map(x=>x+x).join('');if(d.length!==6&&d.length!==8)return null;return {hex:'#'+d.slice(0,6).toLowerCase(),format:'hex',alpha:d.slice(6,8).toLowerCase()}}
    m=raw.match(/^(rgba?)\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)(?:\s*,\s*([^)]+))?\s*\)$/i);if(!m)return null;
    const channel=v=>{v=v.trim();const percent=v.endsWith('%'),number=Number(percent?v.slice(0,-1):v);if(!v||!Number.isFinite(number))return null;const n=percent?clamp(number,0,100)*2.55:clamp(number,0,255);return pyRound(n)};
    const rgb=[channel(m[2]),channel(m[3]),channel(m[4])];if(rgb.some(v=>v===null))return null;
    if(m[1].toLowerCase()==='rgba'){if(!m[5])return null;const alpha=m[5].trim(),number=Number(alpha.endsWith('%')?alpha.slice(0,-1):alpha);if(!alpha||!Number.isFinite(number))return null}
    return {hex:'#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join(''),format:m[1].toLowerCase(),alpha:(m[5]||'').trim()};
  }
  function replacePaint(value,map){
    const parsed=parsePaint(value);if(!parsed||!map[parsed.hex])return null;const hex=map[parsed.hex];
    if(parsed.format==='rgba'){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return `rgba(${r},${g},${b},${parsed.alpha})`}
    if(parsed.format==='rgb'){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return `rgb(${r},${g},${b})`}
    return hex+parsed.alpha;
  }
  /* PAINT_MATH_END */
  function paintMap(){
    const out=Object.create(null);for(const c of controls){if(c.value.toLowerCase()===defaults[c.dataset.role])continue;Object.assign(out,mappingFor(roles[c.dataset.role],c.value))}return out;
  }
  function freshSvg(){
    const doc=new DOMParser().parseFromString(sourceText,'image/svg+xml');
    const svg=doc.documentElement;
    if(doc.querySelector('parsererror')||!svg||svg.localName!=='svg'||svg.namespaceURI!==SVG_NS)return null;
    return svg;
  }
  function update(){
    const svg=freshSvg(),map=paintMap();
    if(!svg){stage.textContent='SVG 預覽無法載入';return}
    for(const el of [svg,...svg.querySelectorAll('*')]){
      for(const attr of ['fill','stroke','stop-color']){
        if(el.hasAttribute(attr)){const replacement=replacePaint(el.getAttribute(attr),map);if(replacement!==null)el.setAttribute(attr,replacement)}
        if(el.style){const current=el.style.getPropertyValue(attr);if(current){const replacement=replacePaint(current,map);if(replacement!==null)el.style.setProperty(attr,replacement,el.style.getPropertyPriority(attr))}}
      }
    }
    const changed=controls.filter(c=>c.value.toLowerCase()!==defaults[c.dataset.role]).length;
    if(changed){const stale=svg.querySelector('#ai-vector-cleanroom-paint-roles');if(stale)stale.remove()}
    const operation={schema:'ai-vector-cleanroom.paint-recolor/v1',roles:Object.fromEntries(controls.map(c=>[c.dataset.role,c.value.toLowerCase()])),rendering_authority:'explicit SVG presentation attributes'};
    let md=svg.querySelector('#ai-vector-cleanroom-paint-recolor');if(!md){md=document.createElementNS(SVG_NS,'metadata');md.setAttribute('id','ai-vector-cleanroom-paint-recolor');svg.insertBefore(md,svg.firstChild)}md.textContent=JSON.stringify(operation);
    stage.replaceChildren(svg);renderedSvg=svg;
    controls.forEach(c=>c.closest('.role').querySelector('output').value=c.value.toLowerCase());
    document.getElementById('status').textContent=changed?`已調整 ${changed} 個色票`:'原色預覽';
  }
  controls.forEach(c=>c.addEventListener('input',update));
  document.getElementById('reset').addEventListener('click',()=>{controls.forEach(c=>c.value=defaults[c.dataset.role]);update()});
  document.getElementById('download').addEventListener('click',()=>{
    if(!renderedSvg)return;const exported=renderedSvg.cloneNode(true);if(!exported.getAttribute('xmlns'))exported.setAttributeNS(XMLNS_NS,'xmlns',SVG_NS);const body=new XMLSerializer().serializeToString(exported);
    const blob=new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n',body,'\n'],{type:'image/svg+xml'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=__DOWNLOAD_NAME__;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1500);
  });
  update();
})();
</script>
</body>
</html>
'''
    page = page.replace("__ROLE_CARDS__", "".join(role_cards))
    page = page.replace("__SVG_SOURCE__", _safe_json(svg_text))
    page = page.replace("__MANIFEST__", _safe_json(manifest))
    page = page.replace("__DOWNLOAD_NAME__", _safe_json(filename))
    version_note = f" · 工具版本 {tool_version}" if tool_version else ""
    page = page.replace("__VERSION__", html.escape(version_note))
    output.parent.mkdir(parents=True, exist_ok=True)
    from svg_postprocess import atomic_replace_bytes
    atomic_replace_bytes(output, page.encode("utf-8"))
    return output


__all__ = ["make_recolor_html"]
