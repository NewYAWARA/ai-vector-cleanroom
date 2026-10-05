from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from handoff_page import _STATE_JS, _display_svg, build_handoff_page


class HandoffPageTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema": "aivc.designer-handoff/v1",
            "svg_sha256": "a" * 64,
            "view_box": [0, 0, 200, 100],
            "objects": [
                {"id": "shape-a", "label": "主輪廓", "member_ids": ["path-a"],
                 "anchor_count": 7, "path_count": 1, "bbox": [0, 0, 40, 50],
                 "suggested_action": "keep", "reasons": ["輪廓穩定"]},
                {"id": "shape-b", "label": "細部", "member_ids": ["path-b"],
                 "anchor_count": 150, "path_count": 3, "bbox": [70, 10, 45, 50],
                 "suggested_action": "redraw", "reasons": ["節點密集"]},
            ],
            "default_decisions": {"shape-a": "keep", "shape-b": "review"},
        }
        self.svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
                    '<path id="path-a" d="M0 0L40 0L40 50Z" fill="#246543"/>'
                    '<path id="path-b" d="M70 10L115 10L115 60Z" fill="#ddab44"/>'
                    '</svg>')

    def page(self, **kwargs):
        return build_handoff_page(self.manifest, self.svg, "data:image/png;base64,YWJj",
                                  result_dir="result_demo", token="local-test", **kwargs)

    @staticmethod
    def data(page):
        match = re.search(r'<script id="handoff-data" type="application/json">(.*?)</script>',
                          page, flags=re.S)
        return json.loads(match.group(1))

    def test_contract_and_authoritative_server_decisions(self):
        data = self.data(self.page(saved_decisions={"shape-a": "redraw", "other": "keep"}))
        self.assertEqual(data["result"], "result_demo")
        self.assertEqual(data["svg_sha256"], "a" * 64)
        self.assertEqual(data["saved_decisions"], {"shape-a": "redraw"})
        self.assertEqual(self.data(self.page())["saved_decisions"], None)
        self.assertEqual(self.data(self.page(saved_decisions={}))["saved_decisions"], {})
        self.assertEqual(data["default_decisions"], {"shape-a": "keep", "shape-b": "review"})
        self.assertIsNone(data["saved_revision"])
        self.manifest["saved_revision"] = "revision-123"
        self.assertEqual(self.data(self.page())["saved_revision"], "revision-123")

    def test_source_audit_status_and_global_unassigned_holes_survive_page_payload(self):
        self.manifest['source_object_audit'] = {'status':'completed'}
        self.manifest['objects'][0]['source_audit_status'] = 'checked'
        self.manifest['scene_source_concerns'] = [
            {'label':'原本相連的部分斷開','x':12,'y':13,'w':3,'h':4}]
        data = self.data(self.page())
        self.assertEqual(data['source_audit_status'],'completed')
        self.assertEqual(data['objects'][0]['source_audit_status'],'checked')
        self.assertEqual(data['scene_source_concerns'],[
            {'label':'原本相連的部分斷開','bbox':[12,13,3,4]}])
        self.assertIn('這些是全圖的連通、孔洞或留白疑點',self.page())

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for highlight geometry tests')
    def test_highlight_stays_five_screen_pixels_outside_artwork_without_tint(self):
        page = self.page()
        body = re.search(r'function svgRect\(.*?\nfunction storageSave', page, re.S).group(0)
        body = body[:body.index('\nfunction storageSave')]
        script = r'''
const assert=require('node:assert/strict');let scale=1;const ns='svg';
const document={createElementNS:()=>({attributes:{},setAttribute(k,v){this.attributes[k]=v;}})};
''' + body + r'''
const geometry=[10,20,80,.2],snapshot=[...geometry];
for(const zoom of [.02,.25,1,4,40]){
 scale=zoom;const r=svgRect(geometry,'selection-box').attributes;
 assert.ok(Math.abs((geometry[0]-r.x)*scale-5)<1e-9);
 assert.ok(Math.abs((geometry[1]-r.y)*scale-5)<1e-9);
 assert.ok(Math.abs((r.width-geometry[2])*scale-10)<1e-9);
 assert.ok(Math.abs((r.height-geometry[3])*scale-10)<1e-9);
 assert.equal(r.fill,'none');assert.deepEqual(geometry,snapshot);
}
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf8',
                             capture_output=True, timeout=20)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('.overlay .selection-box{fill:none;',page)
        self.assertIn('.overlay .hover-box{fill:none;',page)
        layout = re.search(r'function layout\(\).*?\nfunction fit',page,re.S).group(0)
        self.assertIn('drawOverlays();',layout)

    def test_script_payload_cannot_close_script_or_inject_markup(self):
        attack = '</script><img src=x onerror="alert(1)">\u2028'
        self.manifest["objects"][0]["label"] = attack
        self.manifest["objects"][0]["reasons"] = [attack]
        page = build_handoff_page(self.manifest, self.svg, '" onerror="alert(2)',
                                  result_dir=attack, token=attack)
        self.assertNotIn('<img src=x', page)
        self.assertNotIn('src="" onerror=', page)
        self.assertEqual(page.count('</script>'), 2)
        data = self.data(page)
        self.assertEqual(data["objects"][0]["label"], attack)
        self.assertEqual(data["token"], attack)
        self.assertEqual(data["source_data_url"], "")

    def test_svg_preview_blocks_active_and_remote_content_but_keeps_paths(self):
        raw = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                   xmlns:html="http://www.w3.org/1999/xhtml" onload="alert(1)" xml:base="https://bad.example/" viewBox="0 0 10 10">
          <defs><linearGradient id="paint"><stop offset="0" stop-color="red"/></linearGradient></defs>
          <script>alert(2)</script><foreignObject><html:div>bad</html:div></foreignObject>
          <style>@import 'https://bad.example/a.css';</style>
          <a href="https://bad.example/"><script>alert(3)</script><path id="p" d="M0 0L10 10" fill="url(#paint)" onclick="alert(4)"/></a>
          <image href="https://bad.example/a.png"/><use xlink:href="#p"/>
          <path id="q" d="M1 1L2 2" style="fill:url(https://bad.example/x)"/>
          <animate attributeName="href" to="https://bad.example/a"/>
        </svg>'''
        svg, box = _display_svg(raw)
        self.assertEqual(box, [0, 0, 10, 10])
        self.assertNotIn('bad.example', svg)
        self.assertNotIn('alert(', svg)
        self.assertNotIn('foreignObject', svg)
        self.assertNotIn('<script', svg)
        self.assertNotIn('<animate', svg)
        root = ET.fromstring(svg)
        path = next(node for node in root.iter() if node.get('data-source-id') == 'p')
        self.assertEqual(path.get('d'), 'M0 0L10 10')
        self.assertTrue(path.get('fill').startswith('url(#handoff-shape-'))
        self.assertNotEqual(path.get('id'), 'p')
        self.assertTrue(any(node.tag.endswith('use') and any(v.startswith('#handoff-shape-')
                            for v in node.attrib.values()) for node in root.iter()))

    def test_xml_entities_rejected_and_invalid_svg_has_clear_error(self):
        for svg in ['<!DOCTYPE svg [<!ENTITY x "bad">]><svg>&x;</svg>', '<svg>', '<html/>']:
            with self.subTest(svg=svg), self.assertRaises(ValueError):
                _display_svg(svg)

    def test_bad_bbox_and_counts_are_normalized(self):
        self.manifest["view_box"] = [0, 0, float('inf'), 100]
        self.manifest["objects"][0]["bbox"] = [0, 0, float('nan'), 4]
        self.manifest["objects"][0]["anchor_count"] = -4
        self.manifest["objects"][1]["path_count"] = "invalid"
        data = self.data(self.page())
        self.assertEqual(data["view_box"], [0, 0, 200, 100])
        self.assertIsNone(data["objects"][0]["bbox"])
        self.assertEqual(data["objects"][0]["anchor_count"], 0)
        self.assertEqual(data["objects"][1]["path_count"], 0)

    def test_processed_reference_is_not_presented_as_original(self):
        self.manifest["reference_kind"] = "processed_reference"
        page = self.page()
        self.assertEqual(self.data(page)["reference_kind"], "processed_reference")
        self.assertIn('清理後參考圖（非原始圖片）', page)
        self.assertIn('此結果只有清理後參考圖，並非原始圖片', page)
        self.manifest["reference_kind"] = "original"
        self.assertEqual(self.data(self.page())["reference_kind"], "original")
        del self.manifest["reference_kind"]
        self.assertEqual(self.data(self.page())["reference_kind"], "processed_reference")

    def test_local_storage_fingerprint_and_refine_contract_in_page(self):
        page = self.page()
        self.assertIn("'aivc.handoff.v1:' + data.svg_sha256 + ':' + data.result", page)
        self.assertIn("resolveHandoffDraft(data,localRecord)", page)
        self.assertIn("baseRevision", page)
        self.assertIn("取回本機草稿，重新核對", page)
        self.assertIn("下載保留草稿 JSON", page)
        self.assertIn("'X-WB-Token':data.token", page)
        self.assertIn("revision:currentRevision", page)
        self.assertIn("request('refine',{object_ids:ids,error_budget_percent:", page)
        self.assertIn("[...selected].every(id=>state.decisions[id]!=='keep')", page)
        self.assertIn('只看已採用', page)
        self.assertIn("request('prepare',{error_budget_percent:", page)
        self.assertIn('仍有 ${counts.review} 個待確認', page)
        self.assertNotIn('<script src=', page)
        self.assertNotIn('<link ', page)

    def test_preparation_summary_survives_navigation_without_script_injection(self):
        self.manifest['auto_prepare'] = {'summary': {'anchors_before': 120, 'anchors_after': 12},
                                        'units': [{'id': '</script><img onerror=x>', 'reason': 'test'}]}
        page = self.page()
        self.assertEqual(self.data(page)['auto_prepare'], self.manifest['auto_prepare'])
        self.assertEqual(page.count('</script>'), 2)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to execute browser-state tests')
    def test_batched_decisions_undo_redo_and_invalid_ids(self):
        script = _STATE_JS + r'''
const assert = require('node:assert/strict');
const objects=[{id:'a'},{id:'b'},{id:'__proto__'}];
const s=new HandoffState(objects, {a:'keep',b:'review'}, {a:'redraw',b:'bad',other:'keep'});
assert.equal(s.decisions.a,'redraw'); assert.equal(s.decisions.b,'review');
assert.equal(s.decisions['__proto__'],'review');
assert.equal(s.apply(['a','b','a','missing'],'keep'),true);
assert.deepEqual(s.counts(),{keep:2,review:1,redraw:0});
assert.equal(s.history.length,1); assert.equal(s.revision,1);
assert.equal(s.apply(['a'],'keep'),false); assert.equal(s.history.length,1);
assert.equal(s.apply(['a'],'unknown'),false);
assert.equal(s.undo(),true); assert.equal(s.decisions.a,'redraw');assert.equal(s.decisions.b,'review');
assert.equal(s.redo(),true);assert.equal(s.decisions.a,'keep');
s.undo();s.apply(['b'],'redraw');assert.equal(s.redo(),false);
assert.equal(s.decisions.a,'redraw');assert.equal(s.decisions.b,'redraw');
const emptyServer=new HandoffState(objects,{a:'keep',b:'review'},{});
assert.equal(emptyServer.decisions.a,'keep');
console.log('handoff state transitions passed');
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('state transitions passed', run.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for JavaScript syntax validation')
    def test_generated_javascript_parses(self):
        page = self.page()
        code = re.search(r'<script>\s*(.*?)</script>', page, re.S).group(1)
        run = subprocess.run([shutil.which('node'), '--check'], input=code, text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for selection boundary tests')
    def test_real_js_marquee_contains_complete_bounds_in_both_drag_directions(self):
        script = _STATE_JS + r'''
const assert=require('node:assert/strict');
const box=[100,100,15,20];
assert.equal(handoffBoundsContained(box,[98,98],[117,122]),true);
assert.equal(handoffBoundsContained(box,[117,122],[98,98]),true);
assert.equal(handoffBoundsContained(box,[100,100],[115,120]),true); // inclusive boundary
assert.equal(handoffBoundsContained([0,0,1000,1000],[98,98],[117,122]),false);
assert.equal(handoffBoundsContained([114,105,12,10],[98,98],[117,122]),false); // crossing edge
assert.equal(handoffBoundsContained([102,102,0,18],[98,98],[117,122]),true); // vertical line
assert.equal(handoffBoundsContained(null,[98,98],[117,122]),false);
assert.equal(handoffBoundsContained([100,100,-2,2],[98,98],[117,122]),false);
assert.equal(handoffBoundsContained([100,100,Infinity,2],[98,98],[117,122]),false);
console.log('complete bounds passed');
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('complete bounds passed', run.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for DOM selection event tests')
    def test_real_pointer_mode_and_shortcut_handlers_never_operate_on_hidden_selection(self):
        page = self.page()
        # Execute the generated handlers themselves in a bounded DOM event harness.
        # Stubs supply layout and unrelated rendering; selection and decisions are real.
        selectable = re.search(r'function selectableObject\(id\)\{[^\n]+', page).group(0)
        reconcile = re.search(r'function reconcileVisibleSelection\(\)\{.*?\n\}', page, re.S).group(0)
        overlays = re.search(r'function drawOverlays\(\) \{.*?\n\}', page, re.S).group(0)
        selection = re.search(r'function renderSelection\(\) \{.*?\n\}', page, re.S).group(0)
        interaction = page[page.index('for(const vp of viewports){'):
                           page.index("for(const button of document.querySelectorAll('[data-action]'))button.onclick")]
        keyboard = page[page.index("document.addEventListener('keydown',event=>"):
                        page.index("document.addEventListener('keyup',event=>")]
        decision = re.search(r'function decision\(action\)\{[^\n]+', page).group(0)
        list_filter = re.search(r' visibleObjects=objects.filter\([^\n]+', page).group(0)
        script = _STATE_JS + r'''
const assert=require('node:assert/strict');
class Node {
 constructor(){this.style={};this.dataset={};this.attrs={};this.children=[];this.handlers={};this.classes=new Set();
  this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name),remove:name=>this.classes.delete(name)};}
 addEventListener(name,handler){this.handlers[name]=handler;}
 setAttribute(name,value){this.attrs[name]=value;}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;}
 querySelector(selector){if(selector==='.marquee')return this.marquee||(this.marquee=new Node());return null;}
 querySelectorAll(){return [];}
 getBoundingClientRect(){return {left:0,top:0};}
 setPointerCapture(){this.captured=true;}
 hasPointerCapture(){return !!this.captured;}
 releasePointerCapture(){this.captured=false;}
}
const ids=['sourceViewport','vectorViewport','canvasStage','objectList','selectionCount','selectedCaption','decisionTitle','clearSelection','selectionDetail'];
const nodes=Object.fromEntries(ids.map(id=>[id,new Node()]));
const $=id=>nodes[id];
const viewButtons=['compare','vector','adopted'].map(view=>{const n=new Node();n.dataset.view=view;return n;});
const actionButtons=['keep','review','redraw'].map(action=>{const n=new Node();n.dataset.action=action;return n;});
const events={};
const document={querySelectorAll:selector=>selector==='[data-view]'?viewButtons:actionButtons,
 addEventListener:(name,fn)=>events[name]=fn};
const objects=[{id:'background',bbox:[0,0,1000,1000]},
 {id:'small-letter',bbox:[100,100,15,20]}, {id:'crossing-letter',bbox:[114,105,12,10]},
 {id:'kept',bbox:[140,100,20,20]}, {id:'unlocated',bbox:null}].map(o=>({...o,anchor_count:4,reasons:[],suggested_action:'review'}));
const byId=new Map(objects.map(o=>[o.id,o]));
const state=new HandoffState(objects,{}, {background:'review','small-letter':'review','crossing-letter':'review',kept:'keep',unlocated:'review'});
const labels={keep:'採用',review:'待確認',redraw:'交人工'};
let mode='compare', selected=new Set(), hovered=null, visibleObjects=[], drag=null;
let centerX=0,centerY=0,scale=1,panMode=false,spaceHeld=false,fitCount=0;
const viewports=[$('sourceViewport'),$('vectorViewport')],overlays=[new Node(),new Node()];
function element(tag,cls,text){const node=new Node();node.textContent=text;return node;}
function svgRect(box,cls){return {box,cls};}
function screenToWorld(vp,x,y){return [x,y];}
function hitObjects(event){return event.hit||[];}
function zoom(){} function layout(){} function focusObjects(){} function updateRefineButton(){}
function updateVisibility(){} function setPan(){} function save(){} function redo(){} function undo(){}
function fit(){fitCount++;}
function afterChange(){listObjects();renderSelection();}
''' + selectable + reconcile + overlays + selection + decision + r'''
function listObjects(){const filter='all',search='';
''' + list_filter + r'''
}
''' + interaction + keyboard + r'''
function pointer(vp,name,x,y,extra={}){vp.handlers[name]({button:0,pointerId:1,clientX:x,clientY:y,
 preventDefault(){},...extra});}
function marquee(start,end,extra={}){const vp=$('vectorViewport');pointer(vp,'pointerdown',...start,extra);
 pointer(vp,'pointermove',...end);pointer(vp,'pointerup',...end);assert.equal(vp.querySelector('.marquee').style.display,'none');}
marquee([98,98],[117,122]);assert.deepEqual([...selected],['small-letter']);
assert.equal($('selectionCount').textContent,'已選 1 個');
assert.deepEqual(overlays[1].children.map(node=>node.box),[[100,100,15,20]]);
marquee([117,122],[98,98]);assert.deepEqual([...selected],['small-letter']);
selected=new Set(['kept']);marquee([98,98],[117,122],{shiftKey:true});
assert.deepEqual([...selected],['kept','small-letter']); // additive without background
hovered='crossing-letter';viewButtons[2].onclick();
assert.equal(mode,'adopted');assert.deepEqual([...selected],['kept']);assert.equal(hovered,null);
assert.deepEqual(visibleObjects.map(o=>o.id),['kept']);
assert.equal(viewButtons[2].attrs['aria-pressed'],'true');
assert.equal($('canvasStage').classes.has('single'),true);
assert.equal($('selectionCount').textContent,'已選 1 個');
assert.deepEqual(overlays[1].children.map(node=>node.box),[[140,100,20,20]]);
// Adopted preview cannot reselect hidden candidates even with a full-canvas marquee.
marquee([0,0],[1000,1000]);assert.deepEqual([...selected],['kept']);
function key(value){events.keydown({key:value,code:'Digit'+value,target:{matches:()=>false},preventDefault(){}});}
key('3');assert.equal(state.decisions.kept,'redraw');assert.equal(selected.size,0);
assert.equal($('selectionCount').textContent,'未選取');assert.equal(visibleObjects.length,0);
assert.equal(overlays[1].children.length,0);assert.equal(actionButtons.every(b=>b.disabled),true);
key('1');assert.equal(state.decisions.kept,'redraw');assert.equal(state.decisions['small-letter'],'review');
// Returning to the complete view neither resurrects stale selection nor loses candidates.
viewButtons[0].onclick();assert.equal(selected.size,0);assert.equal(visibleObjects.length,5);
assert.equal($('canvasStage').classes.has('single'),false);
selected.add('small-letter');hovered='small-letter';viewButtons[2].onclick();
assert.equal(selected.size,0);assert.equal(hovered,null);key('1');
assert.equal(state.decisions['small-letter'],'review');assert.equal(fitCount,3);
console.log('pointer, preview DOM, and hidden shortcut behavior passed');
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('hidden shortcut behavior passed', run.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for filtered selection tests')
    def test_select_current_list_replaces_old_filtered_selection_before_batch_decision(self):
        page = self.page()
        selectable = re.search(r'function selectableObject\(id\)\{[^\n]+', page).group(0)
        reconcile = re.search(r'function reconcileVisibleSelection\(\)\{.*?\n\}', page, re.S).group(0)
        list_filter = re.search(r' visibleObjects=objects.filter\([^\n]+', page).group(0)
        decision = re.search(r'function decision\(action\)\{[^\n]+', page).group(0)
        handler = re.search(r"\$\('selectVisible'\).onclick=.*?;\$\('clearSelection'\)", page).group(0)
        handler = handler.split(";$('clearSelection')")[0] + ';'
        script = _STATE_JS + r'''
const assert=require('node:assert/strict');
const objects=[{id:'complex',label:'complex object',suggested_action:'redraw',reasons:[]},
 {id:'simple',label:'simple object',suggested_action:'keep',reasons:[]},
 {id:'other',label:'other object',suggested_action:'keep',reasons:[]}];
const state=new HandoffState(objects,{complex:'review',simple:'review',other:'review'},null);
const selected=new Set(['complex']);let hovered=null,mode='compare',visibleObjects=[],filter='all',search='';
const nodes={selectVisible:{}};const $=id=>nodes[id];let selectionRenders=0;
function renderSelection(){selectionRenders++;reconcileVisibleSelection();}
function afterChange(){listObjects();renderSelection();}
''' + selectable + reconcile + decision + r'''
function listObjects(){
''' + list_filter + '\n}\n' + handler + r'''
// The existing explicit selection may persist while searching. The button
// promises selection of the current list, so it must replace that selection.
filter='suggest-keep';listObjects();renderSelection();
assert.deepEqual(visibleObjects.map(o=>o.id),['simple','other']);
assert.deepEqual([...selected],['complex']);
$('selectVisible').onclick();
assert.deepEqual([...selected],['simple','other']);decision('keep');
assert.equal(state.decisions.complex,'review');
assert.equal(state.decisions.simple,'keep');assert.equal(state.decisions.other,'keep');
assert.equal(state.undo(),true);
assert.deepEqual(state.decisions,{complex:'review',simple:'review',other:'review'});
// Search narrowing is equally safe: hidden prior list entries are excluded.
search='simple';listObjects();$('selectVisible').onclick();decision('redraw');
assert.deepEqual([...selected],['simple']);
assert.deepEqual(state.decisions,{complex:'review',simple:'redraw',other:'review'});
// An empty list cannot leave a stale batch behind, even if invoked directly.
search='nothing';listObjects();$('selectVisible').onclick();
assert.equal(selected.size,0);decision('keep');assert.equal(state.decisions.simple,'redraw');
assert.ok(selectionRenders>=6);
console.log('current filtered list selection replaces hidden prior selection');
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('replaces hidden prior selection', run.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for request contract tests')
    def test_request_revision_advances_only_on_success_and_conflict_keeps_decisions(self):
        page = self.page()
        request = re.search(r'async function request\(kind,extra=\{\}\)\{.*?\n\}', page, re.S).group(0)
        script = r'''
const assert=require('node:assert/strict');
const data={result:'demo',svg_sha256:'abc',token:'t'};
const state={decisions:{a:'keep',b:'review'}};
let currentRevision=null, responses=[], calls=[];
let storageWrites=[];
function storageSave(){storageWrites.push({baseRevision:currentRevision,decisions:{...state.decisions}});}
async function fetch(url,options){calls.push({url,headers:options.headers,body:JSON.parse(options.body)});const reply=responses.shift();return {ok:reply.status===200,status:reply.status,json:async()=>reply.data};}
''' + request + r'''
(async()=>{
 responses.push({status:200,data:{revision:'r1'}});
 await request('save');assert.equal(calls[0].body.revision,null);assert.equal(currentRevision,'r1');
 assert.equal(storageWrites[0].baseRevision,'r1');assert.deepEqual(storageWrites[0].decisions,state.decisions);
 responses.push({status:200,data:{revision:'r2',files:[]}});
 await request('export');assert.equal(calls[1].body.revision,'r1');assert.equal(currentRevision,'r2');
 assert.equal(storageWrites[1].baseRevision,'r2');
 responses.push({status:409,data:{error:'stale revision'}});
 await assert.rejects(request('save'),/stale revision/);assert.equal(currentRevision,'r2');
 assert.equal(storageWrites.length,2);
 assert.deepEqual(state.decisions,{a:'keep',b:'review'});
 responses.push({status:200,data:{url:'/new',revision:'new-result'}});
 await request('refine',{object_ids:['b'],error_budget_percent:.25});
 assert.equal(calls[3].body.revision,'r2');assert.deepEqual(calls[3].body.object_ids,['b']);
 assert.equal(currentRevision,'r2');assert.equal(calls[3].headers['X-WB-Token'],'t');
 responses.push({status:200,data:{}});
 await assert.rejects(request('save'));assert.equal(currentRevision,'r2');
 console.log('revision contract passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('revision contract passed', run.stdout)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for draft recovery tests')
    def test_real_js_draft_recovery_conflicts_manual_restore_and_refine_return(self):
        script = _STATE_JS + r'''
const assert=require('node:assert/strict');
const data={objects:[{id:'a'},{id:'b'}],default_decisions:{a:'review',b:'review'},
 saved_decisions:{a:'review',b:'keep'},saved_revision:'r1',svg_sha256:'parent-sha'};
const local=makeHandoffDraft('parent-sha','r1',{a:'redraw',b:'keep'});
// A draft based on the current server revision restores even after an earlier save.
const same=resolveHandoffDraft(data,local);
assert.deepEqual(same.initial,{a:'redraw',b:'keep'});assert.equal(same.recovered,true);
assert.equal(same.serverSnapshot,JSON.stringify(data.saved_decisions));
assert.notEqual(JSON.stringify(same.initial),same.serverSnapshot); // must not display "saved"
// An identical local copy is still accurately marked saved.
const unchanged=resolveHandoffDraft(data,makeHandoffDraft('parent-sha','r1',data.saved_decisions));
assert.equal(unchanged.recovered,false);assert.equal(JSON.stringify(unchanged.initial),unchanged.serverSnapshot);
// A different server revision never silently adopts the draft.
const newer={...data,saved_revision:'r2',saved_decisions:{a:'keep',b:'redraw'}};
const conflict=resolveHandoffDraft(newer,local);
assert.deepEqual(conflict.initial,newer.saved_decisions);assert.equal(conflict.recovered,false);
assert.deepEqual(conflict.pending,local);assert.equal(conflict.backups.length,1);
// Writing current state must preserve the conflicting draft through another reload.
const envelope={...makeHandoffDraft('parent-sha','r2',conflict.initial),
 backups:conflict.backups,pendingDraft:conflict.pending};
const second=resolveHandoffDraft(newer,JSON.parse(JSON.stringify(envelope)));
assert.deepEqual(second.initial,newer.saved_decisions);assert.deepEqual(second.pending,local);
// Explicit manual restore is a single undoable action and does not change server revision.
const state=new HandoffState(data.objects,data.default_decisions,newer.saved_decisions);
assert.equal(restorableHandoffDraft(newer,second.pending),true);
state.restore(second.pending.decisions);assert.deepEqual(state.decisions,local.decisions);
assert.equal(state.history.length,1);assert.equal(state.undo(),true);
assert.deepEqual(state.decisions,newer.saved_decisions);
// The user's prior SVG can be reopened after a successful refine redirect.
// Refine keeps the parent's server revision; the new derived page has its own key/state.
const parentKey='aivc.handoff.v1:parent-sha:result_parent';
const derivedKey='aivc.handoff.v1:derived-sha:result_derived';
const storage={[parentKey]:JSON.stringify(local)};
const derived={...data,svg_sha256:'derived-sha',saved_revision:'derived-r1'};
const derivedOpen=resolveHandoffDraft(derived,JSON.parse(storage[derivedKey]||'null'));
assert.deepEqual(derivedOpen.initial,derived.saved_decisions);
const returnToParent=resolveHandoffDraft(data,JSON.parse(storage[parentKey]));
assert.deepEqual(returnToParent.initial,local.decisions);assert.equal(returnToParent.recovered,true);
// Unknown legacy revisions are preserved for explicit review, never auto-applied.
const legacy=resolveHandoffDraft(data,{svg_sha256:'parent-sha',decisions:local.decisions});
assert.deepEqual(legacy.initial,data.saved_decisions);assert.equal(legacy.pending.legacy_revision_unknown,true);
assert.equal(legacy.backups.length,1);
// Wrong fingerprint and incomplete decisions may be downloaded, not restored.
assert.equal(restorableHandoffDraft(data,{...local,svg_sha256:'other'}),false);
assert.equal(restorableHandoffDraft(data,{...local,decisions:{a:'keep'}}),false);
const invalid=resolveHandoffDraft(data,{unreadable_raw:'broken JSON'});
assert.deepEqual(invalid.backups,[{unreadable_raw:'broken JSON'}]);
// No-server first use still accepts only a draft with explicit null base revision.
const first={...data,saved_revision:null,saved_decisions:null};
assert.equal(resolveHandoffDraft(first,makeHandoffDraft('parent-sha',null,local.decisions)).recovered,true);
// A different tab's unsaved draft is backed up before writing the shared key.
const otherTab={...makeHandoffDraft('parent-sha','r1',{a:'keep',b:'keep'}),writerId:'other-tab',backups:[]};
const backups=[];
const pending=mergeOtherHandoffDraft(data,otherTab,'this-tab',local.decisions,backups,null);
assert.deepEqual(pending.decisions,otherTab.decisions);assert.equal(backups.length,1);
mergeOtherHandoffDraft(data,otherTab,'this-tab',local.decisions,backups,pending);
assert.equal(backups.length,1); // no duplicate backup on every keystroke
assert.equal(mergeOtherHandoffDraft(data,{...otherTab,writerId:'this-tab'},'this-tab',local.decisions,[],null),null);
console.log('draft recovery and refine return passed');
'''
        run = subprocess.run([shutil.which('node'), '-e', script], text=True, encoding='utf-8',
                             capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('draft recovery and refine return passed', run.stdout)


if __name__ == '__main__':
    unittest.main()
