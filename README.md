# AI Vector Cleanroom

繁體中文 | [English](#english)

把平面點陣圖 logo（含 AI 生成 logo）轉成「可編輯的 SVG 底稿」的清稿工具。

> 目前狀態：v0.3.0-alpha 技術預覽版。
> 適合少色、平面、邊界清楚的 logo 與 icon；其他類型的圖輸出仍需人工檢查與補修。
> 「省工 80%」尚未經真人實際編輯計時驗證（見下方誠實聲明）。

由 張進逸（Shinichi Chang）開發與維護。

## 解決什麼問題

AI 生成的 logo 只有 PNG，設計師拿到後常常只能整張重畫：
顏色髒（漸層雜訊、反鋸齒邊）、圓不圓、點不齊、線條變成一堆錨點的填色外框、
圖層全部黏在一起。

這個工具**不是「無損轉向量」**。點陣圖裡沒有原始貝茲曲線、字型、圖層資訊，
無損還原不存在。它做的是把重畫的起點從 0 分拉到高分底稿，並在每個環節
用外部渲染器逐像素把關、低品質就明確標示或判失敗，不把爛結果偽裝成完成品。

## 管線總覽

1. **去背與主色偵測**：加權 k-means++ 抓設計主色、剪掉反鋸齒混色假色，
   把漸層雜訊壓平成乾淨色塊。
2. **等寬線條重建**：心跳線、細線、框線這類等寬線稿，重建成真正的筆畫
   （中心線 path + `stroke-width` + 圓端點圓轉角），不再是高節點填色外框；
   細的反鋸齒線也不再變灰。
3. **漸層重建**：被壓成色帶的平滑漸層還原成真正的 `<linearGradient>`。
4. **原生幾何**：偵測到的正圓 → `<circle>`；同心圓環 → 一個帶 `stroke-dasharray`
   的 stroked circle；符合條件的直線／折線 → `<line>`／`<polyline>`。
5. **compound path 安全拆分**：只在拓撲可證明安全（不破壞洞／島、互不重疊）時
   才把大路徑拆成可單選的零件；精確有理數證明共線的 cubic 才化簡為直線。
6. **Scene Graph 後處理**：跨色零件只有在堆疊順序與像素完全不變時才寫成實體
   `<g>` 群組；不安全的候選只留在 manifest，不冒充可選群組。所有可見元件取得穩定 ID。
7. **品質閘門與候選回退**：低分結果自動比較關閉筆畫／漸層／幾何的候選、取最佳者；
   最佳仍低於 60 分直接判失敗。外觀與可編輯性分開評分。

每個後處理階段都有獨立 rollback，且用外部渲染器做**逐像素**驗證：
compound / Scene Graph / 原生線條若渲染有任何差異就撤回，只有近似的外環偵測
會通過「雙向 1px 容忍」的近似閘門。

## 已實現功能

1. PNG / JPG / WebP / BMP 批次轉為可編輯 SVG（純向量元素，不內嵌點陣圖）。
2. 真筆畫、原生 `<circle>` / `<line>` / `<polyline>`、真 `<linearGradient>`、
   透明度（`fill-opacity` / `stroke-opacity`）。
3. 依實際堆疊順序分層（Inkscape `groupmode="layer"`），可證明安全的跨色群組，
   每個元件有唯一穩定 ID。
4. **色彩調整頁**（`色彩調整.html`）：離線、按 paint-role 全域換色，OKLCH 保留
   同角色亮度／彩度關係，匯出的是明確 SVG 色值、不依賴專用色票。
5. **校稿工作台**（`review.html`）：100%–1600% 縮放平移、物件清單（含節點數與圖層
   開關）、點選高亮、問題熱區點擊跳轉。
6. **本機工作台**（`workbench.py`）：拖放轉檔、逐圖帶參數重跑（保留最多 8 版歷史）、
   結果清單、Stage 1 盲測頁與 Stage 2 實際編輯計時頁。只綁 `127.0.0.1`、POST 需
   session token、上傳做內容驗證。
7. 三分數自我驗證（`flat` / `source` / `foreground` 墨水 ROI，含雙向 1px 容忍）
   加上分層可編輯性稽核（`automation_readiness` / `redraw_complexity` /
   `workflow_friction`），寫進 `report.json`。
8. 批次安全：輸出名稱全域唯一、任一檔失敗 exit code 為 1、描不出東西視為失敗、
   關鍵寫入採原子替換（磁碟失敗不會留半份 XML）。

## 適用與不適用

適用：AI 生成／平面 logo、徽章、icon、標籤；色數有限（約 2–8 色）、邊界清楚；
圓形徽章元素（外圈、鉚釘、同心環、網點、速度線）。

不適用（分數會偏低，輸出僅供參考）：照片、寫實插畫、複雜柔和陰影／光暈、
極細紋理與髮絲線條、需要復原原始字型與可編輯文字、刻意手繪不規則風格
（可用 `--geometry off`）。

## 倉庫結構

```text
ai-vector-cleanroom/
  vector_cleanroom.py         主程式（批次流程、候選閘門、報告、打包）
  clean_base.py               核心引擎（調色、描邊、漸層、幾何規則化、分組）
  stroke_engine.py            等寬線條中心線重建
  trace_engine.py             去背與影像前處理
  svg_postprocess.py          四階段後處理協調（transaction / rollback）
  annulus_detector.py         共圓外環偵測 → 原生 circle
  compound_path_splitter.py   compound 安全拆分 + 精確 cubic 化簡
  exact_native_shapes.py      逐像素證明的 line / polyline 原生化
  scene_graph_postprocess.py  實體群組 / paint-order 不變式
  paint_roles.py              paint-role manifest
  recolor_page.py             離線全域換色頁
  editability_audit.py        分層可編輯性稽核
  designer_ops_audit.py       結構化設計操作驗收
  editing_test_page.py        Stage 2 實際編輯計時頁
  quality_diagnostics.py      前景 / 局部墨水格網品質指標
  workbench.py                本機拖放工作台（離線 UI）
  preflight_check.py          發版前私隱/二進位掃描
  tests/                      黑箱與單元測試（fixtures 由測試即時生成，不進 repo）
  .github/workflows/          CI（Ubuntu + Windows）
```

## 如何使用

需要 Python 3.10 以上。

```powershell
python -m pip install -r requirements.txt
# 選裝：SVG 預覽圖與逐像素自我驗證（缺了核心轉檔仍可跑）
python -m pip install -r requirements-preview.txt
```

Windows 也可直接雙擊 `install_deps.bat`。

把圖放進 `input/`，然後：

```powershell
python vector_cleanroom.py     # Windows 可雙擊 clean.bat
```

本機工作台（拖放、逐圖重跑、校稿、換色、盲測、Stage 2 計時）：

```powershell
python workbench.py            # Windows 可雙擊 workbench.bat
```

結果在 `output/result_<檔名>/`：`_vector.svg`、`_preview.png`、`source_reference.png`、
`review.html`、`色彩調整.html`、`report.json`、`OUTPUT_README.txt`，外加整包 zip。

## 選項

```text
--input DIR            輸入資料夾（預設 ./input）
--output DIR           輸出資料夾（預設 ./output）
--colors N             強制色數（預設 0 = 自動偵測）
--background MODE      auto | keep | transparent（預設 auto）
--white-threshold N    淺色背景判定門檻（預設 220）
--max-size N           描邊前將長邊縮到此尺寸（預設 2048；0 = 不縮）
--strokes on|off       等寬線條重建為真筆畫（預設 on）
--gradients on|off     色帶漸層重建為 linearGradient（預設 on）
--geometry LEVEL       conservative | normal | off（預設 conservative）
--debug                失敗時顯示完整 traceback
```

只有全部檔案成功，exit code 才是 0。

## 品質預期（誠實聲明）

- 平面少色 logo／徽章的 `source match` 通常落在 95%–99%；引用分數時務必標明
  參照（原圖 vs 壓色版）與解析度。
- **`foreground` 是墨水 ROI 指標**：白底不會稀釋分數；含雙向 1px 容忍，細線相位
  偏移不歸零，但線條真的消失仍會歸零。含白色設計元素、透明邊界的圖，透明底
  alpha ROI 會偏低（那是邊緣敏感的度量假象，不代表漏畫），對外請勿單取最漂亮的數字。
- **可編輯性是三條分開的軸**：`automation_readiness`（常用操作的自動化準備度）、
  `redraw_complexity`（自由手局部修形負擔）、`workflow_friction`（導航／選取摩擦）。
  結構把手 5/5 **不等於**真人任務 5/5。
- **「省工 80%」目前沒有真人證據**：Stage 1 盲測只驗證視覺品質與接手意願；
  真正的省工率必須由設計師在 Illustrator／Figma／Inkscape 實際編輯並計時（Stage 2）。
  本工具不宣稱「一鍵轉換任意圖片達 80% 以上」。

## 尚未完成（不應被誤報的能力）

- soft-alpha 真保留（複雜半透明／模糊／陰影仍近似）
- 完整 junction graph（一般 T/X/Y、相接同色物件仍非完整拓撲重建）
- 文字復原（無法可靠取回原字型與可編輯文字；目前是輪廓）
- 通用 primitive fitting（circle 支持強；任意 ellipse／圓角 rect／規則多邊形尚未全面）
- 真正原始設計語意的 Scene Graph（群組是可證明安全的幾何群組，非設計師原意）
- 逐元件重跑（工作台仍是逐圖重跑，有版本歷史但無 A/B 編輯分支）
- 跨編輯器 round-trip 尚未真人驗證

## 開發與測試

```powershell
python -m pip install -r requirements.txt -r requirements-preview.txt
python -m unittest discover -s tests -v
python preflight_check.py
```

測試 fixtures 全部由 `tests/generate_fixtures.py` 在測試中即時生成，倉庫**不含任何
二進位圖片**。CI 於 Ubuntu 與 Windows 上執行（見 `.github/workflows/ci.yml`）。

## 不要提交什麼

- 私人、客戶所有、含商標或授權不明的圖片，以及由其產出的 SVG。
- `input/`、`output/`、`tests/fixtures/` 的實際內容（.gitignore 已擋）。
- API 金鑰或任何憑證（本工具完全離線，不需要金鑰）。

## 作者、引用、授權

由 張進逸（Shinichi Chang）開發與維護。引用格式：

```text
AI Vector Cleanroom by 張進逸 (Shinichi Chang)
```

學術引用見 `CITATION.cff`。授權 MIT，見 `LICENSE`。
依賴授權：vtracer（MIT）、Pillow（MIT-CMU）、NumPy（BSD 系）；選用預覽依賴列於
`requirements-preview.txt`，二次散布前請自行確認授權。

---

# English

[繁體中文](#ai-vector-cleanroom) | English

A cleanup tool that turns flat bitmap logos (including AI-generated ones)
into editable SVG drafts.

> Status: v0.3.0-alpha, technical preview.
> Works well on flat, limited-palette logos and icons; everything else still
> needs human review and touch-up.
> The "80% time saving" claim is **not** yet backed by real designer editing
> timings (see the honesty note below).

Created and maintained by Shinichi Chang (張進逸).

## What problem does this solve

AI-generated logos ship as PNGs. Designers who receive them often have to
redraw the whole mark: dirty colors, circles that are not round, dots that
are not aligned, line work that is really a pile of filled outline anchors,
and every shape fused into one layer.

This is **not** lossless vectorization — bitmaps contain no original Bezier
curves, fonts, or layers. What the tool does is move the starting point of
the redraw from zero to a high-quality draft, gate every stage with an
external pixel-exact renderer, and clearly flag or fail low-quality results
instead of dressing them up as finished.

## Pipeline

1. **Background removal + palette detection** (weighted k-means++, AA-blend
   pruning, flatten gradient noise into clean colors).
2. **Monoline stroke reconstruction**: uniform-width line work becomes real
   strokes (center-line path + `stroke-width`, round caps/joins) instead of
   high-node filled outlines; thin AA lines keep their true color.
3. **Gradient reconstruction**: banded ramps become real `<linearGradient>`.
4. **Native geometry**: perfect circles → `<circle>`; concentric rings → one
   stroked dashed circle; eligible straight runs → `<line>` / `<polyline>`.
5. **Safe compound-path splitting**: large paths split into independently
   selectable parts only when topology (holes/islands, non-overlap) is
   provably safe; a cubic is rewritten to a line only when its control points
   are provably collinear via exact rationals.
6. **Scene Graph post-process**: cross-color parts become real `<g>` groups
   only when stack order and pixels are unchanged; unsafe candidates stay
   manifest-only. Every visible element gets a stable ID.
7. **Quality gate + candidate fallback**: low-scoring results compare
   strokes/gradients/geometry-off candidates and keep the best; below 60%
   they FAIL. Appearance and editability are scored separately.

Every post-process stage has an independent rollback and an external
**pixel-exact** render validator (compound / scene-graph / native lines roll
back on any render difference; only the approximate annulus stage passes a
bidirectional-1px gate).

## Implemented features

Real strokes, native `<circle>`/`<line>`/`<polyline>`, real
`<linearGradient>`, opacity; stack-order layers with unique stable IDs and
provably-safe cross-color groups; offline **recolor page** (paint-role
global recolor, OKLCH-preserving); **review workbench** (100–1600% zoom,
object list, hotspots); **local workbench** (drag-drop, per-image re-run with
up to 8 history versions, Stage 1 blind test, Stage 2 editing-time page,
bound to 127.0.0.1 with a session token); three-score self-check
(`flat`/`source`/`foreground` ink ROI) plus a three-axis editability audit;
batch-safe naming, exit-code-1 on failure, atomic writes.

## Good fit / poor fit

Good: AI-generated / flat logos, badges, icons, labels; limited palettes
(~2–8 colors) with clear boundaries; circular badge elements (rings, rivets,
concentric borders, halftone dots, speed lines).

Poor (expect low scores, reference only): photos, realistic illustrations,
complex soft shadows/glows, hairline texture, anything needing font/text
recovery, intentionally irregular hand-drawn styles (use `--geometry off`).

## Usage

Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-preview.txt   # optional: previews + pixel self-check
python vector_cleanroom.py                          # or double-click clean.bat
python workbench.py                                 # or workbench.bat — drag-drop UI
```

Results land in `output/result_<name>/`: `_vector.svg`, `_preview.png`,
`source_reference.png`, `review.html`, `色彩調整.html` (recolor), `report.json`,
`OUTPUT_README.txt`, plus a zip.

Options: `--input --output --colors --background {auto,keep,transparent}
--white-threshold --max-size --strokes {on,off} --gradients {on,off}
--geometry {conservative,normal,off} --debug`. Exit code is 0 only when every
file succeeded.

## Quality expectations (honest)

- Flat, limited-palette logos score 95–99% `source match`; always state the
  reference (source vs flattened) and the resolution when quoting a number.
- `foreground` is an **ink-ROI** score: an opaque background cannot dilute it;
  bidirectional 1 px tolerance keeps thin-stroke phase shifts from zeroing it,
  but a missing stroke still counts. Images with white design elements /
  transparent edges score lower on the transparent-alpha ROI — that is an
  edge-sensitive measurement artifact, not missing artwork; do not cherry-pick
  the prettiest number.
- Editability is **three separate axes** (`automation_readiness`,
  `redraw_complexity`, `workflow_friction`). A 5/5 structural handle count is
  **not** a 5/5 human task result.
- **The "80% time saving" claim has no human evidence yet.** Stage 1 blind
  testing only measures visual quality and willingness to take over; the real
  saving must be measured by designers actually editing in
  Illustrator/Figma/Inkscape and timing it (Stage 2). No claim of "one-click
  80%+ on arbitrary images."

## Not done yet (must not be over-reported)

soft-alpha preservation, full T/X/Y junction graph, text/font recovery,
general ellipse/rounded-rect/polygon primitives, true original-design scene
semantics, per-element re-run, and cross-editor round-trip validation with
real designers.

## Development

```powershell
python -m pip install -r requirements.txt -r requirements-preview.txt
python -m unittest discover -s tests -v
python preflight_check.py
```

Test fixtures are generated on the fly by `tests/generate_fixtures.py`; the
repository contains **no binary image assets**. CI runs on Ubuntu and Windows.

## Author, citation, license

Created and maintained by Shinichi Chang (張進逸). See `CITATION.cff`. MIT
licensed (`LICENSE`). Dependency licenses: vtracer (MIT), Pillow (MIT-CMU),
NumPy (BSD family); optional preview deps in `requirements-preview.txt`.
