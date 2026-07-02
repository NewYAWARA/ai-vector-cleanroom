# AI Vector Cleanroom

繁體中文 | [English](#english)

把平面點陣圖 logo（含 AI 生成 logo）轉成「可編輯的 SVG 底稿」的清稿工具。

> 目前狀態：v0.1.0-alpha 技術預覽版。
> 適合少色、平面、邊界清楚的 logo 與 icon；其他類型的圖輸出仍需人工檢查與補修。

由 張進逸（Shinichi Chang）開發與維護。

## 解決什麼問題

AI 生成的 logo 只有 PNG，設計師拿到後常常只能整張重畫：
顏色髒（漸層雜訊、反鋸齒邊）、圓不圓、點不齊、圖層全部黏在一起。

這個工具不是「無損轉向量」——點陣圖裡沒有原始貝茲曲線、字型、圖層資訊，
無損還原不存在。它做的是把重畫的起點從 0 分拉到高分底稿：

1. 自動抓出設計主色，把漸層雜訊、反鋸齒邊壓平成乾淨色塊。
2. 依「實際堆疊順序」分成顏色群組：點一下選整組，雙擊可改單一形狀，
   上層物件（例如鉚釘）不會被合併蓋掉。
3. 幾何規則化：畫歪的圓换成數學正圓、鉚釘統一大小並對齊、
   大圓自動同心（normal 等級再把圓環邊緣換成真圓弧）。
4. 產出校稿頁與誠實的品質分數，讓接手的人知道哪裡要修。

## 已實現功能

1. PNG / JPG / WebP / BMP 批次轉為可編輯 SVG（純 path，不內嵌點陣圖）。
2. 主色自動偵測：對唯一顏色做加權 k-means++，並剪掉「反鋸齒混色」
   假色；`--colors N` 可強制色數。
3. 依堆疊順序分組；同色形狀只有在互不遮蓋時才合併，渲染保證不變。
4. 幾何規則化三檔：`conservative`（預設）／`normal`／`off`。
5. 自我驗證雙分數：`flat match`（對壓色版的還原度）與
   `source match`（對原圖的近似度，**對外請引用這個**）；低於 90% 會警告。
6. 每張圖產出：SVG、預覽圖、原圖參考、疊圖校稿頁 review.html、
   機器可讀 report.json、輸出說明、整包 zip。
7. 批次安全：同名不同副檔名不互相覆蓋；任一檔失敗 exit code 為 1；
   檔名含 `&` 等特殊字元仍輸出合法 SVG。

## 適用與不適用

適用：

- AI 生成 logo、平面標誌、徽章、icon、標籤
- 色數有限（約 2–8 色）、邊界清楚的圖
- 圓形徽章元素：外圈、鉚釘、同心環

不適用（分數會偏低，輸出僅供參考）：

- 照片、寫實插畫
- 漸層、柔和陰影、光暈（會被壓成單色）
- 極細的紋理與髮絲線條
- 刻意手繪、故意不規則的風格（請用 `--geometry off`）

## 倉庫結構

```text
ai-vector-cleanroom/
  vector_cleanroom.py   主程式（批次流程、報告、打包）
  clean_base.py         引擎（調色、描邊、幾何規則化、分組）
  trace_engine.py       去背與影像前處理
  clean.bat             Windows 雙擊入口
  install_deps.bat      Windows 一鍵裝依賴
  input/                把圖丟這裡
  output/               結果出現在這裡
  tests/                合成圖測試（16 個，不含任何二進位素材）
  .github/workflows/    CI（Ubuntu + Windows）
```

## 如何使用

需要 Python 3.10 以上。

```powershell
python -m pip install -r requirements.txt
# 選裝：SVG 預覽圖與自我驗證分數
python -m pip install -r requirements-preview.txt
```

Windows 也可以直接雙擊 `install_deps.bat`。

把圖放進 `input/`，然後：

```powershell
python vector_cleanroom.py
```

Windows 可直接雙擊 `clean.bat`。結果在 `output/`：

```text
output/
  result_<檔名>/
    <檔名>_vector.svg        可編輯 SVG（依顏色/圖層分組）
    <檔名>_preview.png       由 SVG 渲染的預覽圖*
    source_reference.png     去背後的原圖參考
    review.html              瀏覽器疊圖校稿頁
    report.json              機器可讀報告
    OUTPUT_README.txt        給接手者的說明
  result_<檔名>.zip
```

\* 未安裝選用套件時，預覽圖會退回使用去背原圖，並在 console、
report.json、OUTPUT_README.txt 三處明確標示「非 SVG 渲染結果」。

## 選項

```text
--input DIR            輸入資料夾（預設 ./input）
--output DIR           輸出資料夾（預設 ./output）
--colors N             強制色數（預設 0 = 自動偵測）
--background MODE      auto | keep | transparent（預設 auto）
                       auto 會移除「連到圖片邊緣的淺色背景」；
                       若貼邊的淺色設計元素被誤刪，請改用 keep
--white-threshold N    淺色背景判定門檻（預設 220）
--max-size N           描邊前將長邊縮到此尺寸（預設 2048；0 = 不縮）
--geometry LEVEL       conservative | normal | off（預設 conservative）
                       conservative：正圓化、鉚釘對齊、大圓同心
                       normal：另外把圓環邊緣換成數學圓弧
                       off：保留原始描邊
```

只有全部檔案成功，exit code 才是 0。

## 品質預期（誠實聲明）

用內建像素比對（單通道差 < 48、白底渲染）量測：

- 平面少色 logo／徽章，`source match` 通常落在 95–99%。
- 有漸層陰影的圖，`flat match` 可能很高、`source match` 掉到 85–90%——
  這個落差是真實資訊：代表細節被壓平了。對外只能引用 `source match`。

本工具不宣稱「一鍵轉換任意圖片達 90% 以上」。

## 開發與測試

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest tests/ -q
python preflight_check.py
```

測試素材全部由 Pillow 在測試中即時生成，倉庫不含任何二進位圖片。
CI 於 Ubuntu 與 Windows 上執行（見 `.github/workflows/ci.yml`）。

## 不要提交什麼

- 私人、客戶所有、含商標或授權不明的圖片。
- 由上述圖片產出的 SVG——輸出繼承原圖的法律限制。
- `input/`、`output/` 的實際內容（.gitignore 已擋，請勿硬加）。
- API 金鑰或任何憑證（本工具完全離線，不需要金鑰）。

## 作者與引用

由 張進逸（Shinichi Chang）開發與維護。引用格式：

```text
AI Vector Cleanroom by 張進逸 (Shinichi Chang)
```

學術引用見 `CITATION.cff`。授權 MIT，見 `LICENSE`。

依賴授權：vtracer（MIT）、Pillow（MIT-CMU）、NumPy（BSD 系）。
選用預覽依賴列於 `requirements-preview.txt`，二次散布前請自行確認授權。

---

# English

[繁體中文](#ai-vector-cleanroom) | English

A cleanup tool that turns flat bitmap logos (including AI-generated ones)
into editable SVG drafts.

> Status: v0.1.0-alpha, technical preview.
> Works well on flat, limited-palette logos and icons; everything else
> still needs human review and touch-up.

Created and maintained by Shinichi Chang (張進逸).

## What problem does this solve

AI-generated logos ship as PNGs. Designers who receive them often have to
redraw the whole mark: dirty colors (gradient noise, antialiased edges),
circles that are not round, dots that are not aligned, and every shape
fused into one layer.

This is **not** lossless vectorization — bitmaps contain no original Bezier
curves, fonts, or layers, so lossless recovery does not exist. What the tool
does is move the starting point of the redraw from zero to a high-quality
draft:

1. Detects the design palette and flattens gradient/antialiasing noise into
   clean solid colors.
2. Groups shapes by their actual stacking order: one click selects a group,
   double-click edits a single shape, and top-layer objects (e.g. rivets)
   are never merged underneath and hidden.
3. Geometry regularization: wobbly circles become mathematical circles,
   rivet dots get unified size and alignment, large circles become
   concentric (the `normal` level also straightens ring edges into true
   arcs).
4. Produces an overlay review page and honest quality scores so the person
   taking over knows exactly what to fix.

## Implemented features

1. Batch conversion of PNG / JPG / WebP / BMP into editable SVG
   (paths only, no embedded bitmaps).
2. Automatic palette detection: weighted k-means++ over unique colors with
   antialiasing-blend pruning; `--colors N` forces a fixed palette size.
3. Stack-order grouping; same-color shapes merge only when nothing in
   between overlaps, so rendering never changes.
4. Three geometry levels: `conservative` (default) / `normal` / `off`.
5. Dual self-check scores: `flat match` (fidelity to the flattened tracing
   input) and `source match` (similarity to the source — **quote this one**);
   a warning is printed below 90%.
6. Per-image outputs: SVG, preview, source reference, overlay review.html,
   machine-readable report.json, output notes, and a zip package.
7. Batch-safe: same-stem inputs never overwrite each other; any failure
   sets exit code 1; filenames containing `&` still produce valid SVG.

## Good fit / poor fit

Good fit:

- AI-generated logos, flat marks, badges, icons, labels
- Limited palettes (roughly 2–8 colors) with clear shape boundaries
- Circular badge elements: rings, rivets, concentric borders

Poor fit (expect low scores; treat output as reference only):

- Photos and realistic illustrations
- Gradients, soft shadows, glows (flattened into solid colors)
- Very fine texture or hairline details
- Intentionally irregular hand-drawn styles (use `--geometry off`)

## Repository layout

```text
ai-vector-cleanroom/
  vector_cleanroom.py   main program (batch flow, reports, packaging)
  clean_base.py         engine (palette, tracing, geometry, grouping)
  trace_engine.py       background removal and preprocessing
  clean.bat             Windows double-click entry
  install_deps.bat      Windows one-click dependency install
  input/                put images here
  output/               results appear here
  tests/                synthetic tests (16, no binary assets)
  .github/workflows/    CI (Ubuntu + Windows)
```

## Usage

Requires Python 3.10+.

```powershell
python -m pip install -r requirements.txt
# optional: SVG-rendered previews and self-check scores
python -m pip install -r requirements-preview.txt
```

On Windows you can double-click `install_deps.bat` instead.

Put images into `input/`, then:

```powershell
python vector_cleanroom.py
```

On Windows, double-click `clean.bat`. Results land in `output/`:

```text
output/
  result_<name>/
    <name>_vector.svg        editable SVG, grouped by color/layer
    <name>_preview.png       preview rendered from the SVG*
    source_reference.png     cleaned source reference
    review.html              browser overlay review page
    report.json              machine-readable run report
    OUTPUT_README.txt        notes for whoever receives the folder
  result_<name>.zip
```

\* If the optional packages are missing, the preview falls back to the
cleaned source image and is clearly labeled as such in the console,
report.json, and OUTPUT_README.txt.

## Options

```text
--input DIR            input folder (default ./input)
--output DIR           output folder (default ./output)
--colors N             force a fixed palette size (default 0 = auto)
--background MODE      auto | keep | transparent (default auto)
                       auto removes light background connected to the
                       border; use keep if a light border-touching design
                       element disappears
--white-threshold N    light background threshold (default 220)
--max-size N           downscale the longest side before tracing
                       (default 2048; 0 disables)
--geometry LEVEL       conservative | normal | off (default conservative)
                       conservative: perfect circles, rivet alignment,
                                     concentric centers
                       normal:       also straightens ring/band edges
                                     into mathematical arcs
                       off:          keep raw traced shapes
```

Exit code is 0 only when every input file succeeded.

## Quality expectations (honest claims)

Measured with the built-in pixel match (max channel diff < 48, rendered on
white):

- Flat, limited-palette logos and badges typically score 95–99%
  `source match`.
- Gradient/shadow artwork can score high on `flat match` while dropping to
  85–90% `source match` — that gap is real information: detail was
  flattened away. Only `source match` should be quoted externally.

This tool makes no claim of "one-click 90%+ vectorization of arbitrary
images."

## Development and tests

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest tests/ -q
python preflight_check.py
```

All test fixtures are generated on the fly with Pillow; the repository
contains no binary image assets. CI runs on Ubuntu and Windows
(`.github/workflows/ci.yml`).

## What not to commit

- Private, client-owned, trademarked, or unclear-license images.
- SVGs generated from such images — outputs inherit the source's legal
  restrictions.
- Real contents of `input/` / `output/` (already gitignored).
- API keys or credentials of any kind (the tool is fully offline and needs
  none).

## Author, citation, license

Created and maintained by Shinichi Chang (張進逸). Credit as:

```text
AI Vector Cleanroom by 張進逸 (Shinichi Chang)
```

See `CITATION.cff` for academic references. MIT licensed, see `LICENSE`.

Dependency licenses: vtracer (MIT), Pillow (MIT-CMU), NumPy (BSD family).
Optional preview dependencies are listed in `requirements-preview.txt`;
review their licenses before binary redistribution.
