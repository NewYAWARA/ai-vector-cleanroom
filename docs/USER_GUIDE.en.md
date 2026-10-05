# AI Vector Cleanroom — Full User Guide

[繁體中文](USER_GUIDE.md) | English · [English README](../README.en.md)

**Keep the usable vector work and leave difficult areas for the designer to finish.**

`v0.6.0-alpha` is a Windows source-only pre-release built around an Illustrator handoff workflow. It converts PNG, JPG, WebP and BMP into SVG, supports whole-image cleanup, and exports editable drafts directly. When areas need redrawing, you can mark objects as Accept, Review or Manual. The goal is to reduce the work needed after conversion, rather than forcing every pixel into more path fragments.

**There is no measured designer time-saving percentage, and Illustrator import and finishing have not been validated in the actual application.** Local program tests, browser previews, node reductions, and geometry/rendering checks cannot replace real editing, timing and final artwork checks by a designer.

This release follows `v0.5.0-alpha` in the public series and provides full English documentation. It packages the completed internal designer-handoff build as `v0.6.0-alpha`; the release-name and documentation corrections do not change that build's conversion algorithms.

The handoff view locates problems by comparing each unit's actually visible contribution against the original image. It can flag differences in color, thickness or opacity, lost tonal variation, and extra paint on near-white or transparent source areas. It no longer flags an entire group just because its bounding box intersects a problem area. Occlusion, grouping and translucent compositing are considered together; generic reminders are not repeated when there is no specific finding. Hints do not change the artwork or accept objects for you.

This comparison has limits on native image size, object count, rendering work and elapsed time. If the original is missing, a limit is reached, or comparison fails, the page reports that the check is incomplete. No hint does not mean approval. New reconstruction methods for thin objects still require independent evidence; a similar total area alone does not relax the source checks.

For supported gradient objects on white backgrounds, contour reconstruction can use the original image's edges directly. Small holes absent from the original and false gradients introduced by resizing are checked separately. The old trace is no longer treated as an unchangeable reference answer. If the source is insufficient, or a change would close a white gap or damage a stroke, the existing candidate is retained and the area is listed for review. Thin lines are checked against source thickness, endpoints and junctions; subtle gradients are not discarded merely because the quantized palette contains only one color.

Color and contour improvements can be evaluated separately. When nodes cannot be safely reduced but the original supports a shared gradient across some paths, that improvement can be retained while the result still requires manual review. Those results still contain multiple original paths; they are not reported as a single reconstructed object.

## First-time setup

1. Install **CPython 3.12 x64** on Windows, including the Python Launcher. Check that `py -3.12` works.
2. Keep the complete source folder and double-click `setup_windows.bat`. Initial installation needs internet access to download the pinned dependencies.
3. Once setup finishes, double-click **`工作台.bat`** (or the equivalent `workbench.bat`). Your browser opens the local workbench.
4. Drag in an image. After conversion, choose **Designer handoff (`設計師接手`)** in its result row.

The workbench normally uses `http://127.0.0.1:8765/`. If that port is occupied, it looks for the next available port; use the URL printed in the launch window. Keep that window open while using the workbench. Closing it stops the local save/export links from responding. Run `工作台.bat` again to access existing results.

## Recommended handoff workflow

**Convert → Designer handoff (`設計師接手`) → optionally clean up the whole image (`自動整理整張圖`) → export the handoff package → open `working.svg` in Illustrator.** You can skip automatic cleanup and start manual editing immediately. You also do not have to classify every object first: `working.svg` contains the full vector candidate, with the reference image hidden by default. The following decisions are optional and useful when you want to distinguish accepted areas from areas to redraw.

1. **Compare the original and vector.** Use the mouse wheel to zoom, hold Space while dragging to pan, click to select objects, or drag a selection rectangle. A rectangle selection must fully contain an object's known geometry bounds, helping avoid selecting a large background along with smaller objects. Select objects without trustworthy bounds from the list on the right.
2. **Review source differences, then structural editing burden.** When the original is available, the right panel lists color/coverage and tonal differences based on actually visible pixels, followed by structural hints such as repeated turns in short curves or high node counts. Select and zoom to inspect them. Holes or connectivity issues that cannot be assigned to an object remain whole-image review items; they are not assigned to every object whose bounding box touches them. You can also sort by node count, select multiple objects and apply decisions in batches. Groups follow the identifiable objects already present; they do not recover the author's original semantics or layers.
3. **Choose a treatment explicitly.** Every object initially has Review (`待確認`) status. A system recommendation never automatically becomes acceptance. Use `1 / 2 / 3` for Accept / Review / Manual, `Ctrl+Z` to undo and `Ctrl+Shift+Z` to redo.
4. **Check Show accepted only (`只看已採用`).** Removing an occluding object may expose a shape beneath it. Retaining original geometry does not guarantee that every subset preserves the original appearance.
5. **Choose Export Illustrator handoff package (`匯出 Illustrator 接手包`) directly.** A successful export also saves the current decisions. To save decisions without exporting, choose Save decisions (`儲存判斷`). You can export a draft with Review or Manual areas remaining; the page keeps their counts visible and does not call it finished artwork.

| Status | Meaning | Export behavior |
|---|---|---|
| Accept (`採用`) | This object is useful as a basis for further editing | Included in `accepted.svg` and visible in `draft.svg` |
| Review (`待確認`) | Still needs inspection or a trial edit | Excluded from the accepted file; retained as a hidden candidate with location markers in the draft |
| Manual (`交人工`) | The designer will redraw or otherwise rework it | Also excluded from the accepted file and listed as work remaining |

New conversions also save the original image before background removal. If an older result only has a cleaned reference, the handoff page and exported data label it **Cleaned reference image, not the original (`清理後參考圖（非原始圖片）`)**. That reference cannot verify details that preprocessing may already have removed.

## Using the handoff package

Each export creates a separate folder and ZIP. It does not overwrite the previous handoff package.

| File | Contents |
|---|---|
| `working.svg` | The complete, directly editable vector candidate, plus a tracing reference at 35% opacity that is hidden by default. No need to accept every object first. Candidates marked Manual are also retained; use the work list to decide what to keep. It contains an embedded raster reference, so **it is not vector-only final artwork**. |
| `draft.svg` | Accepted vectors, a raster reference at 35% opacity, hidden candidates and markers for remaining work. **Not vector-only final artwork.** |
| `accepted.svg` | Only accepted objects. If areas remain unaccepted, this is an incomplete vector draft. Because all objects initially need review, the file is blank until you accept something. |
| `handoff.json` | Work list, reference-image type, objects that could not be located, and export status |
| `handoff-manifest.json`, `handoff-decisions.json` | Object information and the decisions saved with this export |
| `OPEN_IN_ILLUSTRATOR.txt` | Steps for taking over in Illustrator |

In Illustrator, **open `working.svg` first** to edit the complete candidate. Show and lock the reference image when you need it. Use `draft.svg` when you want to retain only accepted areas and redraw against the original. Check opacity, gradients, holes and stacking/occlusion; imported SVG groups and appearance still require manual inspection. Before finishing, remove the reference image, markers and unwanted hidden candidates, then save as `.ai`.

This release does not directly generate `.ai` files or promise to recover original fonts and editable text. Text in the existing conversion pipeline is usually represented as outline paths.

To check original colors, use the original-image pane in the handoff view, or set the reference image in Illustrator to 100% opacity. After cleanup, export from the new result's handoff page. Old review pages and existing handoff packages are not updated automatically.

## Whole-image automatic cleanup

Choose **Clean up the whole image (`自動整理整張圖`)** to search for safe node reductions area by area. Accepted objects stay unchanged. Each path is checked independently: inability to improve one area does not fail the whole operation or force a shape onto that area.

- Supports closed solid-color contours and verifiable linear/radial gradients. A native SVG renderer checks actual gradients, along with holes, transparent cracks, the full scene and local differences against the original.
- The cleanup search budget is 180 seconds, 256 candidate attempts and 48,000 input anchors. Each path is limited to 4,096 anchors and 12 seconds. Final checks and saving take additional time. Unsupported parts or those beyond the budget are retained, with reasons listed.
- The new version reports reduced nodes, retained areas and source-based reconstructions. Changed areas return to Review; accepted geometry, styles and stacking remain the same.
- The same path is not repeatedly simplified to accumulate error. To compare another tolerance, start again from the original version.
- For an opaque original on white containing exactly one solid-color shape without holes, the tool can also try **Reconstruct ellipse from source (`依原圖重建橢圓`)**. It retains a native four-anchor ellipse only when source contour, corner, color and hole checks pass and it matches the original better than the previous vector. This is reconstruction, distinct from conservative node reduction, and is identified in the UI and report. Design intent still needs confirmation.
- Cleanup rebuilds the recoloring page from the latest SVG. Paint types that cannot be fully supported are marked unavailable instead of reusing an outdated recoloring page.

The tolerance is a fitting parameter, **not a percentage of artwork that will need no editing**. It is measured relative to the contour bounding box's diagonal. Ninety-five percent of sampled deviations must be within the chosen value, and the maximum sampled deviation must be no more than three times it. Corners and each small contour's own scale are checked separately. For example, a setting of 0.25% can still allow a maximum sampled deviation of 0.75%. Sampling is not a mathematical proof for every point on every curve. Do not force an irregular design into a regular shape just to minimize nodes.

Initial conversion also handles supported regular shapes. For isolated thin straight lines, the original is used to estimate position, thickness and color, retaining strokes with adjustable width when supported by the checks. For a single opaque gradient circle or rounded rectangle on white, reconstruction to a four-anchor circle or eight-anchor rounded rectangle requires a gradient model that explains the original, passing hole and contour checks, and improved actual rendering error. Merging edge fragments also requires source checks at each location with no regression. Multiple separate shapes, occlusion, compound holes, near-white gradients or insufficient evidence are not forced into this model. If one gradient replacement needs numerous extra fragments, that replacement is withdrawn while other usable work is retained.

Gradient reconstruction also has a contour-preserving option. If rebuilding the outline is unreliable but the original supports a gradient, a small number of existing paths can share one SVG gradient. The fill is replaced only when actual rendered alpha coverage is identical and source-error checks pass for the full scene, each path and its edges. This is neither object merging nor node reduction; recovering a gradient does not label a complex contour easy to edit.

Conversion can occasionally leave small false holes in colored areas. A corresponding hole contour is removed only when both the original and processed reference support filling it, and checks pass for the actual full-scene rendering and each repaired area. Other exterior contours and holes remain unchanged. Hole repair and subsequent curve cleanup are validated separately, and their node reductions are reported separately. Genuine white space, ambiguous evidence or another object's details are not absorbed merely because they resemble noise.

If simplifying an entire gradient contour after hole repair would create cracks, the tool can try bounded cleanup of individual spans. Only spans that pass are simplified; other spans and holes keep their existing geometry. Each attempt is compared with the same original contour, original image and complete scene, rather than repeatedly simplifying and accumulating error. This is a safe improvement found by the search, not proof of the mathematically smallest possible node count.

## Local node reduction: try a small area and keep a separate version

Select areas that have not been accepted, then choose **Simplify selected contours only (`僅簡化所選輪廓`)**. Available geometry-error budgets are conservative `0.1%`, standard `0.25%` and larger `0.5%`. These are neither quality scores nor time-saving percentages.

This operation **reduces nodes on existing SVG contours; it does not retrace the original image**. Its scope is intentionally limited:

- One to four objects at a time, with at most four paths in total; 5–512 nodes per path, and no more than 1,536 nodes overall.
- Only closed paths with explicit solid-color fills are supported. Strokes, gradients/resource dependencies, text, native shapes, transforms, masks, clipping and filters are outside this mode's scope. Selecting an unsupported item rejects the request while retaining the original result and decisions.
- The worker process has a 60-second limit. No new version is published if it times out, fails to reduce nodes, or fails the checks.
- Interactive simplification currently does not support SVGs larger than 5 MiB or canvases with an aspect ratio beyond 8:1.
- The new version preserves geometry, styles and stacking for unselected and accepted objects. Changed objects return to Review (`待確認`).
- A path cannot be repeatedly simplified in derived versions. To compare another error budget, return to the original result and try again to avoid cumulative drift.

Passing local and full-scene rendering checks, hole checks and connectivity checks does not verify every source detail or approve the whole design. Visually review the new version yourself; it does not inherit the old version's overall acceptance score.

## Saving and versions

The browser temporarily retains unsaved decisions. **Choose Save decisions (`儲存判斷`) to write them to the workbench**; a successful export also saves that set of decisions. When you reopen the page, a local draft based on the same version as the workbench is restored and marked unsaved.

Saving, exporting and simplifying check the SVG fingerprint and decision version using compare-and-swap (CAS). If another window saves first, the older window gets a conflict message and does not overwrite the newer data. After reopening, a draft for a different version is not applied automatically: the workbench's saved decisions are displayed while the local draft is retained, with **Recover local draft and recheck (`取回本機草稿，重新核對`)** and **Download retained draft JSON (`下載保留草稿 JSON`)** options. Recovery only loads the draft into the page. You must recheck the areas and save explicitly; you can also undo the recovery. Conflicts are not automatically merged, and recovery does not overwrite the workbench.

Saved accepted objects protect a result from being rerun in place. **Rerun as a new version (`另開版本重跑`)** creates a separate result; the original decisions and exported handoff packages remain available. Local node reduction also creates a new version.

## Installation and data locations

This release provides MIT-licensed source code. It does not bundle a Python runtime and is not a package installable with `pip install ai-vector-cleanroom`. The launchers target **Windows x64 with CPython 3.12 x64**. Other platforms are outside this preview's validation scope.

| Item | Default location |
|---|---|
| Dedicated Python environment | `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha` |
| Working data | `%LOCALAPPDATA%\AIVC\designer4` |
| Inputs/results | `input` / `output` under the working-data folder |

The dedicated `v0.6.0-alpha` environment does not overwrite an older environment. Updating the source or rerunning setup does not clear working data.

When updating from `v0.5.0-alpha`, data in the old source folder is not migrated automatically. Keep old results and exported packages where they are, and import the original images into this release if you want new conversions. The `designer4` folder name is a storage identifier retained from development, not a public version number. Existing data at the configured root is not cleared or automatically reconverted. Close any previous workbench before starting this one.

To customize paths, set them in the same Command Prompt window before running setup and the launcher:

```bat
set "AVC_VENV_DIR=D:\venvs\aivc-v060"
set "AVC_DATA_DIR=D:\AIVC-designer4"
setup_windows.bat
工作台.bat
```

Use short, absolute local paths. A custom environment must be a new location or a dedicated environment created by this setup with a valid ownership marker. The program does not take over another project's virtual environment. Setup checks pinned versions, the Python ABI and required packages, and preserves the existing environment if a repair fails. The dependency lock is `requirements/validated-py312.lock.txt`.

If `AVC_VENV_DIR` points to an environment from an older development build, remove the override with `set "AVC_VENV_DIR="` in Command Prompt, or choose a new empty path, before running setup. An old environment's ownership marker may not match this version; do not edit it manually. Save any decisions and export a handoff package before replacing an existing working setup. Unsaved browser drafts belong to that browser and URL, and may not appear automatically if you change browsers or ports.

Only one process may write to the same data root at a time. Do not place it on OneDrive, a network drive or a shared synchronization folder. The workbench listens only on `127.0.0.1`; conversion and handoff data stay on the local machine, with no image upload to external services. Initial dependency installation needs internet access.

## Batch conversion and advanced features

Put images in `input` under the data root and run `清稿.bat`, or specify paths explicitly:

```bat
清稿.bat --input D:\vector-jobs\input --output D:\vector-jobs\output
```

CLI `--input` and `--output` override the default locations. To open those results in the workbench, use the matching data root.

`clean.bat` is equivalent to `清稿.bat`; `workbench.bat` is the English-filename entry point for `工作台.bat`. Run `clean.bat --help` for the current options:

| Option | Default and purpose |
|---|---|
| `--input` / `--output` | `input` / `output` under the working-data folder; set the batch input and result directories |
| `--colors` | `0` for automatic palette detection; otherwise a fixed palette size from 2 to 64 |
| `--white-threshold` | `220`; light/checker-background cleanup threshold, from 0 to 255 |
| `--background` | `auto` heuristically removes light background connected to the image border; `keep` retains the background; `transparent` forces a background-removal attempt |
| `--max-size` | `2048`; longest-side limit before tracing; `0` disables downscaling. A positive value must be at least 16 pixels |
| `--strokes` | `on` attempts to reconstruct uniform-width strokes with editable thickness; `off` disables it |
| `--gradients` | `on` attempts to reconstruct gradient fills; `off` disables it |
| `--geometry` | `conservative` regularization by default; `normal` also allows ring/band edges to become mathematical arcs; `off` disables it |
| `--curve-error-percent` | `0.25`; normalized curve-fitting p95 error budget, with a permitted setting range of 0.05–2.0. See Whole-image automatic cleanup above for its meaning and limits |
| `--debug` | Off by default; show full tracebacks for failed files |

The deprecated `--no-geometry` option remains an alias for `--geometry off`. These switches enable candidate searches; they do not guarantee that strokes, regular shapes or gradients will be reconstructed. For example:

```bat
clean.bat --input D:\vector-jobs\input --output D:\vector-jobs\output --background keep --colors 8 --geometry conservative
```

The batch exit code is 0 only if all inputs succeed. Failed inputs or no successful output return 1; data-directory or writer-lock errors return 2.

This release retains the conversion, conservative geometry/stroke/gradient processing, review and recoloring features developed through Beta.6 and the designer previews. Workbench conversion defaults to at most 1,200 seconds (20 minutes) and 16 candidates, and can be canceled. Failure or timeout does not publish an incomplete result. Full source and scene checks on complex multicolor images can take more than 15 minutes; the limit is not a promise of completion time. `AVC_JOB_TIMEOUT_SECONDS` can set a conversion limit between 30 and 1,800 seconds. It does not relax quality checks or change the 60-second limit for local node reduction.

Within a conversion, exactly matching SVG, source, reference and candidate parameters can reuse scene evidence for hole repair, span cleanup and local gradients. The cache is not persisted across jobs, and changed scenes are checked again. WebGPU is used only for compatible, validated palette-labeling operations, not for the entire conversion. Set `AVC_GPU_MODE=cpu` to use the CPU mode.

See [CHANGELOG.md](../CHANGELOG.md) for historical behavior, failure diagnostics and rollback records. Failed conversion diagnostics may be stored in `.failed_jobs` under the data root. Partial outputs there are not finished results.

## Conversion outputs, review and recoloring

A normal conversion result is separate from a designer handoff package. Each image has a result folder at `output/result_<name>`, plus a result ZIP. Use the links in its workbench result row, or open the HTML files in the output folder locally.

| File | Purpose |
|---|---|
| `<name>_vector.svg` | The converted vector candidate, distinct from the handoff package's `working.svg`, which also contains a reference image |
| `<name>_preview.png` | A vector preview when SVG rendering succeeds; check the result report and output notes for the actual preview status |
| `source_original.png` | The original-image reference saved by new conversions, before background removal |
| `source_reference.png` | The reference after background cleanup; it must not be treated as the unprocessed original |
| `review.html` | Local browser review page for overlay comparison of contours and objects |
| `色彩調整.html` | Offline recoloring page when available. Change paint roles and download a new SVG. Unsupported paint types may prevent this page from being generated |
| `report.json` | Machine-readable parameters, candidates, checks and diagnostics |
| `OUTPUT_README.txt` | Notes, warnings and available output files for this result |

Choose **Open review (`開啟校稿`)** or **Recolor (`換色`)** in the result row, or open the corresponding HTML file in a local browser. A recolored SVG downloaded by the browser does not automatically overwrite the workbench version or an existing handoff package. After whole-image cleanup or local simplification, use the new result's review/recolor pages rather than an older page. If the renderer failed, a preview may be a labeled fallback based on the cleaned source; it is not evidence that the SVG itself looks correct. Check `report.json` and `OUTPUT_README.txt`.

Choose **Generate blind-test page (`產生盲測頁`)** in the workbench for Stage 1 visual blind review. It collects appearance judgments, not evidence of time saved. **Stage 2 editing time (`Stage 2 實作計時`)** creates a timing page for a designer to actually edit the SVG. It is an entry point for collecting human evidence, not an automatically completed acceptance test. The interface is currently primarily Traditional Chinese; the labels above match the buttons you will see.

## What validation does and does not establish

Complete setup, then run `tests\run_tests.bat` for program regression tests. These are program and synthetic-example checks, **not proof of Illustrator acceptance or actual designer time savings**.

The workbench retains visual blind-review and **Stage 2 editing-time (`Stage 2 實作計時`)** entry points. To evaluate whether the tool is worthwhile, designers should use real work images and separately time completing specified edits from a tool-generated draft and redrawing from scratch. Record areas that require complete redrawing, import distortions and handoff failures. Until those measurements exist, no percentage of edit-free output or time saved is promised.

Third-party packages have their own licenses; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Do not commit user images, outputs, virtual environments, caches or private validation assets to the public repository.
