# AI Vector Cleanroom

[繁體中文](README.md) | English

Turn PNG, JPG, WebP and BMP images into **SVG drafts that designers can take over and edit**. Keep the usable parts and retain a reference for the difficult areas, so the designer can decide what to adjust or redraw.

**Current version: `v0.6.0-alpha`, a Windows source-only pre-release.** The goal is to reduce the time designers spend cleaning up the result. Designer editing times have not yet been measured in a controlled comparison, and Adobe Illustrator import and finishing have not been validated in the actual application. This is not a guarantee of one-click finished artwork, and no percentage of time saved is promised.

Created and maintained by **Shinichi Chang (張進逸)**. MIT licensed.

Full Chinese and English documentation is provided. The workbench interface is still primarily Traditional Chinese; this guide includes the matching Chinese button labels.

## First-time setup

1. Install **CPython 3.12 x64 for Windows x64**, including the Python Launcher. Check that `py -3.12` works.
2. Download the complete source for this pre-release from [Releases](https://github.com/NewYAWARA/ai-vector-cleanroom/releases) and extract it.
3. Double-click `setup_windows.bat`. The first installation needs internet access to download the pinned dependencies.
4. Double-click **`工作台.bat`** (or its English-filename equivalent, `workbench.bat`) to open the local workbench, then drag an image into the browser page.
5. When conversion finishes, choose **Designer handoff (`設計師接手`) → Export Illustrator handoff package (`匯出 Illustrator 接手包`)**. Extract the package and open `working.svg` first.

You can run Clean up the whole image (`自動整理整張圖`) before exporting, or export directly. **You do not have to mark every object as accepted first.** Complex images may take several minutes or longer. The workbench shows progress and supports cancellation; keep its launch window open while using it.

The workbench listens only on local `127.0.0.1`, normally on port 8765. Use the URL printed in the launch window. Conversion needs no API key and does not upload images to an external service.

## What changed since the previous public version

Compared with `v0.5.0-alpha`, this update focuses on what happens **after conversion, when a designer starts editing**. Existing stroke, regular-shape, gradient, grouping and recoloring features are retained. They were not all introduced in this update, and not every image can be reconstructed into those structures.

| Improvement | How it helps with handoff |
|---|---|
| Use original-image evidence to reconstruct supported contours and check false holes and white gaps | Reduces cases where pixel stair-steps or intermediate tracing errors are preserved as if they were part of the design |
| Whole-image cleanup and limited local node reduction | Improves the parts that pass the checks, preserves the rest, and saves a separate version for comparison |
| Side-by-side source/vector comparison, object selection and zoom | Helps locate areas needing work instead of relying only on an average score for the whole image |
| Object-level difference hints | Uses what is actually visible to flag color, coverage and tonal differences, replacing repeated generic reminders |
| Accept / Review / Manual decisions and a handoff package | Lets you edit the full candidate, or retain only accepted parts and redraw the rest |
| Saving, reruns and version-conflict protection | Reduces the risk of overwriting reviewed work or applying stale decisions to a newer result |

Hints are leads to investigate, **not a list of mandatory fixes**. No hint does not mean human approval. Missing source images, comparison failures and computational limits are reported explicitly. This update does not claim that every image or every region is better than in the old version.

`v0.6.0-alpha` follows `v0.5.0-alpha` in the public release series and provides full Chinese and English documentation. It packages the completed internal designer-handoff build; correcting the release name and documentation does not change that build's conversion algorithms.

See the [release notes](release/RELEASE_NOTES.en.md) for this release and [CHANGELOG.md](CHANGELOG.md) for the history.

## Which handoff file should I open?

| File | Purpose |
|---|---|
| **`working.svg`** | Open this first. It contains the complete vector candidate plus an embedded raster reference that is hidden by default. **It is not vector-only final artwork.** |
| `accepted.svg` | Contains only objects you explicitly accepted. All objects initially need review, so this file is blank if you export without making any acceptance decisions. |
| `draft.svg` | Accepted parts, a tracing reference, hidden candidates and location markers for areas still needing work. Also not vector-only final artwork. |
| `handoff.json` and other JSON files | The work list, object data and the decisions saved with this export. |
| `OPEN_IN_ILLUSTRATOR.txt` | Handoff and checking instructions. |

In Illustrator, show and lock the reference image when you need to trace. Before finishing, remove the reference image, markers and unwanted hidden candidates; check holes, transparency, gradients and overlaps; then save as `.ai`. This tool does not directly generate `.ai` files or automatically recover original fonts and editable text.

## Suitable inputs and current limits

**Good candidates to try:** flat icons, labels, badges and graphics with a limited palette and clear boundaries, especially when some manual finishing is acceptable.

**Expect more manual work:** low-resolution text, thin rays, fading tips, complex gradients, and multicolor shapes that touch or overlap. Extra color patches, banding, small gaps, wrong colors or awkward nodes may remain. Native stroke reconstruction may fail and leave filled outlines instead.

Photos, realistic illustrations, hair, textures, complex shadows and blur are not this release's main target. Grouping shapes also does not recover the original author's layers or design intent.

Fewer nodes, a closer visual match and passing program checks do not necessarily make a result easier to edit. **Actual time savings depend on how long a designer takes to complete the same task.**

## Updating from an older version

Extract this version into a new folder and run its setup. Do not overwrite an old environment that is still in use. This release creates a dedicated `v0.6.0-alpha` Python environment.

| Item | Default location |
|---|---|
| Dedicated Python environment | `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha` |
| Images and conversion results | `input` and `output` under `%LOCALAPPDATA%\AIVC\designer4` |

When updating from `v0.5.0-alpha`, old data is not migrated automatically. Drag the original images into the new workbench if you want to convert them again. Old results and exported packages remain in their original locations. The `designer4` folder name is a storage identifier retained from development, not a public version number. Close any old workbench before starting this one; do not run two writers against the same data folder.

This is a source-only release: Python is not bundled, and the project is not a package you can install with `pip install ai-vector-cleanroom`. Other operating systems are outside this preview's usage-validation scope.

If `AVC_VENV_DIR` points to an environment from an older development build, remove that override or choose a new empty path before running setup. An old environment's version marker may be incompatible; do not edit its marker file manually.

See the [full user guide](docs/USER_GUIDE.en.md) for custom data locations, batch conversion, keyboard controls and cleanup limits.

## Feedback from real design work

Please [open an issue](https://github.com/NewYAWARA/ai-vector-cleanroom/issues/new/choose). These details are more useful for deciding what to improve than a single overall score:

- Which parts did you keep immediately, keep after editing, or redraw entirely?
- What took the most time: finding objects, adjusting shapes, recoloring, or cleaning up fragments?
- If you compared workflows, how long did the tool-assisted and your usual methods take to reach the same quality?
- Include the tool version, Windows and Illustrator versions, steps to reproduce, and a shareable screenshot or small synthetic example.

Do not upload client artwork, private images or work you do not have permission to share. A minimal example you create yourself is welcome.

## Development, licensing and validation

Complete setup, then run `tests\run_tests.bat`. See [tests/README.md](tests/README.md) for test instructions and generated synthetic benchmarks, and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance. Program and rendering tests cannot replace checks in Illustrator or timed designer tasks.

See [LICENSE](LICENSE) for the MIT license, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices, and [AUTHORS.md](AUTHORS.md) and [CITATION.cff](CITATION.cff) for authorship and citation details.
