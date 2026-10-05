# v0.6.0-alpha release notes

Version: `v0.6.0-alpha` · 2026-10-06 · **Source-only pre-release**

[繁體中文完整改版說明](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v0.6.0-alpha/release/RELEASE_NOTES.md)

## What changes in this release

This release continues the existing `v0.5.0-alpha → v0.6.0-alpha` public version series and provides complete Traditional Chinese and English versions of the README, user guide and release notes. The mistakenly published `v3-designer-preview.4` Release and tag, which used an internal development identifier, have been removed and are no longer public downloads. Internal Beta and Designer Preview development records remain in the CHANGELOG.

**Preparing this public release has not changed the completed conversion, refinement, handoff diagnostics or export behavior.** The numbering and documentation corrections are not another image-quality improvement. The dedicated Python environment uses the name `v0.6.0-alpha`. Work data retains the internally compatible `AIVC\designer4` path; this directory name does not identify another public release.

The goal remains to help designers take over a draft, retain useful vector work and redraw difficult areas. This is not a one-click finished-artwork release, and no percentage of designer time savings has been demonstrated. The feature changes below are **relative to the earlier public `v0.5.0-alpha` release**; they were not all introduced by this numbering correction.

## Changes designers will use, compared with v0.5

- **A complete handoff workflow.** Compare the original and vector side by side, select objects, zoom in, and batch-mark them as accepted, needing review or assigned to manual work. Export the complete candidate to continue editing in Illustrator without classifying every object first.
- **Separate versions after refinement.** Whole-image automatic cleanup and limited local node reduction improve supported areas while retaining the existing result where proposed changes fail checks. Accepted objects are protected. Reruns, saves and multiple browser windows use revision checks so stale operations cannot overwrite newer decisions.
- **Use the original image to judge some contours and holes.** In supported cases, reconstruction proposals follow the original image's edges and check false holes, white gaps and newly introduced cracks. Intermediate tracing output is no longer the only reference. The original candidate is retained if evidence is insufficient or other regions would be damaged.
- **Evaluate color and geometry separately.** Within supported conditions, gradient-paint improvements that pass source checks may be retained even when contours cannot be simplified safely. A paint-only change is not reported as successful object merging.
- **Locate specific review issues.** Each handoff object's actual visible contribution is compared with the original to flag differences in color or coverage, unwanted paint in blank areas and lost tonal variation. Repeated generic warnings are removed. Whole-scene structural concerns are listed separately instead of being assigned to every large object whose bounding box overlaps them. Selection outlines no longer tint thin lines, making color comparison easier.
- **Separate Windows installation from work data.** A dedicated CPython 3.12 x64 environment uses pinned dependencies, while work data stays in a separate local directory. Conversion has progress reporting, cancellation and time limits. Failed or timed-out jobs are not published as complete results.

Existing SVG strokes, regular shapes, gradients, grouping, recoloring and review tools are retained. They are not all new features of this release. There is no guarantee that every object will become a stroke with adjustable width or a single gradient.

The final internal development iteration primarily improved handoff diagnostics, and this release retains that behavior. Experimental reconstruction of thin elongated objects remains outside the default pipeline. Existing source checks have not simply been relaxed to make candidates pass.

## Open working.svg first after export

- **`working.svg`** contains the complete vector candidate and an embedded original raster reference that is hidden by default. It is the starting point for manual editing, not vector-only finished artwork.
- **`accepted.svg`** contains only objects explicitly accepted by the user. All objects initially need review, so the first export is empty if no decisions have been made.
- **`draft.svg`** contains accepted objects, a raster reference, hidden candidates and location hints to support redrawing. It is also not vector-only finished artwork.
- The handoff package also includes lists, decision JSON and `OPEN_IN_ILLUSTRATOR.txt`. Every export creates a new package without overwriting the previous one.

Before finishing, remove reference images, hint rectangles and unnecessary hidden candidates. Check transparency, holes, gradients and occlusion, then save as `.ai`. The tool does not directly produce `.ai` files. Text usually remains outlined geometry; the original typeface cannot be reliably recovered.

## Installation and updating

1. Prepare Windows x64 and CPython 3.12 x64, including the Python Launcher.
2. Download `AI-Vector-Cleanroom-v0.6.0-alpha.zip`, extract it into a new directory and run `setup_windows.bat`. The first setup needs an internet connection to install pinned dependencies.
3. Close any older workbench using the same data directory, then run `工作台.bat`. The English-named `workbench.bat` entry point behaves identically.

The default environment is `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha`. Data is stored in `%LOCALAPPDATA%\AIVC\designer4`, retaining the compatible internal development path. **This release does not automatically move old data or rerun conversions.** Keep previous outputs and handoff packages. Do not overwrite a program directory that is still running or run two workbenches writing to the same data directory.

If a custom `AVC_VENV_DIR` points to an environment for another version, setup will refuse to reuse it because its environment version marker differs. Clear that override to use the new default environment, or choose a fresh directory; do not edit the environment marker manually. Custom `AVC_DATA_DIR` settings and existing compatible work data are unaffected by the environment name.

No Python runtime is bundled. The validated environment is Windows x64 / CPython 3.12.x; other platforms and Python versions have not been formally validated. The release receipt is `SOURCE_RELEASE_RECEIPT_v0.6.0-alpha.json`.

For instructions, see the [Traditional Chinese README](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v0.6.0-alpha/README.md), [complete Traditional Chinese guide](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v0.6.0-alpha/docs/USER_GUIDE.md), [English README](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v0.6.0-alpha/README.en.md) and [complete English guide](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v0.6.0-alpha/docs/USER_GUIDE.en.md).

## Known limitations and scope of validation

- Text interiors, thin rays, faint tips, color bands, adjoining edges and local colors may still need manual editing or redrawing. Some regions may regress compared with v0.5; the overall average error alone is not sufficient.
- Diagnostics may miss issues or produce too many hints. Their order does not represent design importance or editing time. No warnings, fewer nodes or a machine acceptance status do not mean that artwork is ready to deliver.
- Complex soft edges, shadows, textures, photographs and original-typeface recovery are not this release's focus. Grouping does not recover the author's intended semantics or original layers.
- The earlier frozen Preview 4 development build had a local record of **975 unittest cases, 5 skipped and 0 failures**. Another run covered 36 synthetic benchmarks: 33 were machine-accepted and 3 required manual review. Diagnostic negative controls on 36 known-correct images produced no new difference hints. These are limited historical results, not a claim that this public commit has completed the same validation. They cannot establish that all real images are free of false warnings or regressions.
- **There has been no hands-on Illustrator import or finished-artwork acceptance test, and no timed designer comparison.** Test counts are not quality scores or time-saving percentages. Checks for this public commit are recorded in its own CI and release verification. Tests skipped because private fixtures are unavailable are not counted as passes.

## Feedback that matters most

Please use [Issues](https://github.com/NewYAWARA/ai-vector-cleanroom/issues/new/choose) to tell us which parts you kept, edited or redrew, and which step took the most time. Time estimates may be unknown; you do not need to repeat the work solely to report feedback. If you already have a comparison using the same finishing requirements, include the time taken with the handoff and with your usual method. These practical costs will guide subsequent priorities.

You may attach a minimal example you are allowed to publish. Do not submit client, private or unclearly licensed artwork. Public source includes programmatically generated synthetic test fixtures; production artwork, internal research outputs and local verification records are not included in the release package.
