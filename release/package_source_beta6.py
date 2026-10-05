# -*- coding: utf-8 -*-
"""Deterministic, source-only Designer Preview release builder.

The archive is an explicit allowlist suitable for publishing as a source
repository.  It intentionally excludes the bundled Python runtime, private
validation material, local release evidence, caches, inputs and outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile


SCHEMA = "ai-vector-cleanroom-source-release/1"
RECEIPT_SCHEMA = "ai-vector-cleanroom-source-release-receipt/1"
VERSION = "v3-designer-preview.4"
PACKAGE_NAME = "AI-Vector-Cleanroom-Designer-Preview-4"
MANIFEST_NAME = "SOURCE_MANIFEST.json"
RECEIPT_NAME = "SOURCE_RELEASE_RECEIPT_Designer_Preview_4.json"
FIXED_ZIP_TIME = (2026, 10, 5, 0, 0, 0)

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 1_000
MAX_PATH_CHARS = 220

PRODUCT_FILES = (
    "annulus_detector.py",
    "app_paths.py",
    "clean_base.py",
    "compute_backend.py",
    "component_repair.py",
    "compound_path_splitter.py",
    "curve_refit.py",
    "curve_refit_stage.py",
    "designer_ops_audit.py",
    "designer_quality.py",
    "designer_handoff.py",
    "handoff_page.py",
    "handoff_service.py",
    "local_refine.py",
    "auto_prepare.py",
    "native_geometry_contract.py",
    "svg_renderer.py",
    "svg_bounds.py",
    "alpha_topology.py",
    "contour_relationships.py",
    "trace_component_recovery.py",
    "gradient_residual_provenance.py",
    "gradient_paint_only.py",
    "gradient_contour_spans.py",
    "gradient_source_components.py",
    "source_primitive.py",
    "source_gradient_primitive.py",
    "source_edge_reconstruction.py",
    "source_light_cleanup.py",
    "source_scene_guard.py",
    "source_repair_stage.py",
    "source_flat_paint.py",
    "source_topology_audit.py",
    "source_object_audit.py",
    "source_edge_contacts.py",
    "source_boundary_evidence.py",
    "editability_audit.py",
    "editing_test_page.py",
    "editing_metrics.js",
    "environment_preflight.py",
    "execution_control.py",
    "exact_native_shapes.py",
    "geometry_error_optimizer.py",
    "gradient_candidate_groups.py",
    "gradient_object_engine.py",
    "gradient_reconstruction_stage.py",
    "job_worker.py",
    "paint_roles.py",
    "palette_sampling.py",
    "quality_diagnostics.py",
    "recolor_page.py",
    "scene_graph_postprocess.py",
    "stroke_engine.py",
    "svg_postprocess.py",
    "trace_engine.py",
    "vector_cleanroom.py",
    "workbench.py",
    "setup_windows.py",
)

PUBLIC_ROOT_FILES = PRODUCT_FILES + (
    ".gitattributes",
    ".gitignore",
    "AUTHORS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "LICENSE",
    "NOTICE.md",
    "PUBLISHING.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "VERSION.txt",
    "clean.bat",
    "install_deps.bat",
    "preflight_check.py",
    "setup_windows.bat",
    "workbench.bat",
    "使用說明.txt",
    "requirements.txt",
    "requirements-preview.txt",
    "工作台.bat",
    "清稿.bat",
)

REQUIREMENT_FILES = (
    "requirements/core-py312.lock.txt",
    "requirements/validated-py312.lock.txt",
)

TEST_FILES = (
    "tests/README.md",
    "tests/generate_fixtures.py",
    "tests/highres_smoke.py",
    "tests/run_highres_test.bat",
    "tests/run_tests.bat",
    "tests/test_adaptive_palette.py",
    "tests/test_palette_sampling.py",
    "tests/test_app_paths.py",
    "tests/test_annulus_detector.py",
    "tests/test_candidate_policy.py",
    "tests/test_clean_base_gradient_cache.py",
    "tests/test_clean_base_gradient_integration.py",
    "tests/test_clean_base_p0.py",
    "tests/test_clean_base_prefix_cache.py",
    "tests/test_component_bucketing.py",
    "tests/test_component_repair.py",
    "tests/test_compute_backend.py",
    "tests/test_compound_path_splitter.py",
    "tests/test_curve_refit.py",
    "tests/test_curve_refit_stage.py",
    "tests/test_curve_refit_transaction.py",
    "tests/test_designer_ops_audit.py",
    "tests/test_designer_quality.py",
    "tests/test_designer_handoff.py",
    "tests/test_handoff_page.py",
    "tests/test_handoff_service.py",
    "tests/test_local_refine.py",
    "tests/test_auto_prepare.py",
    "tests/test_native_geometry_contract.py",
    "tests/test_svg_renderer.py",
    "tests/test_svg_bounds.py",
    "tests/test_alpha_topology.py",
    "tests/test_trace_component_recovery.py",
    "tests/test_gradient_residual_provenance.py",
    "tests/test_gradient_paint_only.py",
    "tests/test_gradient_contour_spans.py",
    "tests/test_gradient_source_components.py",
    "tests/test_source_primitive.py",
    "tests/test_source_gradient_primitive.py",
    "tests/test_source_edge_reconstruction.py",
    "tests/test_source_light_cleanup.py",
    "tests/test_source_scene_guard.py",
    "tests/test_source_repair_stage.py",
    "tests/test_source_flat_paint.py",
    "tests/test_source_topology_audit.py",
    "tests/test_source_object_audit.py",
    "tests/test_source_readiness_binding.py",
    "tests/test_curve_counterturn_quality.py",
    "tests/test_source_edge_contacts.py",
    "tests/test_source_boundary_evidence.py",
    "tests/test_foreground_alpha_metrics.py",
    "tests/generate_designer_benchmark.py",
    "tests/test_designer_benchmark.py",
    "tests/test_editability_audit.py",
    "tests/test_editing_test_page.py",
    "tests/test_environment_preflight.py",
    "tests/test_execution_control.py",
    "tests/test_exact_native_shapes.py",
    "tests/test_failure_diagnostics.py",
    "tests/test_geometry_error_optimizer.py",
    "tests/test_gradient_candidate_groups.py",
    "tests/test_gradient_object_engine.py",
    "tests/test_gradient_preview_painter.py",
    "tests/test_gradient_reconstruction.py",
    "tests/test_gradient_reconstruction_stage.py",
    "tests/test_job_worker.py",
    "tests/test_paint_resource_summary.py",
    "tests/test_paint_roles.py",
    "tests/test_quality_diagnostics.py",
    "tests/test_recolor_page.py",
    "tests/test_regression.py",
    "tests/test_scene_graph_postprocess.py",
    "tests/test_setup_windows.py",
    "tests/test_source_release.py",
    "tests/test_stroke_safety.py",
    "tests/test_svg_postprocess.py",
    "tests/test_trace_engine_binary_mask.py",
    "tests/test_visual_gate.py",
    "tests/test_workbench_beta4.py",
    "tests/test_workbench_handoff_links.py",
)

FIXTURE_FILES = (
    "tests/fixtures/line_over_fill_darkgray.png",
    "tests/fixtures/low_contrast_ddd.png",
    "tests/fixtures/mixed_alpha.png",
    "tests/fixtures/multicolor_touch.png",
    "tests/fixtures/one_px_black.png",
    "tests/fixtures/one_px_black_3000.png",
    "tests/fixtures/right_angle.png",
    "tests/fixtures/ring.png",
    "tests/fixtures/soft_alpha_100.png",
    "tests/fixtures/square_frame_5px.png",
    "tests/fixtures/t_junction.png",
    "tests/fixtures/x_junction.png",
    "tests/fixtures/y_junction.png",
)

RELEASE_FILES = (
    "release/__init__.py",
    "release/package_source_beta6.py",
    "release/RELEASE_NOTES.md",
    "docs/USER_GUIDE.md",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/designer_feedback.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
)

PUBLIC_FILES = tuple(sorted(
    PUBLIC_ROOT_FILES + REQUIREMENT_FILES + TEST_FILES + FIXTURE_FILES
    + RELEASE_FILES
))

DENIED_COMPONENTS = frozenset({
    ".agents", ".beta6_runtime", ".codex", ".git", ".venv",
    "__pycache__", "input", "output", "python", "release_work",
    "validation",
})
DENIED_SUFFIXES = frozenset({
    ".bak", ".backup", ".lock", ".log", ".lck", ".orig",
    ".pyc", ".pyo", ".swp", ".temp", ".tmp",
})
TEXT_SUFFIXES = frozenset({
    ".bat", ".cfg", ".cmd", ".json", ".js", ".md", ".py", ".toml", ".txt",
    ".cff", ".yml", ".yaml",
})
ABSOLUTE_USER_PATH_RE = re.compile(
    rb"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9A-F]{64}$")


class SourcePackageError(RuntimeError):
    """A source-release safety or integrity contract failed."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    source: Path
    size: int
    sha256: str

    def record(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise SourcePackageError(f"invalid public path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SourcePackageError(f"unsafe public path: {raw!r}")
    if len(raw) > MAX_PATH_CHARS:
        raise SourcePackageError(f"public path is too long: {raw!r}")
    if any(part.casefold() in DENIED_COMPONENTS for part in path.parts):
        raise SourcePackageError(f"denied public path: {raw!r}")
    if path.suffix.casefold() in DENIED_SUFFIXES:
        raise SourcePackageError(f"denied public suffix: {raw!r}")
    return raw


def _is_reparse_or_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _scan_public_text(relative: str, data: bytes) -> None:
    if PurePosixPath(relative).suffix.casefold() not in TEXT_SUFFIXES:
        return
    if b"\x00" in data:
        raise SourcePackageError(f"public text contains NUL bytes: {relative}")
    if ABSOLUTE_USER_PATH_RE.search(data):
        raise SourcePackageError(
            f"public text contains an absolute Windows user path: {relative}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourcePackageError(f"public text is not UTF-8: {relative}") from exc
    if PurePosixPath(relative).suffix.casefold() == ".bat":
        _validate_windows_batch_bytes(relative, data)


def _validate_windows_batch_bytes(relative: str, data: bytes) -> None:
    """Require bytes that classic ``cmd.exe`` parses predictably.

    UTF-8 batch files intentionally omit a BOM because some Windows command
    processors treat it as part of the first command.  Every physical line
    must use CRLF; LF-only files can make labels and adjacent commands merge.
    """

    if data.startswith(b"\xef\xbb\xbf"):
        raise SourcePackageError(f"Windows batch file has a UTF-8 BOM: {relative}")
    if not data or not data.endswith(b"\r\n"):
        raise SourcePackageError(
            f"Windows batch file must end with CRLF: {relative}")
    without_crlf = data.replace(b"\r\n", b"")
    if b"\r" in without_crlf or b"\n" in without_crlf:
        raise SourcePackageError(
            f"Windows batch file must use CRLF-only line endings: {relative}")
    if not data.isascii():
        raise SourcePackageError(
            f"Windows batch file must be ASCII-only: {relative}")


def collect_entries(root: Path | None = None) -> tuple[FileEntry, ...]:
    base = (root or project_root()).resolve(strict=True)
    entries: list[FileEntry] = []
    folded: dict[str, str] = {}
    for relative in PUBLIC_FILES:
        _validate_relative_path(relative)
        collision_key = relative.casefold()
        if collision_key in folded:
            raise SourcePackageError(
                f"case-insensitive public path collision: {folded[collision_key]!r} "
                f"and {relative!r}")
        folded[collision_key] = relative
        source = base.joinpath(*PurePosixPath(relative).parts)
        if not source.is_file() or _is_reparse_or_link(source):
            raise SourcePackageError(f"missing or unsafe public file: {relative}")
        resolved = source.resolve(strict=True)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise SourcePackageError(f"public file escaped project: {relative}") from exc
        data = source.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise SourcePackageError(f"public file exceeds size limit: {relative}")
        _scan_public_text(relative, data)
        entries.append(FileEntry(
            path=relative,
            source=source,
            size=len(data),
            sha256=_sha256_bytes(data),
        ))
    if tuple(entry.path for entry in entries) != PUBLIC_FILES:
        raise SourcePackageError("public allowlist ordering changed")
    total = sum(entry.size for entry in entries)
    if total > MAX_ARCHIVE_BYTES:
        raise SourcePackageError("source release exceeds total size limit")
    return tuple(entries)


def _manifest(entries: Sequence[FileEntry]) -> dict[str, object]:
    return {
        "files": [entry.record() for entry in entries],
        "package_name": PACKAGE_NAME,
        "policy": "explicit_source_allowlist_no_runtime_or_private_evidence",
        "schema": SCHEMA,
        "version": VERSION,
    }


def _manifest_bytes(entries: Sequence[FileEntry]) -> bytes:
    return _canonical_json(_manifest(entries))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def audit(root: Path | None = None) -> dict[str, object]:
    entries = collect_entries(root)
    return {
        "file_count": len(entries),
        "package_name": PACKAGE_NAME,
        "status": "source_allowlist_ready",
        "total_bytes": sum(entry.size for entry in entries),
        "version": VERSION,
    }


def _write_archive(path: Path, entries: Sequence[FileEntry]) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        for entry in entries:
            archive.writestr(
                _zip_info(f"{PACKAGE_NAME}/{entry.path}"),
                entry.source.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        archive.writestr(
            _zip_info(f"{PACKAGE_NAME}/{MANIFEST_NAME}"),
            _manifest_bytes(entries),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )


def _parse_manifest(data: bytes) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePackageError("source manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise SourcePackageError("source manifest is not an object")
    expected_keys = {"files", "package_name", "policy", "schema", "version"}
    if set(payload) != expected_keys:
        raise SourcePackageError("source manifest fields differ")
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise SourcePackageError("source manifest schema/version differs")
    if payload.get("package_name") != PACKAGE_NAME:
        raise SourcePackageError("source manifest package name differs")
    if payload.get("policy") != "explicit_source_allowlist_no_runtime_or_private_evidence":
        raise SourcePackageError("source manifest policy differs")
    records = payload.get("files")
    if not isinstance(records, list) or len(records) != len(PUBLIC_FILES):
        raise SourcePackageError("source manifest file count differs")
    previous = None
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise SourcePackageError("source manifest record fields differ")
        relative = _validate_relative_path(record.get("path"))
        if previous is not None and relative <= previous:
            raise SourcePackageError("source manifest is not strictly sorted")
        previous = relative
        if relative.casefold() in seen:
            raise SourcePackageError("source manifest has a path collision")
        seen.add(relative.casefold())
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise SourcePackageError("source manifest has an invalid size")
        if not isinstance(record.get("sha256"), str) or not SHA256_RE.fullmatch(record["sha256"]):
            raise SourcePackageError("source manifest has an invalid SHA-256")
    if tuple(record["path"] for record in records) != PUBLIC_FILES:
        raise SourcePackageError("source manifest differs from the public allowlist")
    if data != _canonical_json(payload):
        raise SourcePackageError("source manifest is not canonical JSON")
    return payload, records


def verify(zip_path: Path, receipt_path: Path | None = None) -> dict[str, object]:
    archive_path = zip_path.resolve(strict=True)
    if archive_path.name != PACKAGE_NAME + ".zip":
        raise SourcePackageError("source ZIP has an unexpected filename")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SourcePackageError("source ZIP exceeds size limit")
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_ENTRIES:
            raise SourcePackageError("source ZIP has an invalid entry count")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise SourcePackageError("source ZIP has duplicate members")
        if archive.testzip() is not None:
            raise SourcePackageError("source ZIP CRC verification failed")
        manifest_member = f"{PACKAGE_NAME}/{MANIFEST_NAME}"
        if names.count(manifest_member) != 1:
            raise SourcePackageError("source ZIP lacks one manifest")
        manifest_data = archive.read(manifest_member)
        _, records = _parse_manifest(manifest_data)
        expected_names = [f"{PACKAGE_NAME}/{record['path']}" for record in records]
        expected_names.append(manifest_member)
        if names != expected_names:
            raise SourcePackageError("source ZIP member set/order differs")
        info_by_name = {info.filename: info for info in infos}
        total = 0
        for record in records:
            relative = str(record["path"])
            name = f"{PACKAGE_NAME}/{relative}"
            info = info_by_name[name]
            if info.is_dir() or info.file_size != record["size"]:
                raise SourcePackageError(f"source ZIP size differs: {relative}")
            total += info.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise SourcePackageError("source ZIP expanded size exceeds limit")
            data = archive.read(name)
            if _sha256_bytes(data) != record["sha256"]:
                raise SourcePackageError(f"source ZIP SHA-256 differs: {relative}")
            _scan_public_text(relative, data)
    result = {
        "file_count": len(records),
        "manifest_sha256": _sha256_bytes(manifest_data),
        "package_name": PACKAGE_NAME,
        "status": "source_archive_verified",
        "total_bytes": total,
        "version": VERSION,
        "zip_bytes": archive_path.stat().st_size,
        "zip_sha256": _sha256_file(archive_path),
    }
    if receipt_path is not None:
        try:
            receipt = json.loads(receipt_path.resolve(strict=True).read_text(
                encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourcePackageError("source receipt is unreadable") from exc
        expected_receipt = {
            "file_count": result["file_count"],
            "manifest_sha256": result["manifest_sha256"],
            "package_name": PACKAGE_NAME,
            "schema": RECEIPT_SCHEMA,
            "version": VERSION,
            "zip_bytes": result["zip_bytes"],
            "zip_name": archive_path.name,
            "zip_sha256": result["zip_sha256"],
        }
        if receipt != expected_receipt:
            raise SourcePackageError("source receipt differs from the ZIP")
        result["receipt"] = "verified"
    return result


def build(zip_path: Path, receipt_path: Path, root: Path | None = None) -> dict[str, object]:
    archive_path = zip_path.resolve(strict=False)
    receipt = receipt_path.resolve(strict=False)
    if archive_path.name != PACKAGE_NAME + ".zip":
        raise SourcePackageError("source ZIP filename must be fixed")
    if receipt.name != RECEIPT_NAME:
        raise SourcePackageError("source receipt filename must be fixed")
    if archive_path.exists() or receipt.exists():
        raise SourcePackageError("source build refuses to overwrite existing output")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    entries = collect_entries(root)
    before = _manifest_bytes(entries)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".source-release-", suffix=".zip.tmp", dir=archive_path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        _write_archive(temporary, entries)
        # Verification requires the fixed basename, so validate the staged
        # bytes directly before the no-overwrite atomic publication.
        with zipfile.ZipFile(temporary, "r") as staged:
            manifest_data = staged.read(f"{PACKAGE_NAME}/{MANIFEST_NAME}")
            _parse_manifest(manifest_data)
            if staged.testzip() is not None:
                raise SourcePackageError("staged source ZIP CRC verification failed")
        after_entries = collect_entries(root)
        if _manifest_bytes(after_entries) != before:
            raise SourcePackageError("public sources changed during source build")
        zip_sha256 = _sha256_file(temporary)
        zip_bytes = temporary.stat().st_size
        receipt_payload = {
            "file_count": len(entries),
            "manifest_sha256": _sha256_bytes(before),
            "package_name": PACKAGE_NAME,
            "schema": RECEIPT_SCHEMA,
            "version": VERSION,
            "zip_bytes": zip_bytes,
            "zip_name": archive_path.name,
            "zip_sha256": zip_sha256,
        }
        receipt_bytes = _canonical_json(receipt_payload)
        fd, receipt_temp_name = tempfile.mkstemp(
            prefix=".source-receipt-", suffix=".json.tmp", dir=receipt.parent)
        os.close(fd)
        receipt_temp = Path(receipt_temp_name)
        try:
            receipt_temp.write_bytes(receipt_bytes)
            os.replace(temporary, archive_path)
            try:
                os.replace(receipt_temp, receipt)
            except BaseException:
                archive_path.unlink(missing_ok=True)
                raise
        finally:
            receipt_temp.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return verify(archive_path, receipt)


def _path(value: str) -> Path:
    return Path(value)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--zip", type=_path, required=True)
    build_parser.add_argument("--receipt", type=_path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--zip", type=_path, required=True)
    verify_parser.add_argument("--receipt", type=_path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = audit()
        elif args.command == "build":
            result = build(args.zip, args.receipt)
        else:
            result = verify(args.zip, args.receipt)
    except (OSError, SourcePackageError, zipfile.BadZipFile) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"},
                         ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
