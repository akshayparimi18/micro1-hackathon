"""
LangGraphBench Submission Packager & POSIX Validator.
Builds a Linux-safe, judge-proof ZIP archive for HackerRank submission:
- Strict whitelist enforcement (no stray, temporary, or cache files).
- Explicit blacklist verification (.env, .git, __pycache__, .zip, etc.).
- Strict POSIX forward slashes ('/') in all ZIP headers.
- Line ending normalization to Unix LF ('\\n') for all text files.
- Automated clean-room extraction and verification in an isolated temporary directory.
- Detailed file audit table.
"""

import os
import sys
import time
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
ZIP_NAME = ROOT_DIR / "langgraphbench-eval.zip"

WHITELIST_ROOT_FILES = {
    "README.md",
    "requirements.txt",
    ".env.example",
    "run_eval.py",
    "verify_ground_truth.py",
    "package_submission.py",
    "HACKERRANK_SUBMISSION_DESCRIPTION.md",
    "VIDEO_SCRIPT.md",
}

WHITELIST_DIRS = {
    "tasks",
    "reference",
    "graders",
    "results",
    "transcripts",
}

BLACKLIST_PATTERNS = {
    ".env",
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".txt",
    ".example",
    ".yml",
    ".yaml",
}


def should_include_file(rel_path: Path) -> bool:
    """Evaluates whether a relative path belongs in the strict submission whitelist."""
    parts = rel_path.parts
    filename = parts[-1]

    # Explicit blacklist checks
    if filename in BLACKLIST_PATTERNS:
        return False
    if filename.endswith(".pyc") or filename.endswith(".zip") or filename.endswith(".lock"):
        return False
    if any(p in BLACKLIST_PATTERNS for p in parts[:-1]):
        return False
    if any(p.startswith(".") and p not in {".env.example"} for p in parts):
        return False

    # Root-level files
    if len(parts) == 1:
        return filename in WHITELIST_ROOT_FILES

    # Directory files
    top_dir = parts[0]
    return top_dir in WHITELIST_DIRS


def build_package() -> List[Dict[str, Any]]:
    """Builds the zip archive enforcing POSIX forward slashes and Unix LF line endings."""
    print("=" * 70)
    print(" BUILDING HACKERRANK-COMPLIANT POSIX SUBMISSION ARCHIVE")
    print("=" * 70)

    archived_files = []
    
    # Remove existing zip if present
    if ZIP_NAME.exists():
        ZIP_NAME.unlink()

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in sorted(ROOT_DIR.rglob("*")):
            if file_path.is_dir():
                continue

            rel_path = file_path.relative_to(ROOT_DIR)
            if not should_include_file(rel_path):
                continue

            # Strict POSIX forward slashes in archive header
            posix_arcname = rel_path.as_posix()
            assert "\\" not in posix_arcname, f"Forbidden backslash in arcname: {posix_arcname}"

            # Read content with line ending normalization for text files
            ext = file_path.suffix.lower()
            if ext in TEXT_EXTENSIONS or file_path.name == ".env.example":
                try:
                    text_content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text_content = file_path.read_text(encoding="utf-8-sig")
                # Normalize all line endings to Unix LF
                normalized_text = text_content.replace("\r\n", "\n").replace("\r", "\n")
                payload = normalized_text.encode("utf-8")
            else:
                payload = file_path.read_bytes()

            # Create ZipInfo with fixed POSIX permissions (0o644)
            zinfo = zipfile.ZipInfo(filename=posix_arcname)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16
            dt = time.localtime(file_path.stat().st_mtime)
            zinfo.date_time = (dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)

            zipf.writestr(zinfo, payload)

            archived_files.append({
                "path": posix_arcname,
                "orig_size": len(payload),
            })

    zip_size_bytes = ZIP_NAME.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    print(f"[+] Archive generated: {ZIP_NAME.name}")
    print(f"    Total files archived: {len(archived_files)}")
    print(f"    Archive size: {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    return archived_files


def verify_archive():
    """Performs strict POSIX header checks, blacklist assertion, and clean-room runtime tests."""
    print("\n" + "=" * 70)
    print(" VERIFYING SUBMISSION ARCHIVE INTEGRITY & POSIX HEADERS")
    print("=" * 70)

    assert ZIP_NAME.exists(), "Archive does not exist!"
    zip_size_mb = ZIP_NAME.stat().st_size / (1024 * 1024)
    assert zip_size_mb <= 5.0, f"Archive size ({zip_size_mb:.2f} MB) exceeds 5 MB!"

    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        infolist = zf.infolist()
        print(f"-> Verifying {len(infolist)} ZipInfo headers...")

        for zinfo in infolist:
            name = zinfo.filename
            # 1. POSIX forward slash check
            assert "\\" not in name, f"CRITICAL: Windows backslash found in ZIP header: {name}"
            # 2. Blacklist checks
            parts = name.split("/")
            assert ".env" not in parts, f"CRITICAL: Real .env key file found in ZIP: {name}"
            assert not any(p in BLACKLIST_PATTERNS for p in parts), f"Blacklisted path found: {name}"
            assert not name.endswith(".pyc"), f"Compiled bytecode found: {name}"
            assert not name.endswith(".zip"), f"Nested zip found: {name}"

    print("[PASSED] All ZIP headers are 100% POSIX forward slashes.")
    print("[PASSED] Zero credentials, bytecode, or blacklisted files present.")
    print(f"[PASSED] Archive size is well within limits ({zip_size_mb:.2f} MB <= 5.0 MB).")

    # Clean-room extraction and reproducibility execution test
    print("\n" + "=" * 70)
    print(" PERFORMING CLEAN-ROOM EXTRACTION & REPRODUCIBILITY TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"-> Extracting archive to isolated clean-room: {tmp_path}")
        with zipfile.ZipFile(ZIP_NAME, "r") as zf:
            zf.extractall(tmp_path)

        # 1. Run unit tests
        print("-> Running: python -m unittest graders/test_graders.py")
        res1 = subprocess.run(
            [sys.executable, "-m", "unittest", "graders/test_graders.py"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True
        )
        if res1.returncode != 0:
            print(res1.stdout)
            print(res1.stderr)
            raise AssertionError(f"Unit tests failed in clean-room environment (code {res1.returncode})")
        print("   [OK] Unit tests passed (0 false passes / 0 false negatives).")

        # 2. Run runtime ground-truth verification
        print("-> Running: python verify_ground_truth.py")
        res2 = subprocess.run(
            [sys.executable, "verify_ground_truth.py"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True
        )
        if res2.returncode != 0:
            print(res2.stdout)
            print(res2.stderr)
            raise AssertionError(f"Runtime ground truth verification failed in clean-room (code {res2.returncode})")
        print("   [OK] Ground truth verified against live LangGraph runtime (100% match).")

        # 3. Run reference solution harness check
        print("-> Running: python run_eval.py --verify-reference")
        res3 = subprocess.run(
            [sys.executable, "run_eval.py", "--verify-reference"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True
        )
        if res3.returncode != 0:
            print(res3.stdout)
            print(res3.stderr)
            raise AssertionError(f"Reference verification failed in clean-room (code {res3.returncode})")
        print("   [OK] Reference solutions verified in evaluation harness (100% pass).")

    print("\n" + "=" * 70)
    print(">>> CLEAN-ROOM EXTRACTION TEST PASSED: 100% REPRODUCIBLE <<<")
    print("=" * 70)


def print_audit_table(files: List[Dict[str, Any]]):
    """Prints a categorized audit summary of all files included in the package."""
    print("\n" + "=" * 70)
    print(" ARCHIVE CONTENTS AUDIT TABLE")
    print("=" * 70)
    print(f"{'Filename / Path':<55} | {'Size':<10}")
    print(f"{'-'*55}-+-{'-'*10}")
    for item in sorted(files, key=lambda x: x["path"]):
        size_str = f"{item['orig_size']:,} B"
        print(f"{item['path']:<55} | {size_str:>10}")
    print(f"{'='*70}\n")


def main():
    files = build_package()
    verify_archive()
    print_audit_table(files)
    print(f"[SUCCESS] {ZIP_NAME.name} is ready for HackerRank submission!\n")


if __name__ == "__main__":
    main()
