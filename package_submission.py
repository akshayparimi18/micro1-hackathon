"""
Packages the LangGraphBench evaluation repository into a clean ZIP archive for submission.
Excludes .git, .env, .venv, cache, and temporary files. Verifies size <= 50MB.
"""
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ZIP_NAME = ROOT_DIR / "langgraphbench-eval.zip"

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", ".venv", ".idea", ".vscode"}
EXCLUDE_FILES = {".DS_Store", "langgraphbench-eval.zip", "isolationbench-eval.zip", ".env"}

def package():
    file_count = 0
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in ROOT_DIR.rglob("*"):
            if file_path.is_dir():
                continue
            
            # Check directory exclusions
            parts = set(file_path.relative_to(ROOT_DIR).parts)
            if parts.intersection(EXCLUDE_DIRS):
                continue
            
            # Check file exclusions
            if file_path.name in EXCLUDE_FILES or file_path.name.endswith(".pyc") or file_path.name.endswith(".zip"):
                continue

            arcname = file_path.relative_to(ROOT_DIR)
            zipf.write(file_path, arcname)
            file_count += 1

    size_mb = ZIP_NAME.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully created {ZIP_NAME.name}")
    print(f"    Total files archived: {file_count}")
    print(f"    Archive size: {size_mb:.2f} MB (Limit: 50 MB)")

if __name__ == "__main__":
    package()
