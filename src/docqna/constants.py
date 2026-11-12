from pathlib import Path

# Repository root when running from a source checkout. Runtime paths are
# configured through `Settings.data_dir` / `Settings.log_dir`; this is only
# their default.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt"}
SUPPORTED_FILE_TYPES = {"pdf", "docx", "txt"}

DEFAULT_TEXT_ENCODING = "utf-8"

APP_NAME = "DocQ&A"
APP_VERSION = "0.1.0"
