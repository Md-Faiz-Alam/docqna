from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
METADATA_DIR = DATA_DIR / "metadata"

CHROMA_INDEX_DIR = INDEX_DIR / "chroma"
BM25_INDEX_DIR = INDEX_DIR / "bm25"


# ---------------------------------------------------------
# Document configuration
# ---------------------------------------------------------

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

SUPPORTED_FILE_TYPES = {
    "pdf",
    "docx",
    "txt",
}


# ---------------------------------------------------------
# Text configuration
# ---------------------------------------------------------

DEFAULT_TEXT_ENCODING = "utf-8"


# ---------------------------------------------------------
# Index configuration
# ---------------------------------------------------------

DEFAULT_CHROMA_COLLECTION = "docqna_documents"


# ---------------------------------------------------------
# Application metadata
# ---------------------------------------------------------

APP_NAME = "DocQ&A"
APP_VERSION = "0.1.0"