from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "repo-qa"
DEFAULT_CHAT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 6
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_FILE_BYTES = 200_000
