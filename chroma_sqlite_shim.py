"""
chroma_sqlite_shim.py
Force Chroma to use DuckDB and provide a modern sqlite shim BEFORE chromadb imports.
"""

from __future__ import annotations
import os, sys

# 1) Force DuckDB backend so Chroma won't default to sqlite at all.
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
os.environ.setdefault("CHROMADB_DEFAULT_DATABASE", "duckdb+parquet")
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")

# 2) Preload a modern sqlite implementation via pysqlite3 (wheel) for import-time checks.
#    This satisfies chroma's "sqlite >= 3.35.0" assertion if anything tries to touch sqlite anyway.
try:
    import pysqlite3  # requires pysqlite3-binary in requirements.txt
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except Exception:
    # If not installed yet, chroma may still fail; ensure requirements include pysqlite3-binary.
    pass
