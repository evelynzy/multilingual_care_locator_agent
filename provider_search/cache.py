from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Mapping, Optional

from provider_search.models import ProviderSearchCacheEntry


logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_CACHE_FILENAME = "provider_search.sqlite3"
DEFAULT_PROVIDER_CACHE_SUBDIR = "multilingual-care-locator-agent/provider-search-cache"


def resolve_provider_cache_path(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Resolve a writable cache database path, falling back to a temp directory."""

    environment = env or os.environ
    configured_dir = environment.get("PROVIDER_CACHE_DIR")

    candidate_dirs = []
    if configured_dir:
        candidate_dirs.append(Path(configured_dir).expanduser())

    candidate_dirs.append(Path(tempfile.gettempdir()) / DEFAULT_PROVIDER_CACHE_SUBDIR)

    for candidate_dir in candidate_dirs:
        if _prepare_cache_directory(candidate_dir):
            return candidate_dir / DEFAULT_PROVIDER_CACHE_FILENAME

    return None


class SQLiteProviderSearchCache:
    """Small optional SQLite cache for PHI-free provider-search metadata."""

    def __init__(self, database_path: Optional[Path] = None) -> None:
        self.database_path = database_path or resolve_provider_cache_path()
        self.enabled = self.database_path is not None

        if self.enabled:
            self._initialize()

    def get(self, cache_key: str) -> Optional[ProviderSearchCacheEntry]:
        if not self.enabled or self.database_path is None:
            return None

        query = (
            "SELECT cache_key, request_fingerprint, provider_ids_json, sources_json, stored_at, expires_at "
            "FROM provider_search_cache WHERE cache_key = ?"
        )

        try:
            with self._connect() as connection:
                row = connection.execute(query, (cache_key,)).fetchone()
        except sqlite3.Error as exc:
            logger.warning("Provider search cache read failed: %s", exc)
            return None

        if row is None:
            return None

        return ProviderSearchCacheEntry(
            cache_key=str(row["cache_key"]),
            request_fingerprint=str(row["request_fingerprint"]),
            provider_ids=tuple(json.loads(row["provider_ids_json"])),
            sources=tuple(json.loads(row["sources_json"])),
            stored_at=str(row["stored_at"]),
            expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
        )

    def set(self, entry: ProviderSearchCacheEntry) -> bool:
        if not self.enabled or self.database_path is None:
            return False

        statement = (
            "INSERT OR REPLACE INTO provider_search_cache "
            "(cache_key, request_fingerprint, provider_ids_json, sources_json, stored_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )

        parameters = (
            entry.cache_key,
            entry.request_fingerprint,
            json.dumps(list(entry.provider_ids)),
            json.dumps(list(entry.sources)),
            entry.stored_at,
            entry.expires_at,
        )

        try:
            with self._connect() as connection:
                connection.execute(statement, parameters)
        except sqlite3.Error as exc:
            logger.warning("Provider search cache write failed: %s", exc)
            return False

        return True

    def _initialize(self) -> None:
        if self.database_path is None:
            return

        statement = (
            "CREATE TABLE IF NOT EXISTS provider_search_cache ("
            "cache_key TEXT PRIMARY KEY,"
            "request_fingerprint TEXT NOT NULL,"
            "provider_ids_json TEXT NOT NULL,"
            "sources_json TEXT NOT NULL,"
            "stored_at TEXT NOT NULL,"
            "expires_at TEXT"
            ")"
        )

        try:
            with self._connect() as connection:
                connection.execute(statement)
        except sqlite3.Error as exc:
            logger.warning("Provider search cache disabled after initialization failure: %s", exc)
            self.enabled = False

    def _connect(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise sqlite3.OperationalError("Provider search cache path is not configured.")

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _prepare_cache_directory(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Provider search cache directory unavailable at %s: %s", directory, exc)
        return False

    return directory.is_dir() and os.access(directory, os.W_OK | os.X_OK)
