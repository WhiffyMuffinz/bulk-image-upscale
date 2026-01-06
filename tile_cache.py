"""
Tile cache implementation using a temporary SQLite database.
Handles concurrent access and automatic cleanup.
"""

import logging
import os
import pickle
import sqlite3
import tempfile
import threading
import time
from typing import Dict, List, Optional


class RunLocalTileCache:
    """
    A temporary, run-local SQLite-based cache for storing upscaled tiles.
    Uses a tempfile that is automatically deleted on exit.
    Thread-safe.
    """

    def __init__(self):
        # Create a named temporary file
        # We set delete=False initially so we can close the file handle
        # and let SQLite open it by name, but we manage deletion manually/via destructor
        self._temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self._temp_file.name
        self._temp_file.close()  # Close the file handle so SQLite can open it

        logging.info(f"Initialized RunLocalTileCache at {self.db_path}")

        # Initialize DB
        self._init_db()

        # Thread-local storage for connections (SQLite connections can't be shared across threads)
        self.local = threading.local()

    def _get_conn(self):
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(
                self.db_path, timeout=60.0
            )  # Long timeout for concurrency
            # Enable WAL mode for better concurrency
            self.local.conn.execute("PRAGMA journal_mode=WAL;")
            self.local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self.local.conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    hash TEXT PRIMARY KEY,
                    data BLOB
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

    def get_batch(self, hashes: List[str]) -> Dict[str, bytes]:
        """
        Retrieve a batch of tiles from the cache.
        Returns a dictionary mapping hash -> data for found items.
        """
        if not hashes:
            return {}

        conn = self._get_conn()
        cursor = conn.cursor()

        # Split into chunks to avoid "too many SQL variables" error
        chunk_size = 900
        result = {}

        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT hash, data FROM tiles WHERE hash IN ({placeholders})"
            cursor.execute(query, chunk)
            result.update({row[0]: row[1] for row in cursor.fetchall()})

        return result

    def put_batch(self, items: Dict[str, bytes]):
        """
        Store a batch of tiles in the cache.
        Items is a dict mapping hash -> data.
        """
        if not items:
            return

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            # executemany is efficient
            cursor.executemany(
                "INSERT OR IGNORE INTO tiles (hash, data) VALUES (?, ?)",
                items.items(),
            )
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error writing to tile cache: {e}")

    def cleanup(self):
        """Close connections and remove the temporary file."""
        # Note: We can't easily close thread-local connections from here,
        # but the OS will clean up sockets on process exit.
        # We largely care about deleting the file.

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                # Also remove WAL keys if they exist
                if os.path.exists(self.db_path + "-wal"):
                    os.remove(self.db_path + "-wal")
                if os.path.exists(self.db_path + "-shm"):
                    os.remove(self.db_path + "-shm")
                logging.info(f"Cleaned up tile cache at {self.db_path}")
            except OSError as e:
                logging.warning(f"Failed to cleanup tile cache: {e}")

    def __del__(self):
        self.cleanup()
