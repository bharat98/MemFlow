"""
MemFlow File Watcher
Sub-Task 2.4: Monitor vault for changes and auto-update index

Uses watchdog to monitor the Obsidian vault for .md file changes,
debounces updates over a 30-second window, and re-indexes only changed files.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Set, Optional
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from index_vault import (
    load_index,
    update_documents,
    VAULT_PATH,
    INDEX_STORAGE,
    build_exclude_paths,
    is_excluded_path,
    get_memflow_config,
    is_included_path,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEBOUNCE_SECONDS = int(os.environ.get("DEBOUNCE_SECONDS", "30"))
WATCH_EXTENSIONS = {".md"}


class VaultWatcher(FileSystemEventHandler):
    """
    Watches Obsidian vault for markdown file changes.
    Batches changes and re-indexes after debounce period.
    """

    def __init__(self, vault_path: str = VAULT_PATH):
        super().__init__()
        self.vault_path = vault_path
        self.pending_changes: Set[str] = set()
        self.last_change_time: float = 0
        self.index = None
        self.exclude_paths = build_exclude_paths(Path(vault_path))
        self.config = get_memflow_config()
        self._lock = threading.Lock()
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the file watcher."""
        logger.info(f"Loading index from {INDEX_STORAGE}...")
        self.index = load_index()

        if self.index is None:
            logger.error("No index found. Please run index_vault.py first.")
            sys.exit(1)

        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        logger.info(f"Watching vault: {self.vault_path}")
        logger.info(f"Debounce period: {DEBOUNCE_SECONDS} seconds")

    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=5)
        logger.info("File watcher stopped")

    def _is_valid_file(self, path: str) -> bool:
        """Check if the file should be watched."""
        if not path:
            return False

        path_obj = Path(path)

        # Check extension
        if path_obj.suffix.lower() not in WATCH_EXTENSIONS:
            return False

        # Ignore hidden files and directories
        if any(part.startswith('.') for part in path_obj.parts):
            return False

        # Ignore excluded paths
        if is_excluded_path(path_obj, Path(self.vault_path), self.exclude_paths):
            return False
        if not is_included_path(path_obj, Path(self.vault_path), self.config):
            return False

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        if self._is_valid_file(event.src_path):
            with self._lock:
                self.pending_changes.add(event.src_path)
                self.last_change_time = time.time()
            logger.debug(f"File modified: {event.src_path}")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        if self._is_valid_file(event.src_path):
            with self._lock:
                self.pending_changes.add(event.src_path)
                self.last_change_time = time.time()
            logger.debug(f"File created: {event.src_path}")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return

        if self._is_valid_file(event.src_path):
            with self._lock:
                self.pending_changes.add(event.src_path)
                self.last_change_time = time.time()
            logger.debug(f"File deleted: {event.src_path}")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events."""
        if event.is_directory:
            return

        # Handle source file (treat as delete)
        if self._is_valid_file(event.src_path):
            with self._lock:
                self.pending_changes.add(event.src_path)
                self.last_change_time = time.time()

        # Handle destination file (treat as create)
        if hasattr(event, 'dest_path') and self._is_valid_file(event.dest_path):
            with self._lock:
                self.pending_changes.add(event.dest_path)
                self.last_change_time = time.time()

        logger.debug(f"File moved: {event.src_path} -> {getattr(event, 'dest_path', 'unknown')}")

    def _process_loop(self) -> None:
        """Background thread to process batched changes."""
        while self._running:
            time.sleep(5)  # Check every 5 seconds
            self._check_and_process()

    def _check_and_process(self) -> None:
        """Check if debounce period has passed and process pending changes."""
        with self._lock:
            if not self.pending_changes:
                return

            time_since_last_change = time.time() - self.last_change_time
            if time_since_last_change < DEBOUNCE_SECONDS:
                remaining = DEBOUNCE_SECONDS - time_since_last_change
                logger.debug(f"Waiting {remaining:.0f}s before processing {len(self.pending_changes)} changes")
                return

            # Copy and clear pending changes
            changes_to_process = list(self.pending_changes)
            self.pending_changes.clear()

        # Process outside the lock
        self._process_changes(changes_to_process)

    def _process_changes(self, filepaths: list) -> None:
        """Process a batch of file changes."""
        if not filepaths:
            return

        logger.info(f"Processing {len(filepaths)} file changes...")
        start_time = time.time()

        try:
            self.index = update_documents(filepaths, self.index)
            elapsed = time.time() - start_time
            logger.info(f"✅ Updated index in {elapsed:.1f}s")

            # Log updated files
            for fp in filepaths:
                status = "updated" if Path(fp).exists() else "removed"
                logger.info(f"  - {Path(fp).name}: {status}")

        except Exception as e:
            logger.error(f"Error processing changes: {e}")

    def get_stats(self) -> dict:
        """Get current watcher statistics."""
        with self._lock:
            return {
                "vault_path": self.vault_path,
                "pending_changes": len(self.pending_changes),
                "last_change_time": datetime.fromtimestamp(self.last_change_time).isoformat() if self.last_change_time else None,
                "debounce_seconds": DEBOUNCE_SECONDS,
                "running": self._running,
            }


def run_watcher(vault_path: str = VAULT_PATH) -> None:
    """
    Run the file watcher as a standalone process.

    Args:
        vault_path: Path to the Obsidian vault to watch
    """
    watcher = VaultWatcher(vault_path)
    watcher.start()

    observer = Observer()
    observer.schedule(watcher, vault_path, recursive=True)
    observer.start()

    logger.info("=" * 50)
    logger.info("MemFlow File Watcher Started")
    logger.info("=" * 50)
    logger.info(f"Vault: {vault_path}")
    logger.info(f"Debounce: {DEBOUNCE_SECONDS}s")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        observer.stop()
        watcher.stop()

    observer.join()
    logger.info("File watcher stopped")


# Global watcher instance for integration with MCP server
_watcher: Optional[VaultWatcher] = None
_observer: Optional[Observer] = None


def start_background_watcher(vault_path: str = VAULT_PATH) -> VaultWatcher:
    """
    Start the file watcher in the background (for use with MCP server).

    Returns:
        VaultWatcher instance
    """
    global _watcher, _observer

    if _watcher is not None:
        logger.info("Watcher already running")
        return _watcher

    _watcher = VaultWatcher(vault_path)
    _watcher.start()

    _observer = Observer()
    _observer.schedule(_watcher, vault_path, recursive=True)
    _observer.start()

    logger.info(f"Background watcher started for: {vault_path}")
    return _watcher


def stop_background_watcher() -> None:
    """Stop the background file watcher."""
    global _watcher, _observer

    if _observer:
        _observer.stop()
        _observer.join(timeout=5)
        _observer = None

    if _watcher:
        _watcher.stop()
        _watcher = None

    logger.info("Background watcher stopped")


def get_watcher() -> Optional[VaultWatcher]:
    """Get the current watcher instance."""
    return _watcher


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch Obsidian vault for changes")
    parser.add_argument(
        "--vault",
        type=str,
        default=VAULT_PATH,
        help="Path to Obsidian vault"
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=DEBOUNCE_SECONDS,
        help="Debounce period in seconds"
    )
    args = parser.parse_args()

    if args.debounce:
        DEBOUNCE_SECONDS = args.debounce

    run_watcher(args.vault)
