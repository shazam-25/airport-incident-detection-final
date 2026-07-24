import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List

# Automatically identify project root directory (two levels up from src/storage/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class RelationalAuditLogger:
    """
    Cold Path Evaluation Layer.
    Provides structured database logging for events, timestamps, and compliance metrics.
    """
    def __init__(self, db_path: str = "storage/incident_audit.db"):
        self.db_path = PROJECT_ROOT / db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incident_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    stream_source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status_value TEXT NOT NULL,
                    action_trigger TEXT NOT NULL,
                    image_path TEXT
                )
            """)
            conn.commit()
    
    def log_event(
        self,
        stream_source: str,
        event_type: str,
        status_value: str,
        action_trigger: str,
        image_path: str = ""
    ):
        """
        Inserts a structured log record into the relational event table.
        """
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incident_logs (timestamp, stream_source, event_type, status_value, action_trigger, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp_str, stream_source, event_type, status_value, action_trigger, image_path))
            conn.commit()
    
    def fetch_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, stream_source, event_type, status_value, action_trigger, image_path
                FROM incident_logs
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]