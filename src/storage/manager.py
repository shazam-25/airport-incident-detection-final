from src.storage.object_storage import LocalObjectStorage
from src.storage.redis_cache import RedisHotCache
from src.storage.postgres_logger import RelationalAuditLogger
import numpy as np
from typing import List, Optional

class StorageManager:
    def __init__(self):
        self.blob_store = LocalObjectStorage()
        self.hot_cache = RedisHotCache()
        self.cold_logger = RelationalAuditLogger()

    def process_and_store_event(
        self,
        stream_source: str,
        event_type: str,
        status_value: str,
        action_trigger: str,
        frame: np.ndarray,
        crop_box: Optional[List[int]] = None
    ):
        # 1. Save blob PNG image
        image_path = self.blob_store.save_violation_image(
            frame=frame,
            stream_name=stream_source,
            event_type=event_type,
            crop_box=crop_box
        )

        # 2. Cold Path Postgres / SQLite logging
        self.cold_logger.log_event(
            stream_source=stream_source,
            event_type=event_type,
            status_value=status_value,
            action_trigger=action_trigger,
            image_path=image_path
        )

        # 3. Hot path Redis Pub/Sub broadcast
        self.hot_cache.publish_alert(
            stream_source=stream_source,
            event_type=event_type,
            data={
                "status": status_value,
                "action": action_trigger,
                "image_path": image_path
            }
        )