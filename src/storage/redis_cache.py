import json
import time
from typing import Dict, Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class RedisHotCache:
    """
    Hot Path Evaluation Layer.
    Publishes sub-2ms telemetry broadcats and volatile alert states to the frontend.
    """
    def __init__(self, host: str="localhost", port: int = 6379, channel: str: "aiport_alerts"):
        self.channel = channel
        self.client = None
        self.fallback_memory_cache = {}

        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(host=host, port=port, db=0, socket_connect_timeout=1)
                self.client.ping()
                print("⚡ [Redis Cache] Connected successfully.")
            except Exception as e:
                print(f"⚠️ [Redis Cache] Connection failed ({e}). Falling back to in-memory cache.")
                self.client = None
        else:
            print("⚠️ [Redis Cache] redis-py module not installed. Using in-memory fallback.")
        
    def publish_alert(self, stream_source: str, event_type:str, data: Dict[str, Any]):
        """
        Broadcasts live incident payload over Redis Pub/Sub channel.
        """
        payload = {
            "timestamp": time.strftime("%H:%M:%S"),
            "unix_time": time.time(),
            "stream_source": stream_source,
            "event_type": event_type,
            "data": data
        }

        json_payload = json.dumps(payload)

        # In-memory fallback tracking
        self.fallback_memory_cache[stream_source] = payload

        if self.client:
            try:
                self.client.publish(self.channel, json_payload)
                self.client.set(f"latest_alert:{stream_source}", json_payload, ex=60)
            except Exception as e:
                print(f"❌ Radis publish error: {e}")
    
    def get_latest_state(self, stream_source: str) -> Optional[Dict[str, Any]]:
        if self.client:
            try:
                val = self.client.get(f"latest_alert:{stream_source}")
                if val:
                    return json.loads(val.decode('utf-8'))
            except Exception:
                pass
        return self.fallback_memory_cache.get(stream_source)