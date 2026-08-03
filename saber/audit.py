import json
import logging
import threading
from pathlib import Path
from saber.signal import BaseSignal

logger = logging.getLogger("AuditLedger")

class AuditLedger:
    """
    Thread-safe, append-only JSON-Lines audit log.
    Records every query's full lifecycle to logs/audit.jsonl.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, log_path: str = "logs/audit.jsonl"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AuditLedger, cls).__new__(cls)
                cls._instance._init(log_path)
            return cls._instance
            
    def _init(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file exists
        self.log_path.touch(exist_ok=True)
        self.file_lock = threading.Lock()
        
    def log_signal(self, signal: BaseSignal):
        """
        Records a SABER signal into the audit ledger.
        Ensures the signal is frozen and hashed before logging.
        """
        if not signal.integrity_hash:
            signal.freeze_and_hash()
            
        payload = signal.model_dump()
        payload["_signal_type"] = signal.__class__.__name__
        
        with self.file_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")
                
    def log_event(self, event_type: str, details: dict):
        """
        Helper method to log standard events without instantiating a full AUDIT_SIGNAL manually.
        """
        from saber.signal import AUDIT_SIGNAL
        import uuid
        
        signal = AUDIT_SIGNAL(
            signal_id=f"audit_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            details=details
        )
        self.log_signal(signal)

# Global singleton instance for easy imports
audit_ledger = AuditLedger()
