import logging
import sys
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    OPTIONAL_FIELDS = (
        "session_id",
        "es_query",
        "safety_status",
        "error_type",
        "error_detail",
        "result_count",
    )
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now().isoformat() + "Z",
            "level": record.levelname,
            "service": "backend",
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields if present (e.g., session_id, query_hash)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        for field in self.OPTIONAL_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging() -> None:
    """Configure structured JSON logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.error").propagate = True
    logging.getLogger("uvicorn.access").propagate = True