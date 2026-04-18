"""Edge service entrypoint for coordinator + processor."""

from __future__ import annotations

import threading

from .config import settings
from .coordinator_service import app as coordinator_app
from .processor_service import app as processor_app
from .utils import configure_logging


def _run_processor() -> None:
    processor_app.run(
        host=settings.processor_host,
        port=settings.processor_port,
        threaded=True,
        use_reloader=False,
    )


def _run_coordinator() -> None:
    coordinator_app.run(
        host=settings.coordinator_host,
        port=settings.coordinator_port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    logger = configure_logging("edge-main")
    logger.info("starting processor service")
    processor_thread = threading.Thread(target=_run_processor, daemon=True)
    processor_thread.start()
    logger.info("starting coordinator service")
    _run_coordinator()
