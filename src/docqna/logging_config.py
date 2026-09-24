from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path

from docqna.constants import LOG_DIR
from docqna.settings import get_settings


_REQUEST_ID: ContextVar[str] = ContextVar(
    "docqna_request_id",
    default="-",
)

_CONFIGURED = False


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every DocQ&A log record."""

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        setattr(
            record,
            "request_id",
            _REQUEST_ID.get(),
        )
        return True


def set_request_id(
    request_id: str,
) -> Token[str]:
    """Set the request ID for the current execution context."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(
    token: Token[str],
) -> None:
    """Restore the previous request ID context."""

    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    """Return the current request ID."""

    return _REQUEST_ID.get()


def configure_logging(
    *,
    force: bool = False,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """
    Configure DocQ&A application logging.

    Only the ``docqna`` logger hierarchy is configured so
    third-party libraries keep their own logging behavior.
    """

    global _CONFIGURED

    logger = logging.getLogger("docqna")

    if _CONFIGURED and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    settings = get_settings()

    level_name = settings.log_level.strip().upper()
    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    logger.setLevel(level)
    logger.propagate = False

    request_filter = RequestIdFilter()

    formatter = logging.Formatter(
        (
            "%(asctime)s | %(levelname)s | %(name)s | "
            "request_id=%(request_id)s | %(message)s"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)
    logger.addHandler(console_handler)

    if settings.log_to_file:
        target_dir = (
            Path(log_dir)
            if log_dir is not None
            else LOG_DIR
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = RotatingFileHandler(
            filename=(
                target_dir / "docqna.log"
            ),
            maxBytes=max(
                1024,
                settings.log_max_bytes,
            ),
            backupCount=max(
                1,
                settings.log_backup_count,
            ),
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_filter)
        logger.addHandler(file_handler)

    _CONFIGURED = True

    if not isinstance(
        getattr(logging, level_name, None),
        int,
    ):
        logger.warning(
            "invalid_log_level configured=%s fallback=INFO",
            settings.log_level,
        )

    return logger

def shutdown_logging() -> None:
    """
    Close DocQ&A handlers.

    Primarily useful for clean process shutdown and tests
    that place log files inside temporary directories.
    """

    global _CONFIGURED

    logger = logging.getLogger(
        "docqna"
    )

    for handler in list(
        logger.handlers
    ):
        handler.close()
        logger.removeHandler(
            handler
        )

    _CONFIGURED = False

