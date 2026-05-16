#!/usr/bin/env python3
"""
Shared loguru configuration for the concomtorch CI orchestration scripts.

Every ``ci/`` entry point calls :func:`setup_logging` once at the top of
its ``main()``. The default stderr sink is replaced with two sinks: a
colorized stdout stream for interactive and journald capture, and a
rotating, compressed file under ``logs/`` for the long unattended daily
tick. Standard-library ``logging`` (used by ``cibuildwheel``,
``urllib3``, ``torch-wheel-index``, etc.) is routed into loguru through
an intercept handler so a run produces one coherent stream.

Exception frames are logged with ``backtrace`` but without ``diagnose``:
``ci/release.py`` handles ``GH_TOKEN``, and variable-value rendering in
tracebacks would serialize that secret into the log file.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent

_STDOUT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[component]}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[component]} | "
    "{name}:{function}:{line} - "
    "{message}"
)


class _InterceptHandler(logging.Handler):
    """
    Route standard-library ``logging`` records into loguru.

    This is the canonical loguru intercept handler: it maps the stdlib
    level to the loguru level by name (falling back to the numeric
    level) and rewinds the call stack so the originating module, not
    ``logging``, is attributed in the message.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Re-emit one stdlib record through the loguru logger.

        Parameters
        ----------
        record : logging.LogRecord
            The standard-library record to forward.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _resolve_level(level: str | None) -> str:
    """
    Resolve the effective log level.

    Parameters
    ----------
    level : str or None
        Explicit level passed by the caller. When None, the
        ``CONCOMTORCH_LOG_LEVEL`` environment variable is consulted,
        defaulting to ``"INFO"``.

    Returns
    -------
    str
        An uppercase loguru level name.
    """
    if level is not None:
        return level.upper()
    return os.environ.get("CONCOMTORCH_LOG_LEVEL", "INFO").upper()


def _resolve_log_dir(log_dir: Path | None) -> Path:
    """
    Resolve and create the directory that holds rotating log files.

    Parameters
    ----------
    log_dir : Path or None
        Explicit directory. When None, ``CONCOMTORCH_LOG_DIR`` is used,
        defaulting to ``<repo_root>/logs``.

    Returns
    -------
    Path
        The created log directory.
    """
    if log_dir is not None:
        resolved = log_dir
    else:
        env_dir = os.environ.get("CONCOMTORCH_LOG_DIR", "")
        resolved = Path(env_dir) if env_dir != "" else REPO_ROOT / "logs"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logging(
    component: str,
    *,
    level: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """
    Install the stdout and rotating-file loguru sinks for one CI script.

    The default stderr sink is removed first so output is not
    duplicated. The stdout sink is colorized when attached to a tty.
    The file sink rotates at 50 MB, keeps 14 days of history, and
    compresses rotated files as zip. Both sinks use ``enqueue=True`` so
    logging never blocks a long build and is safe across the threads the
    docker image pool uses. The standard-library root logger is
    redirected into loguru.

    Parameters
    ----------
    component : str
        Short tag identifying the calling script (e.g. ``"run"``,
        ``"build_wheel"``), shown in every line and used in the log file
        name.
    level : str, optional
        Minimum level. Defaults to ``CONCOMTORCH_LOG_LEVEL`` or
        ``"INFO"``.
    log_dir : Path, optional
        Directory for log files. Defaults to ``CONCOMTORCH_LOG_DIR`` or
        ``<repo_root>/logs``.

    Returns
    -------
    Path
        The active log file path for this component.
    """
    effective_level = _resolve_level(level)
    directory = _resolve_log_dir(log_dir)
    log_file = directory / f"concomtorch-{component}.log"

    logger.remove()
    logger.configure(extra={"component": component})

    logger.add(
        sys.stdout,
        level=effective_level,
        format=_STDOUT_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    logger.add(
        log_file,
        level=effective_level,
        format=_FILE_FORMAT,
        rotation="50 MB",
        retention="14 days",
        compression="zip",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    logger.info(
        "logging initialized: component={} level={} | writing rotating log file to {}",
        component,
        effective_level,
        log_file,
    )
    return log_file


def subprocess_log_path(
    component: str,
    *,
    tag: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """
    Build the path for a dedicated subprocess-transcript log file.

    Long child processes (the cibuildwheel build, repair, and
    in-container pytest run) emit thousands of lines. Those are streamed
    to the console for live progress and written verbatim to this file so
    the full transcript is preserved without burying the structured
    orchestration records in the component's rotating log file. The name
    carries a timestamp so concurrent or successive builds in one daily
    tick never overwrite each other's transcript.

    Parameters
    ----------
    component : str
        Short tag identifying the calling script (e.g. ``"build_wheel"``).
    tag : str, optional
        Extra identity folded into the filename (e.g.
        ``"cu124-torch2.6.0-cp312"``) so per-build transcripts are
        distinguishable.
    log_dir : Path, optional
        Directory for log files. Defaults to ``CONCOMTORCH_LOG_DIR`` or
        ``<repo_root>/logs``.

    Returns
    -------
    Path
        The transcript file path. The parent directory is created.
    """
    directory = _resolve_log_dir(log_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    middle = f"{component}-subprocess"
    if tag is not None:
        middle = f"{middle}-{tag}"
    return directory / f"concomtorch-{middle}-{stamp}.log"
