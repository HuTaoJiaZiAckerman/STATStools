"""Shared terminal and file logging for long-running STATStools commands."""

from datetime import datetime
import logging
from pathlib import Path
import shlex
import sys


def configure_logging(
    output_dir: Path | str,
    tool_name: str,
    context: str | None = None,
) -> Path:
    """Write identical UTF-8 logs to the terminal and output directory."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    name_parts = [tool_name]
    if context:
        name_parts.append(context)
    name_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_path = log_dir / ("_".join(name_parts) + ".log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logger = logging.getLogger(tool_name)
    logger.info("日志文件 %s", log_path)
    logger.info("命令行 %s", shlex.join(sys.argv))
    return log_path
