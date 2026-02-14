import logging
import inspect
from pathlib import Path

import colorlog
from config import *


def _get_caller_name() -> str:
    stack = inspect.stack()
    for frame_info in stack[2:]:
        module = inspect.getmodule(frame_info.frame)
        if module is not None and module.__name__ != __name__:
            return Path(module.__file__ or module.__name__).stem
    return "alphazero"


def setup_logger(
    name: str | None = None,
    level: int = DEFULT_LOG_LEVEL,
    log_file: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """设置彩色日志记录器

    Args:
        name: logger名称，None时自动使用调用文件名
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None则不保存到文件
        console: 是否输出到控制台

    Returns:
        配置好的logger
    """
    if name is None:
        name = _get_caller_name()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 清除已有的handler

    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 彩色控制台输出
    if console:
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(level)

        color_formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=date_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    # 文件输出（无颜色）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
