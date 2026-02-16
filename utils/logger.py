import logging
import inspect
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import colorlog
from config import *


class RateLimitedLogger:
    """带频率限制的日志记录器包装器"""

    def __init__(self, logger: logging.Logger, interval: float = 1.0):
        """
        Args:
            logger: 被包装的logger对象
            interval: 相同消息的最小输出间隔（秒）
        """
        self._logger = logger
        self._interval = interval
        self._last_log_time: Dict[Tuple[int, str], float] = {}

    def _should_log(self, level: int, msg: str) -> bool:
        """检查是否应该输出此日志"""
        key = (level, msg)
        current_time = time.time()
        last_time = self._last_log_time.get(key, 0)

        if current_time - last_time >= self._interval:
            self._last_log_time[key] = current_time
            return True
        return False

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(logging.DEBUG, msg):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(logging.INFO, msg):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(logging.WARNING, msg):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        # 错误和严重错误总是输出
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        # 错误和严重错误总是输出
        self._logger.critical(msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """将其他属性请求转发到内部logger"""
        return getattr(self._logger, name)


def _get_caller_name() -> str:
    stack = inspect.stack()
    for frame_info in stack[2:]:
        module = inspect.getmodule(frame_info.frame)
        if module is not None and module.__name__ != __name__:
            return Path(module.__file__ or module.__name__).stem
    return "alphazero"


def setup_logger(
    name: str | None = None,
    level: int = DEFAULT_LOG_LEVEL,
    log_file: str | None = None,
    console: bool = True,
    rate_limit: float | None = None,
) -> logging.Logger | RateLimitedLogger:
    """设置彩色日志记录器

    Args:
        name: logger名称，None时自动使用调用文件名
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None则不保存到文件
        console: 是否输出到控制台
        rate_limit: 频率限制间隔（秒），None表示不限制

    Returns:
        配置好的logger（如果设置了rate_limit则返回包装后的logger）
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

    # 如果需要频率限制，包装logger
    if rate_limit is not None:
        return RateLimitedLogger(logger, interval=rate_limit)

    return logger
