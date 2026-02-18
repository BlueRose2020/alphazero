import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any

import colorlog
from config import *


class RateLimitedLogger:
    """带频率限制的日志记录器包装器。"""

    def __init__(self, logger: logging.Logger, interval: float = 1.0):
        """
        Args:
            logger: 被包装的logger对象
            interval: 相同消息的最小输出间隔（秒）
        """
        self._logger = logger
        self._interval = interval
        self._last_log_time: dict[tuple[int, str], float] = {}

    def _should_log(self, level: int, msg: str) -> bool:
        """检查是否应该输出此日志。"""
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
        """将其他属性请求转发到内部logger。"""
        return getattr(self._logger, name)


_ANSI_COLOR_CODES: dict[str, str] = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "orange": "38;5;208",
    "pink": "38;5;213",
    "purple": "38;5;141",
    "gray": "90",
    "grey": "90",
    "light_blue": "38;5;117",
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}

_ANSI_BG_COLOR_CODES: dict[str, str] = {
    "bg_white": "47",
}


def colorize(
    text: str, color: str | tuple[int, int, int], *, bold: bool = False
) -> str:
    """为部分字符串设置颜色（仅对终端输出有效）。

    color 可传入颜色名（如 "red"）或 RGB 元组 (R, G, B)。
    """
    if isinstance(color, tuple):
        r, g, b = color
        if not all(0 <= v <= 255 for v in (r, g, b)):
            return text
        style = f"38;2;{r};{g};{b}"
        if bold:
            style = "1;" + style
        return f"\x1b[{style}m{text}\x1b[0m"

    code = _ANSI_COLOR_CODES.get(color)
    if not code:
        return text
    style = "1;" + code if bold else code
    return f"\x1b[{style}m{text}\x1b[0m"


class _StripAnsiFilter(logging.Filter):
    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._ansi_re.sub("", record.msg)
        return True


class _LevelColorFormatter(colorlog.ColoredFormatter):
    _reset = "\x1b[0m"

    def _build_log_color(self, record: logging.LogRecord) -> str:
        color_spec = self.log_colors.get(record.levelname, "")
        if not color_spec:
            return ""

        codes: list[str] = []
        for token in color_spec.split(","):
            token = token.strip()
            if not token:
                continue
            if token == "bold":
                codes.append("1")
                continue
            if token in _ANSI_COLOR_CODES:
                codes.append(_ANSI_COLOR_CODES[token])
                continue
            if token in _ANSI_BG_COLOR_CODES:
                codes.append(_ANSI_BG_COLOR_CODES[token])
                continue

        if not codes:
            return ""
        return f"\x1b[{';'.join(codes)}m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        level_color = self._build_log_color(record)
        if not level_color:
            return message

        message = message.replace(self._reset, f"{self._reset}{level_color}")
        if not message.endswith(self._reset):
            message = f"{message}{self._reset}"
        return message


def _get_caller_name() -> str:
    for frame_info in inspect.stack()[2:]:
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
    """设置彩色日志记录器。

    Args:
        name: logger名称，None时自动使用调用文件名
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None则不保存到文件
        console: 是否输出到控制台
        rate_limit: 频率限制间隔（秒），None表示不限制

    Returns:
        配置好的logger（如果设置了rate_limit则返回包装后的logger）
    """
    logger_name = _get_caller_name() if name is None else name

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()  # 清除已有的handler

    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%H:%M:%S"

    # 彩色控制台输出
    if console:
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(level)

        color_formatter = _LevelColorFormatter(
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

        # 移除ANSI颜色控制符，避免写入日志文件
        file_handler.addFilter(_StripAnsiFilter())

        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 如果需要频率限制，包装logger
    if rate_limit is not None:
        return RateLimitedLogger(logger, interval=rate_limit)

    return logger
