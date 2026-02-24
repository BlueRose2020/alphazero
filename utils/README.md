# 工具与基础组件

包含经验池、日志、共享缓冲区等基础工具。

- experience_pool.py：经验池
- share_ring_buffer.py：共享环形缓冲区
- data_enhancer.py：数据增强器（仅提供基础的旋转和翻转，具体请打开文件查看说明）
- history_manager.py：历史记录管理
- logger.py：日志封装

在logger.py中提供了**colorize**API，参数如下：

```python
def colorize(
    text: str, 
    color: str | tuple[int, int, int], 
    bold: bool = False
) -> str:
```

说明：

- `color` 可以传入预设颜色名（如 `"red"`, `"green"`, `"yellow"`, `"cyan"`, `"bright_magenta"` 等，具体参见 logger 内 `_ANSI_COLOR_CODES`）或 0-255 的 RGB 元组。
- 传入 `bold=True` 会在颜色的基础上叠加粗体强调效果，常用于高亮关键信息。
- 不支持的颜色名或非法 RGB 会原样返回原始文本，避免输出异常控制符。

示例：

```python
from utils.logger import colorize

msg = colorize("[Self-Play]", "cyan", bold=True)
score_line = colorize("win: 12 / lose: 8", (255, 165, 0))
print(msg, score_line)
```

输出：
![](/src/pic/logger_example1.png)

对于logger而言，该API仅修改其参数部分字符串的颜色，其余部分仍会显示当前logger等级对应的颜色，例如：

```python
from utils.logger import setup_logger, colorize

logger = setup_logger(__name__)

msg1 = colorize("这是消息1", "red")
msg2 = "这是消息2"

logger.info(msg1 + msg2)
```

输出：
![](/src/pic/logger_example2.png)
