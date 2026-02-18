SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 当前选择的主题（可修改此变量切换主题）
CURRENT_THEME = "wood"  # 可选值: "dark", "light", "ocean", "forest", "wood", "sunset", "neon", "mint"

# 主题配置字典
THEMES = {
    "dark": {
        "background": (15, 15, 20),
        "board_bg": (45, 50, 60),
        "grid": (70, 75, 85),
        "primary": (100, 200, 255),
        "danger": (255, 120, 120),
        "text": (240, 240, 245),
    },
    "light": {
        "background": (240, 240, 245),
        "board_bg": (220, 225, 230),
        "grid": (150, 150, 160),
        "primary": (30, 120, 220),
        "danger": (220, 50, 50),
        "text": (20, 20, 25),
    },
    "ocean": {
        "background": (10, 20, 35),
        "board_bg": (25, 45, 70),
        "grid": (60, 90, 130),
        "primary": (100, 220, 255),
        "danger": (255, 150, 100),
        "text": (235, 245, 255),
    },
    "forest": {
        "background": (15, 25, 15),
        "board_bg": (35, 55, 40),
        "grid": (70, 100, 75),
        "primary": (100, 220, 150),
        "danger": (255, 100, 100),
        "text": (235, 245, 235),
    },
    "wood": {
        "background": (25, 18, 12),
        "board_bg": (139, 90, 43),
        "grid": (101, 67, 33),
        "primary": (0, 0, 0),
        "danger": (255, 255, 255),
        "text": (245, 240, 235),
    },
    "sunset": {
        "background": (40, 20, 30),
        "board_bg": (180, 100, 60),
        "grid": (150, 80, 50),
        "primary": (255, 220, 100),
        "danger": (255, 150, 150),
        "text": (255, 245, 235),
    },
    "neon": {
        "background": (0, 0, 10),
        "board_bg": (20, 20, 40),
        "grid": (80, 80, 150),
        "primary": (0, 255, 150),
        "danger": (255, 0, 150),
        "text": (200, 255, 200),
    },
    "mint": {
        "background": (20, 35, 30),
        "board_bg": (60, 140, 120),
        "grid": (80, 160, 140),
        "primary": (150, 255, 220),
        "danger": (255, 120, 150),
        "text": (240, 250, 245),
    },
}


# ====================================================================
# 以下内容请勿修改，除非你知道自己在做什么，否则可能会导致程序无法运行
# ====================================================================
# 根据选择的主题设置颜色
_theme = THEMES[CURRENT_THEME]
BACKGROUND_COLOR = _theme["background"]
BOARD_BG_COLOR = _theme["board_bg"]
GRID_COLOR = _theme["grid"]
PRIMARY_COLOR = _theme["primary"]
DANGER_COLOR = _theme["danger"]
TEXT_COLOR = _theme["text"]
