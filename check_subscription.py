"""
检查当前 GitHub Copilot 订阅信息。

使用方法：
    python check_subscription.py --token <your_github_token>

或设置环境变量 GITHUB_TOKEN 后直接运行：
    python check_subscription.py
"""

import argparse
import os
import urllib.request
import urllib.error
import json
import sys


def check_copilot_subscription(token: str) -> None:
    """通过 GitHub API 查询并打印当前 GitHub Copilot 订阅信息。

    Args:
        token: GitHub 个人访问令牌（需要 `read:user` 或 `manage_billing:copilot` 权限）。
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. 获取当前用户信息
    user_url = "https://api.github.com/user"
    req = urllib.request.Request(user_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            user_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"获取用户信息失败：HTTP {e.code} {e.reason}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"网络错误：{e.reason}")
        sys.exit(1)

    login = user_data.get("login", "未知用户")
    print(f"当前 GitHub 用户：{login}")

    # 2. 查询个人 Copilot 席位详情（/user/copilot_seat_details）
    seat_url = "https://api.github.com/user/copilot_seat_details"
    req = urllib.request.Request(seat_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            seat_data = json.loads(resp.read().decode())

        plan_type = seat_data.get("plan_type", "未知")
        assignee = seat_data.get("assignee", {}).get("login", login)
        pending_cancellation = seat_data.get("pending_cancellation_date")
        last_activity = seat_data.get("last_activity_at", "无记录")

        print(f"\n=== GitHub Copilot 订阅信息 ===")
        print(f"用户：{assignee}")
        print(f"订阅计划：{plan_type}")
        print(f"最近活动时间：{last_activity}")
        if pending_cancellation:
            print(f"订阅将于以下日期取消：{pending_cancellation}")

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "\n未检测到 GitHub Copilot 订阅。\n"
                "请前往 https://github.com/settings/copilot 查看或激活订阅。"
            )
        elif e.code == 403:
            print(
                "\n无权限查询 Copilot 席位信息。\n"
                "请确保令牌具有 read:user 权限，并已激活 GitHub Copilot 订阅。\n"
                "可前往 https://github.com/settings/copilot 查看订阅状态。"
            )
        else:
            print(f"查询 Copilot 订阅失败：HTTP {e.code} {e.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查询当前 GitHub Copilot 订阅信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub 个人访问令牌（默认读取环境变量 GITHUB_TOKEN）",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "请通过 --token 参数或环境变量 GITHUB_TOKEN 提供 GitHub 个人访问令牌。\n"
            "可在 https://github.com/settings/tokens 创建令牌（需要 read:user 权限）。"
        )

    check_copilot_subscription(args.token)


if __name__ == "__main__":
    main()
