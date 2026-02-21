"""点格棋游戏实现 (Dots and Boxes)"""

from __future__ import annotations

import torch

from config import *
from games.base import BaseGame
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DotsAndBoxesGame(BaseGame):
    """点格棋 (Dots and Boxes)

    状态表示 (三维张量，不包含玩家通道):
        state.shape = (3, DOT_ROWS, DOT_COLS)
        - channel 0: 水平边 (valid: [0:DOT_ROWS, 0:BOX_COLS])
        - channel 1: 垂直边 (valid: [0:BOX_ROWS, 0:DOT_COLS])
        - channel 2: 盒子归属 (valid: [0:BOX_ROWS, 0:BOX_COLS])

    注意:
        为保证 MCTS 可推断当前玩家，本实现将当前玩家标记存放在
        state[2, BOX_ROWS, BOX_COLS] 位置，该位置不参与盒子归属。
    """

    BOX_ROWS = 3
    BOX_COLS = 3

    DOT_ROWS = BOX_ROWS + 1
    DOT_COLS = BOX_COLS + 1

    H_EDGE_COUNT = DOT_ROWS * BOX_COLS
    V_EDGE_COUNT = BOX_ROWS * DOT_COLS
    NUM_ACTION = H_EDGE_COUNT + V_EDGE_COUNT

    STATE_SHAPE = (3, DOT_ROWS, DOT_COLS)

    @staticmethod
    def initial_state() -> TensorGameState:
        state = torch.zeros(DotsAndBoxesGame.STATE_SHAPE, dtype=torch.float32)
        state[2, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS] = float(PLAYER1)
        return state

    @staticmethod
    def current_player(state: TensorGameState) -> int:
        value = int(state[2, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS].item())
        return value if value in (PLAYER1, PLAYER2) else PLAYER1

    @staticmethod
    def next_state(state: TensorGameState, action: int) -> tuple[TensorGameState, int]:
        if action < 0 or action >= DotsAndBoxesGame.NUM_ACTION:
            raise ValueError("action 越界")

        edge_type, r, c = DotsAndBoxesGame._action_to_edge(action)
        next_state = state.clone()

        if edge_type == "H":
            if next_state[0, r, c].item() != 0:
                raise ValueError("非法落子")
            next_state[0, r, c] = 1.0
        else:
            if next_state[1, r, c].item() != 0:
                raise ValueError("非法落子")
            next_state[1, r, c] = 1.0

        current_player = DotsAndBoxesGame.current_player(state)
        completed = DotsAndBoxesGame._update_boxes(next_state, edge_type, r, c, current_player)

        next_player = current_player if completed else (-current_player)
        next_state[2, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS] = float(next_player)
        return next_state, next_player

    @staticmethod
    def legal_action_mask(state: TensorGameState) -> TensorActions:
        mask = torch.zeros((1, DotsAndBoxesGame.NUM_ACTION), dtype=torch.float32)
        # horizontal edges
        for r in range(DotsAndBoxesGame.DOT_ROWS):
            for c in range(DotsAndBoxesGame.BOX_COLS):
                if state[0, r, c].item() == 0:
                    action = DotsAndBoxesGame._edge_to_action("H", r, c)
                    mask[0, action] = 1.0
        # vertical edges
        for r in range(DotsAndBoxesGame.BOX_ROWS):
            for c in range(DotsAndBoxesGame.DOT_COLS):
                if state[1, r, c].item() == 0:
                    action = DotsAndBoxesGame._edge_to_action("V", r, c)
                    mask[0, action] = 1.0
        return mask

    @staticmethod
    def is_terminal(state: TensorGameState) -> bool:
        h_sum = torch.sum(state[0, : DotsAndBoxesGame.DOT_ROWS, : DotsAndBoxesGame.BOX_COLS]).item()
        v_sum = torch.sum(state[1, : DotsAndBoxesGame.BOX_ROWS, : DotsAndBoxesGame.DOT_COLS]).item()
        return int(h_sum + v_sum) == DotsAndBoxesGame.NUM_ACTION

    @staticmethod
    def _get_winner(state: TensorGameState) -> int | None:
        if not DotsAndBoxesGame.is_terminal(state):
            return None
        boxes = state[2, : DotsAndBoxesGame.BOX_ROWS, : DotsAndBoxesGame.BOX_COLS]
        p1 = int(torch.sum(boxes == PLAYER1).item())
        p2 = int(torch.sum(boxes == PLAYER2).item())
        if p1 > p2:
            return PLAYER1
        if p2 > p1:
            return PLAYER2
        return None

    @staticmethod
    def _action_to_edge(action: int) -> tuple[str, int, int]:
        if action < DotsAndBoxesGame.H_EDGE_COUNT:
            r = action // DotsAndBoxesGame.BOX_COLS
            c = action % DotsAndBoxesGame.BOX_COLS
            return "H", r, c
        idx = action - DotsAndBoxesGame.H_EDGE_COUNT
        r = idx // DotsAndBoxesGame.DOT_COLS
        c = idx % DotsAndBoxesGame.DOT_COLS
        return "V", r, c

    @staticmethod
    def _edge_to_action(edge_type: str, r: int, c: int) -> int:
        if edge_type == "H":
            return r * DotsAndBoxesGame.BOX_COLS + c
        return DotsAndBoxesGame.H_EDGE_COUNT + r * DotsAndBoxesGame.DOT_COLS + c

    @staticmethod
    def _is_box_complete(state: TensorGameState, r: int, c: int) -> bool:
        top = state[0, r, c].item() != 0
        bottom = state[0, r + 1, c].item() != 0
        left = state[1, r, c].item() != 0
        right = state[1, r, c + 1].item() != 0
        return top and bottom and left and right

    @staticmethod
    def _update_boxes(
        state: TensorGameState, edge_type: str, r: int, c: int, player: int
    ) -> bool:
        completed = False
        if edge_type == "H":
            # box above
            if r > 0:
                br, bc = r - 1, c
                if state[2, br, bc].item() == 0 and DotsAndBoxesGame._is_box_complete(state, br, bc):
                    state[2, br, bc] = float(player)
                    completed = True
            # box below
            if r < DotsAndBoxesGame.BOX_ROWS:
                br, bc = r, c
                if state[2, br, bc].item() == 0 and DotsAndBoxesGame._is_box_complete(state, br, bc):
                    state[2, br, bc] = float(player)
                    completed = True
        else:
            # box left
            if c > 0:
                br, bc = r, c - 1
                if state[2, br, bc].item() == 0 and DotsAndBoxesGame._is_box_complete(state, br, bc):
                    state[2, br, bc] = float(player)
                    completed = True
            # box right
            if c < DotsAndBoxesGame.BOX_COLS:
                br, bc = r, c
                if state[2, br, bc].item() == 0 and DotsAndBoxesGame._is_box_complete(state, br, bc):
                    state[2, br, bc] = float(player)
                    completed = True
        return completed

    @staticmethod
    def get_enhanced_data(
        state: NNState, policy: TensorActions, value: TensorValue
    ) -> list[ExperienceDate]:
        """点格棋自定义数据增强

        需要保持玩家标记位在 state[2, BOX_ROWS, BOX_COLS] 不随增强移动。
        """
        enhanced: list[ExperienceDate] = []

        for k in range(4):
            rotated_state = DotsAndBoxesGame._transform_nn_state(state, k, False)
            rotated_policy = DotsAndBoxesGame._transform_policy(policy, k, False)
            enhanced.append((rotated_state, rotated_policy, value))

            flipped_state = DotsAndBoxesGame._transform_nn_state(state, k, True)
            flipped_policy = DotsAndBoxesGame._transform_policy(policy, k, True)
            enhanced.append((flipped_state, flipped_policy, value))
        return enhanced


    @staticmethod
    def _transform_nn_state(state: NNState, k: int, flip: bool) -> NNState:
        batch, channels, height, width = state.shape
        if (height, width) != (DotsAndBoxesGame.DOT_ROWS, DotsAndBoxesGame.DOT_COLS):
            raise ValueError("DotsAndBoxes 状态尺寸不匹配")

        base_channels = DotsAndBoxesGame.STATE_SHAPE[0]
        has_player_channel = (channels - 1) % base_channels == 0
        if has_player_channel:
            base_channel_count = channels - 1
            player_channel = state[:, -1:, :, :]
        elif channels % base_channels == 0:
            base_channel_count = channels
            player_channel = None
        else:
            raise ValueError("DotsAndBoxes 状态通道数不匹配")

        blocks = base_channel_count // base_channels
        out = torch.zeros_like(state)

        for b in range(blocks):
            start = b * base_channels
            end = start + base_channels
            for n in range(batch):
                transformed = DotsAndBoxesGame._transform_board(
                    state[n, start:end, :, :], k, flip
                )
                out[n, start:end, :, :] = transformed

        if player_channel is not None:
            out[:, -1:, :, :] = player_channel

        if state.dim() == 3:
            return out.squeeze(0).detach().clone()
        return out.detach().clone()

    @staticmethod
    def _transform_policy(policy: TensorActions, k: int, flip: bool) -> TensorActions:
        if policy.dim() == 1:
            policy_in = policy.unsqueeze(0)
            squeeze_back = True
        elif policy.dim() == 2:
            policy_in = policy
            squeeze_back = False
        else:
            raise ValueError("DotsAndBoxes 仅支持 1D/2D 的 policy")

        mapped = torch.zeros_like(policy_in)
        for b in range(policy_in.shape[0]):
            for action in range(DotsAndBoxesGame.NUM_ACTION):
                edge_type, r, c = DotsAndBoxesGame._action_to_edge(action)
                new_type, nr, nc = DotsAndBoxesGame._transform_edge(edge_type, r, c, k, flip)
                new_action = DotsAndBoxesGame._edge_to_action(new_type, nr, nc)
                mapped[b, new_action] = policy_in[b, action]

        if squeeze_back:
            return mapped.squeeze(0).detach().clone()
        return mapped.detach().clone()

    @staticmethod
    def _transform_board(board: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
        transformed = torch.zeros_like(board)

        player_marker = board[
            2, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS
        ].item()

        for r in range(DotsAndBoxesGame.DOT_ROWS):
            for c in range(DotsAndBoxesGame.BOX_COLS):
                if board[0, r, c].item() != 0:
                    new_type, nr, nc = DotsAndBoxesGame._transform_edge("H", r, c, k, flip)
                    channel = 0 if new_type == "H" else 1
                    transformed[channel, nr, nc] = board[0, r, c]

        for r in range(DotsAndBoxesGame.BOX_ROWS):
            for c in range(DotsAndBoxesGame.DOT_COLS):
                if board[1, r, c].item() != 0:
                    new_type, nr, nc = DotsAndBoxesGame._transform_edge("V", r, c, k, flip)
                    channel = 0 if new_type == "H" else 1
                    transformed[channel, nr, nc] = board[1, r, c]

        for r in range(DotsAndBoxesGame.BOX_ROWS):
            for c in range(DotsAndBoxesGame.BOX_COLS):
                if board[2, r, c].item() != 0:
                    nr, nc = DotsAndBoxesGame._transform_box(r, c, k, flip)
                    transformed[2, nr, nc] = board[2, r, c]

        transformed[
            2, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS
        ] = player_marker
        return transformed

    @staticmethod
    def _transform_edge(
        edge_type: str, r: int, c: int, k: int, flip: bool
    ) -> tuple[str, int, int]:
        if edge_type == "H":
            p1 = (r, c)
            p2 = (r, c + 1)
        else:
            p1 = (r, c)
            p2 = (r + 1, c)

        p1, rot_rows, rot_cols = DotsAndBoxesGame._rotate_point(
            p1, k, DotsAndBoxesGame.DOT_ROWS, DotsAndBoxesGame.DOT_COLS
        )
        p2, _, _ = DotsAndBoxesGame._rotate_point(
            p2, k, DotsAndBoxesGame.DOT_ROWS, DotsAndBoxesGame.DOT_COLS
        )
        if flip:
            p1 = DotsAndBoxesGame._flip_point(
                p1, rot_rows, rot_cols
            )
            p2 = DotsAndBoxesGame._flip_point(
                p2, rot_rows, rot_cols
            )

        if p1[0] == p2[0]:
            nr = p1[0]
            nc = min(p1[1], p2[1])
            return "H", nr, nc
        nr = min(p1[0], p2[0])
        nc = p1[1]
        return "V", nr, nc

    @staticmethod
    def _transform_box(r: int, c: int, k: int, flip: bool) -> tuple[int, int]:
        (r, c), rot_rows, rot_cols = DotsAndBoxesGame._rotate_point(
            (r, c), k, DotsAndBoxesGame.BOX_ROWS, DotsAndBoxesGame.BOX_COLS
        )
        if flip:
            r, c = DotsAndBoxesGame._flip_point(
                (r, c), rot_rows, rot_cols
            )
        return r, c

    @staticmethod
    def _rotate_point(
        point: tuple[int, int], k: int, rows: int, cols: int
    ) -> tuple[tuple[int, int], int, int]:
        r, c = point
        for _ in range(k % 4):
            r, c = cols - 1 - c, r
            rows, cols = cols, rows
        return (r, c), rows, cols

    @staticmethod
    def _flip_point(point: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
        r, c = point
        if rows != cols:
            raise ValueError("DotsAndBoxes 翻转仅支持方形棋盘")
        return c, r
