from __future__ import annotations

from pathlib import Path
from typing import Type, Optional, Any

import os
import torch
import torch.multiprocessing as mp

from config import *
from games.base import BaseGame
from nn_models.base import BaseModel
from training.nn_trainer import Trainer
from training.self_play import ChessArena
from utils.experience_pool import ExperiencePool
from utils.share_ring_buffer import SharedRingBuffer
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _infer_nn_state_shape(game_cls: Type[BaseGame]) -> ShapeType:
    """nn_state,包含批次通道
    注意玩家视角还有一个通道
    """
    state_shape = tuple(game_cls.initial_state().shape)

    if USE_HISTORY:
        if len(state_shape) >= 3:
            return (1, state_shape[0] * HISTORY_LEN + 1, *state_shape[1:])
        return (1, HISTORY_LEN + 1, *state_shape)

    if len(state_shape) >= 3:
        return (1, state_shape[0] + 1, *state_shape[1:])
    return (1, 2, *state_shape)


def create_experience_pool(game_cls: Type[BaseGame]) -> ExperiencePoolType:
    if USE_MULTIPROCESSING:
        action_mask = game_cls.legal_action_mask(game_cls.initial_state())
        num_action = int(action_mask.numel())
        nn_state_shape = _infer_nn_state_shape(game_cls)
        return SharedRingBuffer(
            state_shape=nn_state_shape,
            num_action=(1, num_action),
            _capacity=DEFAULT_CAPACITY,
        )
    return ExperiencePool(capacity=DEFAULT_CAPACITY)


def _self_player_worker(
    model_cls: Type[BaseModel],
    game_cls: Type[BaseGame],
    model_state: dict[str, torch.Tensor],
    experience_pool: ExperiencePoolType,
    num_games: int,
    worker_id: int,
    model_lock: Any,
) -> None:
    try:
        torch.set_num_threads(1)
        arena = ChessArena(model_cls, game_cls)
        arena.model.load_state_dict(model_state)
        tempertature = START_TEMPERATURE
        for game_idx in range(num_games):
            if (game_idx + 1) % UPDATE_MODEL_FREQUENCY == 0:
                logger.info(
                    f"Worker {worker_id}: 已完成 {game_idx+1} 场自对弈，正在更新共享模型状态..."
                )
                with model_lock:
                    arena.model.load_state_dict(model_state)
                logger.info(f"Worker {worker_id}: 模型状态更新完成，继续自对弈...")

            arena.self_play(tao=tempertature, experience_pool=experience_pool)
            tempertature = max(tempertature * TEMPERATURE_DECAY, END_TEMPERATURE)

            if (game_idx + 1) % 50 == 0:
                logger.info(
                    f"Worker {worker_id}: 已完成 {game_idx+1} 场自对弈，当前温度: {tempertature:.4f}"
                )
    except Exception as e:
        logger.error(f"Worker {worker_id} 遇到错误: {e}", exc_info=True)


def _training_worker(
    model_cls: Type[BaseModel],
    experience_pool: ExperiencePoolType,
    self_play_done: Any,  # 自对弈完成标志
) -> None:
    try:
        import time

        torch.set_num_threads(2)
        model = model_cls()
        trainer = Trainer(model, experience_pool)
        batch_size = BATCH_SIZE
        if BATCH_SIZE > (l := len(experience_pool)):
            logger.warning(
                f"[训练进程] BATCH_SIZE ({BATCH_SIZE}) 大于经验池容量 ({l}),BATCH_SIZE 将被调整为 {l}"
            )
            batch_size = l

        n = 0  # 控制log输出频率
        while not self_play_done.value:
            exp_size = experience_pool.size()
            if exp_size < max(batch_size, MIN_EXP_SIZE_FOR_TRAINING):
                time.sleep(10)
                logger.info(
                    f"[训练进程] 经验池中数据不足以进行训练，当前大小: {exp_size}，等待中..."
                )
                continue

            # 训练模型
            try:
                trainer.train()
                n += 1
            except Exception as e:
                logger.error(f"[训练进程] 训练过程中遇到错误: {e}", exc_info=True)
                continue

            if n % TRAIN_LOG_FREQUENCY == 0:
                logger.info(
                    f"[训练进程] 已完成 {n} 轮训练，当前经验池大小: {experience_pool.size()}"
                )

            if n % MODEL_SAVE_FREQUENCY == 0:
                model_save_dir = str(Path("result") / "models" / model_cls.__name__)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model_path = os.path.join(model_save_dir, f"model_{n}.pth")
                trainer.save_model(model_path)
                logger.info(f"[训练进程] 已保存模型状态: {model_path}")

        logger.info("[训练进程] 自对弈完成，训练进程退出")

    except Exception as e:
        logger.error(f"[训练进程] 遇到错误: {e}", exc_info=True)


class AlphaZeroTrainer:
    def __init__(
        self,
        model_cls: Type[BaseModel],
        game_cls: Type[BaseGame],
        exp: Optional[ExperiencePoolType] = None,
    ) -> None:
        self.model = model_cls()
        self.game = game_cls()
        self.exp = exp if exp is not None else create_experience_pool(game_cls)

        self.trainer = Trainer(self.model, self.exp)
        self.arena = ChessArena(model_cls, game_cls)
        self._model_save_dir = str(Path("result") / "models" / model_cls.__name__)

        if os.path.exists(self._model_save_dir):
            self._load_previous_state()

        logger.info("AlphaZero 训练器初始化完成")
        logger.info(f"模型: {model_cls.__name__}")
        logger.info(f"游戏: {game_cls.__name__}")
        logger.info(f"多进程自对弈: {'启用' if USE_MULTIPROCESSING else '禁用'}")

    def train(
        self,
    ) -> None:
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)

        if USE_MULTIPROCESSING and isinstance(self.exp, SharedRingBuffer):
            self._multi_process_parallel()
        elif not USE_MULTIPROCESSING and isinstance(self.exp, ExperiencePool):
            self._single_process_self_play()
        else:
            logger.warning("经验池类型与多进程配置不匹配，改为单进程自对弈")
            for _ in range(NUM_SELF_PLAY_GAMES // TRAIN_FREQUENCY):
                self._single_process_self_play()
                self._train_phase()

        logger.info("保存最后的模型状态...")
        if not os.path.exists(self._model_save_dir):
            os.makedirs(self._model_save_dir)

        trainer = Trainer(self.model, self.exp)
        trainer.save_model(os.path.join(self._model_save_dir, "last_model.pth"))
        logger.info("模型保存完成")

        logger.info("保存经验池")
        self.exp.save(
            os.path.join(
                self._model_save_dir,
                (
                    "experience_pool.pic"
                    if isinstance(self.exp, ExperiencePool)
                    else "experience_pool.pth"
                ),
            )
        )
        logger.info("经验池保存完成")

    def _multi_process_parallel(self) -> None:
        ctx = mp.get_context("spawn")
        workers = min(SELF_PLAY_WORKER_NUM, NUM_SELF_PLAY_GAMES)
        if workers <= 0:  # 还有一个收集经验的进程
            logger.warning("自对弈进程数量过少，改为单进程自对弈")
            self._single_process_self_play()
            self._train_phase()
            return

        model_state = {
            k: v.clone().share_memory_() for k, v in self.model.state_dict().items()
        }
        model_lock = ctx.Lock()
        self_play_done = ctx.Value("b", False)

        games_per_worker = [NUM_SELF_PLAY_GAMES // workers] * workers
        for i in range(NUM_SELF_PLAY_GAMES % workers):
            games_per_worker[i] += 1

        processes = []
        for worker_id, games in enumerate(games_per_worker, start=1):
            p = ctx.Process(
                target=_self_player_worker,
                args=(
                    type(self.model),
                    type(self.game),
                    model_state,
                    self.exp,
                    games,
                    worker_id,
                    model_lock,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"已启动自对弈进程 {worker_id}，负责 {games} 场自对弈")

        trainer_process = ctx.Process(
            target=_training_worker,
            args=(
                type(self.model),
                self.exp,
                self_play_done,
            ),
        )
        trainer_process.start()

        for p in processes:
            p.join()
        logger.info("所有自对弈进程已完成")

        self_play_done.value = True
        trainer_process.join()
        logger.info("训练进程已完成")

    def _single_process_self_play(self) -> None:
        self.arena.model.load_state_dict(self.model.state_dict())
        temp = START_TEMPERATURE
        for game_num in range(TRAIN_FREQUENCY):
            try:
                self.arena.self_play(tao=temp, experience_pool=self.exp)
                temp = max(temp * TEMPERATURE_DECAY, END_TEMPERATURE)
                if (game_num + 1) % 50 == 0:
                    logger.info(f"已完成 {game_num+1} 场自对弈，当前温度: {temp:.4f}")
            except Exception as e:
                logger.error(
                    f"[单进程自对弈] 第 {game_num} 场自对弈失败: {e}", exc_info=True
                )

    def _train_phase(self) -> None:
        """单进程的训练阶段"""
        for epoch in range(TRAIN_EPOCHS):
            try:
                batch_size = min(BATCH_SIZE, self.exp.size())
                if batch_size == 0:
                    logger.warning("经验池为空，跳过训练阶段")
                    return
                self.trainer.train(batch_size=batch_size)
            except Exception as e:
                logger.error(f"[训练阶段] 第 {epoch} 轮训练失败: {e}", exc_info=True)

    def _load_previous_state(self) -> None:
        """加载之前的模型和经验池状态"""
        if os.path.exists(os.path.join(self._model_save_dir, "last_model.pth")):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self._model_save_dir, "last_model.pth"),
                    weights_only=True,
                )
            )
            logger.info(
                f"已加载之前的模型状态，继续训练: {self._model_save_dir}/last_model.pth"
            )

        experience_file = (
            "experience_pool.pic"
            if isinstance(self.exp, ExperiencePool)
            else "experience_pool.pth"
        )

        if os.path.exists(os.path.join(self._model_save_dir, experience_file)):
            self.exp.load(
                os.path.join(
                    self._model_save_dir,
                    experience_file,
                )
            )
            logger.info(
                f"已加载之前的经验池状态，继续训练: {self._model_save_dir}/{experience_file}"
            )