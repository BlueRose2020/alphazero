from __future__ import annotations

import random
from pathlib import Path
from typing import Type, Optional, Any

import os
import torch
import torch.multiprocessing as mp

from config import *
from games.base import BaseGame
from nn_models.base import BaseModel
from training.nn_trainer import Trainer, get_train_device
from training.self_play import ChessArena
from utils.experience_pool import ExperiencePool
from utils.share_ring_buffer import SharedRingBufferExperiencePool
from utils.logger import setup_logger, colorize

logger = setup_logger(__name__)


class TrainerUtils:
    _game_cls: Type[BaseGame] = BaseGame  # 用于设置使用快速模型时的保存目录

    @staticmethod
    def set_game_cls(game_cls: Type[BaseGame]) -> None:
        """设置游戏类，用于在使用QuickModel时确定保存目录"""
        TrainerUtils._game_cls = game_cls

    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_save_dirs(model_cls: Type[BaseModel]) -> tuple[str, str, str]:
        if model_cls.__name__ == "QuickModel":
            dir_name = (
                "Quick"
                + TrainerUtils._game_cls.__name__.replace("Game", "")
                + ("_history" if USE_HISTORY else "")
            )
        else:
            dir_name = model_cls.__name__.replace("Model", "") + (
                "_history" if USE_HISTORY else ""
            )
        model_dir = str(Path("result") / "models" / dir_name)
        optim_dir = str(Path("result") / "optimizers" / dir_name)
        exp_dir = str(Path("result") / "experiences" / dir_name)
        return model_dir, optim_dir, exp_dir

    @staticmethod
    def sync_model_from_shared(
        model: BaseModel, model_state: dict[str, torch.Tensor], model_lock: Any
    ) -> None:
        with model_lock:
            model.load_state_dict({k: v.clone() for k, v in model_state.items()})

    @staticmethod
    def sync_shared_from_model(
        model: BaseModel, model_state: dict[str, torch.Tensor], model_lock: Any
    ) -> None:
        with model_lock:
            new_state = model.state_dict()
            for k, v in model_state.items():
                v.copy_(new_state[k].detach().cpu())

    @staticmethod
    def load_optimizer_state(optim: torch.optim.Optimizer, optim_save_dir: str) -> None:
        optim_path = os.path.join(optim_save_dir, "last_optim.pth")
        if os.path.exists(optim_path):
            try:
                optim.load_state_dict(torch.load(optim_path))
                logger.info(f"已加载之前的优化器状态: {optim_path}")
            except Exception as e:
                logger.warning(f"加载优化器状态失败: {e}")
        else:
            logger.info("未找到之前的优化器状态，将使用新初始化的优化器")

    @staticmethod
    def save_model(
        model: BaseModel, model_save_dir: str, epoch: Optional[int] = None
    ) -> None:
        TrainerUtils.ensure_dir(model_save_dir)
        model_path = (
            os.path.join(model_save_dir, f"current_model.pth")
            if epoch is not None
            else os.path.join(model_save_dir, "last_model.pth")
        )
        torch.save(model.state_dict(), model_path)
        logger.info(colorize(f"已保存模型状态: {model_path}", SAVE_MODEL_COLOR))

    @staticmethod
    def save_optimizer(
        optim: torch.optim.Optimizer, optim_save_dir: str, epoch: Optional[int] = None
    ) -> None:
        TrainerUtils.ensure_dir(optim_save_dir)
        optim_path = (
            os.path.join(optim_save_dir, f"current_optim.pth")
            if epoch is not None
            else os.path.join(optim_save_dir, "last_optim.pth")
        )
        torch.save(optim.state_dict(), optim_path)
        logger.info(colorize(f"已保存优化器状态: {optim_path}", SAVE_MODEL_COLOR))

    @staticmethod
    def save_experience_pool(
        experience_pool: ExperiencePoolType, exp_save_dir: str
    ) -> None:
        TrainerUtils.ensure_dir(exp_save_dir)
        exp_path = os.path.join(
            exp_save_dir,
            (
                f"experience_pool_{len(experience_pool)}.pth"
                if isinstance(experience_pool, SharedRingBufferExperiencePool)
                else f"experience_pool_{len(experience_pool)}.pkl"
            ),
        )
        experience_pool.save(exp_path)
        logger.info(colorize(f"已保存经验池状态: {exp_path}", SAVE_EXP_COLOR))

    @staticmethod
    def load_model_state(model: BaseModel, model_save_dir: str) -> None:
        model_path = os.path.join(model_save_dir, "last_model.pth")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, weights_only=True))
                logger.info(f"已加载之前的模型状态: {model_path}")
            except Exception as e:
                logger.warning(f"加载模型状态失败: {e}")
        else:
            logger.info("未找到之前的模型状态，将使用随机初始化的模型")

    @staticmethod
    def load_experience_pool_state(
        experience_pool: ExperiencePoolType, exp_save_dir: str
    ) -> None:
        experience_path = TrainerUtils.get_experience_path(
            experience_pool, exp_save_dir
        )

        if os.path.exists(experience_path):
            try:
                experience_pool.load(experience_path)
                logger.info(f"已加载之前的经验池状态: {experience_path}")
            except Exception as e:
                logger.warning(f"加载经验池状态失败: {e}")
        else:
            logger.info("未找到之前的经验池状态，将使用空的经验池")

    @staticmethod
    def get_experience_path(
        experience_pool: ExperiencePoolType, exp_save_dir: str
    ) -> str:
        experience_file = (
            f"experience_pool_{len(experience_pool)}.pkl"
            if isinstance(experience_pool, ExperiencePool)
            else f"experience_pool_{len(experience_pool)}.pth"
        )
        return os.path.join(exp_save_dir, experience_file)


# 训练和自对弈的工作函数，分别在不同的进程中运行
def _self_player_worker(
    model_cls: Type[BaseModel],
    game_cls: Type[BaseGame],
    model_state: dict[str, torch.Tensor],
    experience_pool: ExperiencePoolType,
    num_self_play_games: int,
    worker_id: int,
    model_lock: Any,
) -> None:
    try:
        # 初始化进程
        seed = SEED_BIAS + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if SELF_PLAY_DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.set_num_threads(1)
        arena = ChessArena(model_cls, game_cls, device=torch.device(SELF_PLAY_DEVICE))
        TrainerUtils.sync_model_from_shared(arena.model, model_state, model_lock)

        # 开始自对弈
        tempertature = START_TEMPERATURE
        for game_idx in range(num_self_play_games):
            logger.info(
                colorize(f"自对弈进程 {worker_id}:", PROCESSING_COLOR)
                + f"开始第 {game_idx+1}/{num_self_play_games} 场自对弈，当前温度: {tempertature:.4f}"
            )

            # 每隔一定轮数更新一次模型状态，以便自对弈过程中使用最新的模型进行决策
            if (game_idx + 1) % SELF_PLAY_UPDATE_MODEL_FREQUENCY == 0:
                logger.info(
                    colorize(f"自对弈进程 {worker_id} :", PROCESSING_COLOR)
                    + colorize(
                        "正在更新共享模型状态...",
                        UPDATE_MODEL_COLOR,
                    )
                )
                TrainerUtils.sync_model_from_shared(
                    arena.model, model_state, model_lock
                )
                logger.info(
                    colorize(f"自对弈进程 {worker_id} :", PROCESSING_COLOR)
                    + colorize(
                        "模型状态更新完成，继续自对弈...",
                        UPDATE_MODEL_COLOR,
                    )
                )

            arena.self_play(tao=tempertature, experience_pool=experience_pool)
            tempertature = max(tempertature * TEMPERATURE_DECAY, END_TEMPERATURE)

        logger.info(
            colorize(f"自对弈进程 {worker_id}:", PROCESSING_COLOR)
            + colorize("已完成所有自对弈，退出进程", WORKER_FINISH_COLOR)
        )
    except Exception as e:
        logger.error(f"自对弈进程 {worker_id} 遇到错误: {e}", exc_info=True)


def _training_worker(
    model_cls: Type[BaseModel],
    game_cls: Type[BaseGame],
    experience_pool: ExperiencePoolType,
    self_play_done: Any,  # 自对弈完成标志
    model_state: dict[str, torch.Tensor],
    model_lock: Any,
) -> None:
    try:
        import time

        torch.set_num_threads(2)
        TrainerUtils.set_game_cls(game_cls)
        train_device = get_train_device()
        model = model_cls().to(train_device)
        TrainerUtils.sync_model_from_shared(model, model_state, model_lock)

        optim = create_optimizer(model.parameters(), lr=LEARNING_RATE)
        batch_size = BATCH_SIZE

        last_exp_save_time = time.time()

        if BATCH_SIZE > (l := len(experience_pool)):
            logger.warning(
                f"[训练进程] BATCH_SIZE ({BATCH_SIZE}) 大于经验池容量 ({l}),BATCH_SIZE 将被调整为 {l}"
            )
            batch_size = l

        n = 0  # 控制log输出频率
        model_save_dir, optim_save_dir, exp_save_dir = TrainerUtils.get_save_dirs(
            model_cls
        )

        TrainerUtils.load_optimizer_state(optim, optim_save_dir)

        trainer = Trainer(model, experience_pool, optim, device=train_device)

        # 定义一个函数来更新模型状态到共享内存，减小代码冗余
        def update_model() -> None:
            TrainerUtils.sync_shared_from_model(trainer.model, model_state, model_lock)

        while not self_play_done.value:
            # 检查经验池中是否有足够的数据进行训练，如果没有则等待一段时间后继续检查，避免频繁尝试训练导致的性能问题
            exp_size = experience_pool.size()
            if exp_size < max(batch_size, MIN_EXP_SIZE_FOR_TRAINING):
                time.sleep(10)
                logger.info(
                    colorize("训练进程：", PROCESSING_COLOR)
                    + f"经验池中数据不足以进行训练，当前大小: {exp_size}，等待中..."
                )
                continue

            # 训练模型
            try:
                trainer.train()
                n += 1
            except Exception as e:
                logger.error(f"[训练进程] 训练过程中遇到错误: {e}", exc_info=True)
                continue

            # 判断是否需要更新模型状态到共享内存，以及是否需要保存模型和经验池状态，控制日志输出频率，避免过于频繁的操作导致性能问题
            if n % TRAIN_UPDATE_MODEL_FREQUENCY == 0:
                update_model()
            if n % TRAIN_LOG_FREQUENCY == 0:
                logger.info(
                    colorize("训练进程: ", PROCESSING_COLOR)
                    + colorize(
                        f" 已完成 {n} 轮训练，当前经验池大小: {experience_pool.size()}",
                        TRAIN_EPOCH_COLOR,
                    )
                )

            if n % MODEL_SAVE_FREQUENCY == 0:
                TrainerUtils.save_model(
                    trainer.model, model_save_dir, n // MODEL_SAVE_FREQUENCY
                )
                TrainerUtils.save_optimizer(
                    trainer.optim, optim_save_dir, n // MODEL_SAVE_FREQUENCY
                )

            if time.time() - last_exp_save_time >= EXP_SAVE_FREQUENCY:
                TrainerUtils.save_experience_pool(experience_pool, exp_save_dir)
                last_exp_save_time = time.time()

        # 等待所有自对弈完成后，进行最后的训练和保存，确保最后的自对弈经验得到使用
        for _ in range(TRAIN_EPOCHS_AFTER_SELF_PLAY_DONE):
            try:
                trainer.train()
            except Exception as e:
                logger.error(f"[训练进程] 训练过程中遇到错误: {e}", exc_info=True)

        # 保存最终的结果并输出日志
        logger.info(
            colorize("训练进程: ", PROCESSING_COLOR)
            + colorize("正在保存最终模型和经验池状态...", FINISH_COLOR)
        )
        update_model()
        TrainerUtils.save_optimizer(trainer.optim, optim_save_dir)
        TrainerUtils.save_model(trainer.model, model_save_dir)
        TrainerUtils.save_experience_pool(experience_pool, exp_save_dir)

        logger.info(
            colorize("训练进程: ", PROCESSING_COLOR)
            + colorize("保存完毕，训练进程退出", FINISH_COLOR)
        )

    except Exception as e:
        logger.error(f"[训练进程] 遇到错误: {e}", exc_info=True)


class AlphaZeroTrainer:
    def __init__(
        self,
        model_cls: Type[BaseModel],
        game_cls: Type[BaseGame],
    ) -> None:
        # 初始化模型、游戏和经验池
        self.device = get_train_device()
        self.model = model_cls().to(device=self.device)
        self.game = game_cls()
        self.optim = create_optimizer(self.model.parameters(), lr=LEARNING_RATE)
        self.experience_pool = AlphaZeroTrainer._create_experience_pool(game_cls)

        TrainerUtils.set_game_cls(game_cls)  # 设置游戏类以确定保存目录

        (
            self._model_save_dir,
            self._optim_save_dir,
            self._exp_save_dir,
        ) = TrainerUtils.get_save_dirs(model_cls)

        if model_cls.__name__ == "QuickModel":
            self._load_quick_model_previous_state()
        else:
            self._load_previous_state()  # 会自动判断是否存在之前的模型,优化器和经验池状态并加载

        # 初始化训练器和自对弈环境
        self.trainer = Trainer(
            model=self.model,
            experience_pool=self.experience_pool,
            optim=self.optim,
            device=self.device,
        )

        self.arena = ChessArena(
            model_cls, game_cls, device=torch.device(SELF_PLAY_DEVICE)
        )

        # 输出日志
        logger.info("AlphaZero 训练器初始化完成")
        logger.info(f"模型: {model_cls.__name__}")
        logger.info(f"游戏: {game_cls.__name__}")
        logger.info(f"多进程自对弈: {'启用' if USE_MULTIPROCESSING else '禁用'}")

    # 训练函数
    def train(self) -> None:
        # 判断经验池类型与多进程配置是否匹配，并选择相应的训练流程
        if USE_MULTIPROCESSING and isinstance(
            self.experience_pool, SharedRingBufferExperiencePool
        ):
            self._multi_process_parallel()
        elif not USE_MULTIPROCESSING and isinstance(
            self.experience_pool, ExperiencePool
        ):
            self._single_process_self_play()
        else:
            logger.warning("经验池类型与多进程配置不匹配，改为单进程自对弈")
            for _ in range(NUM_SELF_PLAY_GAMES // TRAIN_FREQUENCY):
                self._single_process_self_play()
                self._train_phase()

    def _multi_process_parallel(self) -> None:
        # 使用spawn方式创建子进程，确保每个子进程都有独立的内存空间，避免数据竞争和死锁问题
        ctx = mp.get_context("spawn")
        workers = min(SELF_PLAY_WORKER_NUM, NUM_SELF_PLAY_GAMES)
        if workers <= 0:  # 还有一个收集经验的进程
            logger.warning("自对弈进程数量过少，改为单进程自对弈")
            self._single_process_self_play()
            self._train_phase()
            return

        model_state = {
            k: v.detach().cpu().clone().share_memory_()
            for k, v in self.model.state_dict().items()
        }
        model_lock = ctx.Lock()
        self_play_done = ctx.Value("b", False)

        # 将总的自对弈场数平均分配给每个自对弈进程，确保所有自对弈场数都被分配完
        games_per_worker: list[int] = [NUM_SELF_PLAY_GAMES // workers] * workers
        for i in range(NUM_SELF_PLAY_GAMES % workers):
            games_per_worker[i] += 1

        # 启动自对弈进程
        processes = []
        for worker_id, games in enumerate(games_per_worker, start=1):
            p = ctx.Process(
                target=_self_player_worker,
                args=(
                    type(self.model),
                    type(self.game),
                    model_state,
                    self.experience_pool,
                    games,
                    worker_id,
                    model_lock,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"已启动自对弈进程 {worker_id}，负责 {games} 场自对弈")

        # 启动训练进程
        trainer_process = ctx.Process(
            target=_training_worker,
            args=(
                type(self.model),
                type(self.game),
                self.experience_pool,
                self_play_done,
                model_state,
                model_lock,
            ),
        )
        trainer_process.start()

        # 等待所有进程完成
        for p in processes:
            p.join()
        logger.info(colorize("所有自对弈进程已完成", FINISH_COLOR))

        self_play_done.value = True
        trainer_process.join()

    def _single_process_self_play(self) -> None:
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        self.arena.model.load_state_dict(cpu_state)
        temp = START_TEMPERATURE
        for game_num in range(TRAIN_FREQUENCY):
            try:
                self.arena.self_play(tao=temp, experience_pool=self.experience_pool)
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
                batch_size = min(BATCH_SIZE, self.experience_pool.size())
                if batch_size == 0:
                    logger.warning("经验池为空，跳过训练阶段")
                    return
                self.trainer.train(batch_size=batch_size)
            except Exception as e:
                logger.error(f"[训练阶段] 第 {epoch} 轮训练失败: {e}", exc_info=True)

    def _load_quick_model_previous_state(self) -> None:
        """专门为QuickModel加载之前的状态，以兼容不同配置下的模型结构变化"""
        try:
            self._load_previous_state()
        except Exception as e:
            logger.warning(
                f"尝试加载之前的模型状态失败，可能是由于模型结构变化导致的: {e}"
            )
            input_str = ""
            while input_str.lower() not in {"y", "n"}:
                input_str = input("是否覆盖？y/n: ")
            if input_str.lower() == "y":
                logger.info("将覆盖之前的模型状态，使用随机初始化的模型进行训练")

                self.model = type(self.model)().to(device=self.device)  # 重新初始化模型
            else:
                logger.info("退出训练")
                exit(0)

    def _load_previous_state(self) -> None:
        """加载之前的模型，优化器和经验池状态"""
        TrainerUtils.load_model_state(self.model, self._model_save_dir)
        TrainerUtils.load_optimizer_state(self.optim, self._optim_save_dir)
        TrainerUtils.load_experience_pool_state(
            self.experience_pool, self._exp_save_dir
        )

    # 静态辅助函数
    @staticmethod
    def _create_experience_pool(game_cls: Type[BaseGame]) -> ExperiencePoolType:
        if USE_MULTIPROCESSING:
            nn_state_shape = AlphaZeroTrainer._infer_nn_state_shape(game_cls)
            action_mask = game_cls.legal_action_mask(game_cls.initial_state())
            num_action = int(action_mask.numel())
            return SharedRingBufferExperiencePool(
                state_shape=nn_state_shape,
                num_action=(1, num_action),
                _game_cls=game_cls,
                _capacity=DEFAULT_CAPACITY,
            )
        return ExperiencePool(_game_cls=game_cls, capacity=DEFAULT_CAPACITY)

    @staticmethod
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
