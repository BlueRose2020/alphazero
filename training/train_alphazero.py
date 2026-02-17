"""AlphaZero 训练主程序（支持单进程/多进程自对弈）"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type, Optional, Any

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
    """推断用于经验缓冲区的 NNState 形状（含批次维度）"""
    state_shape = tuple(game_cls.initial_state().shape)

    if USE_HISTORY:
        if len(state_shape) >= 3:
            return (state_shape[0] * HISTORY_LEN + 1, *state_shape[1:])
        return (1, HISTORY_LEN + 1, *state_shape)

    if len(state_shape) >= 3:
        return (state_shape[0] + 1, *state_shape[1:])
    return (1, 2, *state_shape)


def create_experience_pool(game_cls: Type[BaseGame]) -> ExperiencePoolType:
    """根据配置创建经验池"""
    if USE_MULTIPROCESSING:
        action_mask = game_cls.legal_action_mask(game_cls.initial_state())
        action_dim = int(action_mask.numel())
        nn_state_shape = _infer_nn_state_shape(game_cls)
        return SharedRingBuffer(
            state_shape=nn_state_shape,
            num_action=(1, action_dim),
            capacity=DEFAULT_CAPACITY,
        )
    return ExperiencePool(capacity=DEFAULT_CAPACITY)


def _self_play_worker(
    model_cls: Type[BaseModel],
    game_cls: Type[BaseGame],
    model_state: dict[str, torch.Tensor],
    experience_pool: SharedRingBuffer,
    temperature: float,
    num_games: int,
    worker_id: int,
    model_lock: Any,
    reload_interval: int = 10,
) -> None:
    """多进程自对弈工作进程"""
    try:
        torch.set_num_threads(1)
        arena = ChessArena(model_cls, game_cls)
        arena.model.load_state_dict(model_state)

        for game_idx in range(1, num_games + 1):
            # 定期从共享内存加载最新模型
            if game_idx > 1 and (game_idx - 1) % reload_interval == 0:
                with model_lock:
                    arena.model.load_state_dict(model_state)
                logger.info(f"[Worker {worker_id}] 已加载最新模型 (第 {game_idx} 局)")
            
            arena.self_play(tao=temperature, experience_pool=experience_pool)
            if game_idx % 50 == 0:
                logger.info(f"[Worker {worker_id}] 已完成 {game_idx}/{num_games} 局")
        
        logger.info(f"[Worker {worker_id}] 所有对局完成，共 {num_games} 局")
    except Exception as e:
        logger.error(f"[Worker {worker_id}] 自对弈进程异常: {e}", exc_info=True)


def _training_worker(
    model_cls: Type[BaseModel],
    experience_pool: SharedRingBuffer,
    train_steps: int,
    shared_model_state: dict[str, torch.Tensor],
    model_lock: Any,
    self_play_done: Any,
    capacity: int,
) -> None:
    """训练进程：持续从经验池采样并训练模型"""
    try:
        import time
        torch.set_num_threads(2)
        model = model_cls()
        trainer = Trainer(model, experience_pool)
        
        # 检查BATCH_SIZE是否合理
        if BATCH_SIZE > capacity:
            logger.error(f"[训练进程] BATCH_SIZE ({BATCH_SIZE}) 大于经验池容量 ({capacity})，无法训练！")
            return
        
        completed_steps = 0
        wait_count = 0
        max_wait_count = 60  # 最多等待60秒
        
        while completed_steps < train_steps:
            # 检查经验池是否有足够数据
            current_size = experience_pool.size.value  # type: ignore
            if current_size < BATCH_SIZE:
                # 检查自对弈是否已完成
                if self_play_done.value:  # type: ignore
                    logger.warning(
                        f"[训练进程] 自对弈已结束，但经验池数据不足 ({current_size}/{BATCH_SIZE})，"
                        f"已完成 {completed_steps}/{train_steps} 步训练后退出"
                    )
                    break
                
                wait_count += 1
                if wait_count <= max_wait_count:
                    logger.info(f"[训练进程] 经验不足 ({current_size}/{BATCH_SIZE})，等待1秒...")
                    time.sleep(1)
                    continue
                else:
                    logger.warning(f"[训练进程] 等待超时，经验池数据仍不足，退出训练")
                    break
            
            wait_count = 0  # 重置等待计数
            
            try:
                trainer.train()
                completed_steps += 1
                
                if completed_steps % 10 == 0:
                    logger.info(f"[训练进程] 已完成 {completed_steps}/{train_steps} 步训练")
                
                # 定期更新共享模型状态（调整为每20步）
                if completed_steps % 20 == 0:
                    with model_lock:
                        for k, v in model.state_dict().items():
                            shared_model_state[k].copy_(v)
                    logger.info(f"[训练进程] 已同步模型状态到共享内存")
                    
            except Exception as e:
                logger.warning(f"[训练进程] 训练步骤异常: {e}")
                
        # 训练结束，最后一次同步模型
        with model_lock:
            for k, v in model.state_dict().items():
                shared_model_state[k].copy_(v)
        logger.info(f"[训练进程] 训练完成，共 {completed_steps} 步")
        
    except Exception as e:
        logger.error(f"[训练进程] 异常退出: {e}", exc_info=True)


class AlphaZeroTrainer:
    """AlphaZero 训练管理器"""

    def __init__(
        self,
        model_cls: Type[BaseModel],
        game_cls: Type[BaseGame],
        experience_pool: Optional[ExperiencePoolType] = None,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 10,
        self_play_workers: Optional[int] = None,
    ) -> None:
        self.model = model_cls()
        self.game_cls = game_cls
        self.experience_pool = (
            experience_pool
            if experience_pool is not None
            else create_experience_pool(game_cls)
        )
        if checkpoint_dir is None:
            checkpoint_dir = str(Path("result") / "models" / game_cls.__name__)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        if self_play_workers is None:
            self.self_play_workers = max(1, mp.cpu_count() - 1)
        else:
            self.self_play_workers = max(1, self_play_workers)

        self.trainer = Trainer(self.model, self.experience_pool)
        self.arena = ChessArena(model_cls, game_cls)

        logger.info("AlphaZero 训练器初始化完成")
        logger.info(f"模型: {model_cls.__name__}")
        logger.info(f"游戏: {game_cls.__name__}")
        logger.info(f"检查点目录: {self.checkpoint_dir}")
        logger.info(f"多进程自对弈: {'启用' if USE_MULTIPROCESSING else '禁用'}")

    def train(
        self,
        num_iterations: int = 100,
        self_play_games: int = SELF_PLAY_NUM,
        train_steps: int = 100,
        temperature: float = 1.0,
        save_interval: int = 10,
    ) -> None:
        logger.info("=" * 60)
        logger.info(f"开始训练 - 总迭代: {num_iterations}")
        logger.info(f"每次迭代: {self_play_games} 局自对弈, {train_steps} 步训练")
        logger.info("=" * 60)

        for iteration in range(1, num_iterations + 1):
            logger.info(f"\n[迭代 {iteration}/{num_iterations}]")

            # 并行自对弈+训练
            logger.info(f"启动 {self_play_games} 局自对弈 + {train_steps} 步训练（并行执行）...")
            self._parallel_self_play_and_train(self_play_games, train_steps, temperature)

            if iteration % save_interval == 0:
                checkpoint_path = self.checkpoint_dir / f"model_{iteration}.pth"
                self.trainer.save_model(str(checkpoint_path))

            if iteration % self.log_interval == 0:
                self._log_statistics(iteration)

        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info("=" * 60)

        final_model_path = self.checkpoint_dir / f"model_{num_iterations}.pth"
        self.trainer.save_model(str(final_model_path))

    def _parallel_self_play_and_train(
        self, num_games: int, train_steps: int, temperature: float
    ) -> None:
        """并行执行自对弈和训练"""
        if USE_MULTIPROCESSING and isinstance(self.experience_pool, SharedRingBuffer):
            self._multi_process_parallel(num_games, train_steps, temperature)
        else:
            # 单进程模式：先自对弈再训练
            self._single_process_self_play(num_games, temperature)
            self._train_phase(train_steps)

    def _multi_process_parallel(
        self, num_games: int, train_steps: int, temperature: float
    ) -> None:
        """多进程并行模式：多个自对弈进程 + 1个训练进程"""
        if not isinstance(self.experience_pool, SharedRingBuffer):
            logger.warning("经验池不是 SharedRingBuffer，切换为单进程模式")
            self._single_process_self_play(num_games, temperature)
            self._train_phase(train_steps)
            return

        ctx = mp.get_context("spawn")
        workers = min(self.self_play_workers, num_games)
        if workers <= 1:
            logger.warning("工作进程数不足，切换为单进程模式")
            self._single_process_self_play(num_games, temperature)
            self._train_phase(train_steps)
            return

        # 准备共享模型状态和同步标志
        model_state = {k: v.clone().share_memory_() for k, v in self.model.state_dict().items()}
        model_lock = ctx.Lock()
        self_play_done = ctx.Value('i', 0)  # 自对弈完成标志

        # 分配自对弈任务
        games_per_worker = [num_games // workers] * workers
        for i in range(num_games % workers):
            games_per_worker[i] += 1

        # 启动自对弈进程
        self_play_processes = []
        for worker_id, games in enumerate(games_per_worker, start=1):
            if games <= 0:
                continue
            p = ctx.Process(
                target=_self_play_worker,
                args=(
                    type(self.model),
                    self.game_cls,
                    model_state,
                    self.experience_pool,
                    temperature,
                    games,
                    worker_id,
                    model_lock,
                    10,  # reload_interval: 每10局重新加载模型
                ),
            )
            p.start()
            self_play_processes.append(p)
            logger.info(f"启动自对弈进程 {worker_id}，负责 {games} 局")

        # 启动训练进程
        train_process = ctx.Process(
            target=_training_worker,
            args=(
                type(self.model),
                self.experience_pool,
                train_steps,
                model_state,
                model_lock,
                self_play_done,
                self.experience_pool.capacity,
            ),
        )
        train_process.start()
        logger.info(f"启动训练进程，目标 {train_steps} 步")

        # 等待所有自对弈进程完成
        for p in self_play_processes:
            p.join()
        logger.info("所有自对弈进程已完成")
        
        # 设置自对弈完成标志
        self_play_done.value = 1  # type: ignore
        
        # 等待训练进程完成
        train_process.join()
        logger.info("训练进程已完成")

        # 从共享内存同步模型状态
        with model_lock:
            self.model.load_state_dict(model_state)
        logger.info("已从训练进程同步模型状态")

    def _self_play_phase(self, num_games: int, temperature: float) -> None:
        if USE_MULTIPROCESSING and isinstance(self.experience_pool, SharedRingBuffer):
            self._multi_process_self_play(num_games, temperature)
        else:
            self._single_process_self_play(num_games, temperature)

    def _single_process_self_play(self, num_games: int, temperature: float) -> None:
        for game_num in range(1, num_games + 1):
            try:
                self.arena.model.load_state_dict(self.model.state_dict())
                self.arena.self_play(
                    tao=temperature, experience_pool=self.experience_pool
                )
                if game_num % 50 == 0:
                    logger.info(f"  已完成 {game_num}/{num_games} 局")
            except Exception as e:
                logger.error(f"自对弈第 {game_num} 局出错: {e}")

    def _multi_process_self_play(self, num_games: int, temperature: float) -> None:
        if not isinstance(self.experience_pool, SharedRingBuffer):
            logger.warning("经验池不是 SharedRingBuffer，切换为单进程自对弈")
            self._single_process_self_play(num_games, temperature)
            return

        ctx = mp.get_context("spawn")
        workers = min(self.self_play_workers, num_games)
        if workers <= 1:
            self._single_process_self_play(num_games, temperature)
            return

        model_state = {k: v for k, v in self.model.state_dict().items()}

        games_per_worker = [num_games // workers] * workers
        for i in range(num_games % workers):
            games_per_worker[i] += 1

        processes = []
        for worker_id, games in enumerate(games_per_worker, start=1):
            if games <= 0:
                continue
            p = ctx.Process(
                target=_self_play_worker,
                args=(
                    type(self.model),
                    self.game_cls,
                    model_state,
                    self.experience_pool,
                    temperature,
                    games,
                    worker_id,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def _train_phase(self, num_steps: int) -> float:
        total_loss = 0.0
        successful_steps = 0

        for step in range(1, num_steps + 1):
            try:
                if not self._has_enough_experience():
                    continue

                loss_before = self._get_current_loss()
                self.trainer.train()

                if loss_before is not None:
                    total_loss += loss_before
                    successful_steps += 1
            except Exception as e:
                logger.warning(f"训练步骤 {step} 出错: {e}")

        avg_loss = total_loss / successful_steps if successful_steps > 0 else 0.0
        return avg_loss

    def _has_enough_experience(self) -> bool:
        if hasattr(self.experience_pool, "size"):
            return self.experience_pool.size.value >= BATCH_SIZE  # type: ignore
        if hasattr(self.experience_pool, "__len__"):
            return len(self.experience_pool) >= BATCH_SIZE  # type: ignore
        return False

    def _get_current_loss(self) -> float | None:
        if not self._has_enough_experience():
            return None

        try:
            batch = self.experience_pool.sample(BATCH_SIZE)
            if batch is None:
                return None

            states, target_policies, target_values = batch
            with torch.no_grad():
                pred_policies, pred_values = self.model(states)
                log_policies = torch.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(
                    pred_values.squeeze(), target_values.squeeze()
                )
                total_loss = policy_loss + value_loss
                return total_loss.item()
        except (ValueError, Exception):
            return None

    def _log_statistics(self, iteration: int) -> None:
        if hasattr(self.experience_pool, "size"):
            pool_size = self.experience_pool.size.value  # type: ignore
        elif hasattr(self.experience_pool, "__len__"):
            pool_size = len(self.experience_pool)  # type: ignore
        else:
            pool_size = "未知"
        logger.info(f"经验池大小: {pool_size}")
        logger.info(f"已完成迭代: {iteration}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.trainer.load_model(checkpoint_path)
        self.arena.load_model(checkpoint_path)
        logger.info(f"已加载检查点: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero 训练程序")

    parser.add_argument(
        "--quick-model",
        action="store_true",
        help="使用内置 QuickModel (仅替换模型类)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="训练迭代次数 (默认: 100)",
    )
    parser.add_argument(
        "--self-play-games",
        type=int,
        default=SELF_PLAY_NUM,
        help=f"每次迭代的自对弈局数 (默认: {SELF_PLAY_NUM})",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="每次迭代的训练步数 (默认: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="自对弈温度参数 (默认: 1.0)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="模型保存间隔 (默认: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="模型保存目录 (默认: result/models/<游戏类名>)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="日志输出间隔 (默认: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="自对弈进程数 (默认: CPU核心数-1)",
    )

    return parser.parse_args()
