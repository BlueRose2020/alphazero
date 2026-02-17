import torch
import pytest

import config
import core.MCTS_alphazero as mcts_module
from core.MCTS_alphazero import MCTS, MCTSNode
from examples.tictactoe.game import TicTacToeGame


class SimpleModel(torch.nn.Module):
	"""简单的井字棋模型，用于测试"""
	def __init__(self, policy_logits: torch.Tensor) -> None:
		super().__init__()
		self.policy_logits = policy_logits.detach().clone()

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		# 返回 [1, 9] 的策略和 [1, 1] 的价值
		return self.policy_logits.unsqueeze(0), torch.tensor([[0.2]], dtype=torch.float32)


class MCTSForTest(MCTS):
	def get_prior_for_test(
		self, policy: torch.Tensor, mask: torch.Tensor, use_Dilichlet: bool
	) -> torch.Tensor:
		return self._get_prior(policy, mask, use_Dilichlet)


def test_mctsnode_expand_and_select_prefers_high_prior() -> None:
	"""测试MCTSNode展开和选择倾向于高先验"""
	state = TicTacToeGame.initial_state()
	node = MCTSNode(TicTacToeGame, state, config.PLAYER1)

	# 初始状态有9个合法动作，设置先验偏向动作0和1
	prior = torch.zeros(1, 9, dtype=torch.float32)
	prior[0, 0] = 0.2
	prior[0, 1] = 0.8
	node.expand(prior)

	assert node.is_expand is True
	# 所有9个位置都应该有子节点，因为先验对所有位置都有非零值
	assert len(node.children) == 9

	node.visits = 10
	for child in node.children.values():
		child.value = 0.0

	selected = node.select(c_puct=1.0)
	assert selected is node.children[1]


def test_mcts_get_prior_applies_mask_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
	"""测试MCTS先验应用掩码和归一化"""
	monkeypatch.setattr(mcts_module, "USE_HISTORY", False)
	monkeypatch.setattr(config, "USE_HISTORY", False)

	mcts = MCTSForTest(TicTacToeGame)
	policy = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
	mask = torch.zeros(1, 9, dtype=torch.float32)
	mask[0, [0, 2, 4, 6, 8]] = 1.0  # 设置5个合法位置

	prior = mcts.get_prior_for_test(policy, mask, use_Dilichlet=False)

	assert prior.shape == torch.Size([1, 9])
	assert prior[0, [1, 3, 5, 7]].sum().item() == 0.0  # 非法位置为0
	assert torch.isclose(prior.sum(), torch.tensor(1.0))
	assert prior[0, 2].item() > prior[0, 0].item()  # 高策略值的位置优先级高


def test_mcts_search_returns_normalized_prior(monkeypatch: pytest.MonkeyPatch) -> None:
	"""测试MCTS搜索返回归一化的先验"""
	monkeypatch.setattr(mcts_module, "USE_HISTORY", False)
	monkeypatch.setattr(config, "USE_HISTORY", False)

	mcts = MCTS(TicTacToeGame)
	# 为9个动作设置策略值
	policy_logits = torch.tensor(
		[0.1, 2.0, -1.0, 0.5, 1.0, -0.5, 0.0, 0.3, 0.2],
		dtype=torch.float32
	)
	model = SimpleModel(policy_logits)

	prior = mcts.search(
		model,
		TicTacToeGame.initial_state(),
		config.PLAYER1,
		num_simulation=2,
		c_puct=1.0,
		use_Dirichlet=False,
	)

	assert prior.shape == torch.Size([1, 9])
	assert torch.isclose(prior.sum(), torch.tensor(1.0))
	# 动作1和4有较高的策略值，应该获得较高的先验
	assert int(torch.argmax(prior).item()) in [1, 4]


def test_mcts_search_requires_history_state_when_enabled(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""测试启用历史状态时MCTS搜索的要求"""
	monkeypatch.setattr(mcts_module, "USE_HISTORY", True)
	monkeypatch.setattr(config, "USE_HISTORY", True)

	mcts = MCTS(TicTacToeGame)
	policy_logits = torch.ones(9, dtype=torch.float32)
	model = SimpleModel(policy_logits)

	with pytest.raises(ValueError):
		mcts.search(
			model,
			TicTacToeGame.initial_state(),
			config.PLAYER1,
			history_state=None,
			num_simulation=1,
			use_Dirichlet=False,
		)
