import os
import tempfile
import torch
from config import ExperienceDate
from utils.experience_pool import ExperiencePool


def _make_experience(
    state_dim: tuple[int, ...], num_action: int, value: float
) -> ExperienceDate:
    state = torch.randn(state_dim, dtype=torch.float32)
    policy = torch.softmax(torch.randn(1, num_action, dtype=torch.float32), dim=1)
    val = torch.tensor([value], dtype=torch.float32)
    return state, policy, val


def test_experience_pool_put_len_and_sample_shapes() -> None:
    pool = ExperiencePool(deque_len=3)

    shapes: list[tuple[int, ...]] = [(3, 3), (5, 5), (6,), (7, 8, 9)]
    for shape in shapes:
        for i in range(4):
            pool.put(_make_experience(shape, 3, float(i)))

        assert len(pool) == 3

        result = pool.sample(batch_size=2)
        assert result is not None
        states, policies, values = result

        assert states.shape == (2, *shape)
        assert policies.shape == (2, 3)
        assert values.shape == (2, 1)


def test_experience_pool_save_load_and_clear() -> None:
    pool = ExperiencePool(deque_len=2)
    shapes: list[tuple[int, ...]] = [(2, 2), (3, 3), (4,), (5, 6, 7)]
    for shape in shapes:
        pool.put(_make_experience(shape, 2, 1.0))
        pool.put(_make_experience(shape, 2, 2.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "pool.pkl")
            pool.save(file_path)

            new_pool = ExperiencePool(deque_len=2)
            new_pool.load(file_path)

            assert len(new_pool) == 2
            sample = new_pool.sample(batch_size=1)
            assert sample is not None
            states, _, _ = sample
            assert states.shape == (1, *shape)

    pool.clear()
    assert len(pool) == 0
