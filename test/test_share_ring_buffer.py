import time
import torch
import torch.multiprocessing as mp
from config import ExperienceDate

from utils.share_ring_buffer import SharedRingBuffer


def _make_item(
    state_shape: tuple[int, ...], num_action: int, value: float
) -> ExperienceDate:
    state = torch.randn(state_shape, dtype=torch.float32)
    policy = torch.softmax(torch.randn(num_action, dtype=torch.float32), dim=0)
    val = torch.tensor([value], dtype=torch.float32)
    return state, policy, val


def _mp_put_items(
    buffer: SharedRingBuffer,
    items: list[ExperienceDate],
    result_queue: "mp.Queue[bool]",
) -> None:
    for item in items:
        if not buffer.put(*item, timeout=1.0):
            result_queue.put(False)
            return
    result_queue.put(True)


def _mp_get_items(
    buffer: SharedRingBuffer,
    expected_items: list[ExperienceDate],
    result_queue: "mp.Queue[bool]",
) -> None:
    for expected in expected_items:
        out = buffer.get(timeout=1.0)
        if out is None:
            result_queue.put(False)
            return
        if not (
            torch.allclose(out[0], expected[0])
            and torch.allclose(out[1], expected[1])
            and torch.allclose(out[2], expected[2])
        ):
            result_queue.put(False)
            return
    result_queue.put(True)


def test_shared_ring_buffer_put_get_order_and_wrap() -> None:
    shapes:list[tuple[int, ...]] = [(4,), (5, 5), (6,), (7, 8, 9)]
    for shape in shapes:
        buffer = SharedRingBuffer(state_shape=shape, action_dim=3, capacity=2)

        item1 = _make_item(shape, 3, 1.0)
        item2 = _make_item(shape, 3, 2.0)
        item3 = _make_item(shape, 3, 3.0)
        assert buffer.put(*item1)
        assert buffer.put(*item2)
        assert buffer.size.value == 2  # type: ignore

        out1 = buffer.get(timeout=0.1)
        assert out1 is not None
        assert torch.allclose(out1[0], item1[0])
        assert torch.allclose(out1[1], item1[1])
        assert torch.allclose(out1[2], item1[2])
        assert buffer.size.value == 1  # type: ignore

        assert buffer.put(*item3)
        assert buffer.size.value == 2  # type: ignore

        out2 = buffer.get(timeout=0.1)
        assert out2 is not None
        assert torch.allclose(out2[0], item2[0])
        assert torch.allclose(out2[1], item2[1])
        assert torch.allclose(out2[2], item2[2])

        out3 = buffer.get(timeout=0.1)
        assert out3 is not None
        assert torch.allclose(out3[0], item3[0])
        assert torch.allclose(out3[1], item3[1])
        assert torch.allclose(out3[2], item3[2])

        assert buffer.size.value == 0  # type: ignore


def test_shared_ring_buffer_timeout_and_sample() -> None:
    buffer = SharedRingBuffer(state_shape=(2,), action_dim=2, capacity=2)

    start = time.time()
    out = buffer.get(timeout=0.01)
    elapsed = time.time() - start

    assert out is None
    assert elapsed < 0.5

    sample = buffer.sample(batch_size=1, timeout=0.01)
    assert sample is None

    item = _make_item((2,), 2, 1.0)
    buffer.put(*item)

    batch = buffer.sample(batch_size=4, timeout=0.1)
    assert batch is not None

    states, policies, values = batch
    assert states.shape == (4, 2)
    assert policies.shape == (4, 2)
    assert values.shape == (4, 1)


def test_shared_ring_buffer_multiprocess_put_get() -> None:
    ctx = mp.get_context("spawn")
    buffer = SharedRingBuffer(state_shape=(3,), action_dim=2, capacity=4)

    items = [
        _make_item((3,), 2, 1.0),
        _make_item((3,), 2, 2.0),
        _make_item((3,), 2, 3.0),
    ]

    put_queue: "mp.Queue[bool]" = ctx.Queue()
    get_queue: "mp.Queue[bool]" = ctx.Queue()

    producer = ctx.Process(target=_mp_put_items, args=(buffer, items, put_queue))
    consumer = ctx.Process(target=_mp_get_items, args=(buffer, items, get_queue))

    consumer.start()
    producer.start()

    producer.join(timeout=10)
    consumer.join(timeout=10)

    if producer.is_alive():
        producer.terminate()
        producer.join()
        assert False, "producer did not finish"
    if consumer.is_alive():
        consumer.terminate()
        consumer.join()
        assert False, "consumer did not finish"

    assert put_queue.get(timeout=1.0) is True
    assert get_queue.get(timeout=1.0) is True
