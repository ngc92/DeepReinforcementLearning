import pytest
from unittest import mock
from tfdeeprl.builder import AgentBuilder
from .utils import *


@in_new_graph
def test_linear_schedule():
    sched = linear_schedule(1.0, 0.0, 0, 1000)

    assert sched(100).eval() == pytest.approx(0.9)
    assert sched(500).eval() == pytest.approx(0.5)
    assert sched(1500).eval() == pytest.approx(0.0)

    sched = linear_schedule(1.0, 0.0, 1000, 2000)
    assert sched(100).eval() == pytest.approx(1.0)
    assert sched(1100).eval() == pytest.approx(0.9)
    assert sched(1500).eval() == pytest.approx(0.5)

    sched = linear_schedule(0.7, 0.2, 0, 1000)
    assert sched(-100).eval() == pytest.approx(0.7)
    assert sched(500).eval() == pytest.approx((0.7 + 0.2) / 2)
    assert sched(1100).eval() == pytest.approx(0.2)


def test_add_replay_memory_checks():
    with pytest.raises(TypeError):
        @add_replay_memory(64, 1000)
        class Wrong: pass


@pytest.mark.parametrize("batch_size, actual_batch_size", [(None, 64), (32, 32)])
@pytest.mark.parametrize("memory_size, actual_memory_size", [(None, 1000), (50, 50)])
@in_new_graph
def test_add_replay_memory_memory_interaction(memory_size, actual_memory_size, batch_size, actual_batch_size):
    class MockBuilder(AgentBuilder):
        _build_train = mock.Mock()
        original = _build_train

    params = {}
    if memory_size is not None:
        params["memory_size"] = memory_size
    if batch_size is not None:
        params["batch_size"] = batch_size

    with mock.patch("tfdeeprl.helpers.Memory") as Memory, mock.patch("tfdeeprl.helpers.utils.tf.control_dependencies"):
        append = Memory.return_value.append = mock.Mock()
        sample = Memory.return_value.sample = mock.Mock()

        NewBuilderClass = add_replay_memory(64, 1000)(MockBuilder)

        observation = tf.placeholder(tf.float32, shape=(5, 10))
        action = tf.placeholder(tf.int32, shape=(2,))

        data = {
            "observation": observation,
            "action": action,
            "reward": "reward",
            "next_observation": "next",
            "terminal": "terminal"
        }

        obj = NewBuilderClass()
        obj._build_train(data, params)

        Memory.assert_called_once_with(actual_memory_size, (5, 10), (2,), tf.int32)
        append.assert_called_once_with(state=observation, action=action, reward="reward",
                                       terminal="terminal", next_state="next")
        sample.assert_called_once_with(actual_batch_size)

        sampled_transition = {
            "observation":      sample.return_value.current,
            "next_observation": sample.return_value.next,
            "action":           sample.return_value.action,
            "reward":           sample.return_value.reward,
            "terminal":         sample.return_value.terminal
        }

        MockBuilder.original.assert_called_once_with(obj, transition=sampled_transition, params=params)
