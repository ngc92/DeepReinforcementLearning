import pytest
import os

from .builder import AgentActSpec, AgentTrainSpec
from .helpers import in_new_graph
from .agent import *
from .builder_test import MockBuilder
import gym
from unittest import mock


def test_agent_init():
    agent = Agent(MockBuilder(), "mdir", {1: 5})
    assert agent.params == {1: 5}
    assert agent.model_dir == "mdir"

    agent = Agent(MockBuilder())
    assert os.path.exists(agent.model_dir)
    assert agent.params == {}

    with pytest.raises(TypeError):
        agent = Agent(5)


def test_agent_params_protection():
    original_dict = {1: 5, 2: []}
    agent = Agent(MockBuilder(), "mdir", original_dict)
    original_dict[1] = 10

    assert agent.params == {1: 5, 2: []}
    agent.params[1] = 10
    assert agent.params == {1: 5, 2: []}
    agent.params[2].append(1)
    original_dict[2].append(10)
    assert agent.params == {1: 5, 2: []}


@in_new_graph
def test_agent_build_train_graph():
    import gym.spaces

    # setting up the mocks
    builder = MockBuilder()

    def _build_explore(o, params):
        return AgentActSpec(actions=tf.constant(1), metrics={}, is_exploring=True)

    def _build_train(o, params):
        assert "observation" in o
        assert "action" in o
        assert "reward" in o
        assert "terminal" in o
        assert "next_observation" in o
        return AgentTrainSpec(loss=None, train_op=tf.no_op(), metrics={})

    builder._build_explore = mock.Mock(wraps=_build_explore)
    builder._build_train = mock.Mock(wraps=_build_train)

    env = mock.create_autospec(gym.Env(),)
    env.observation_space = gym.spaces.Box(0, 1, shape=(1,))

    params = {1: 5, 2: 10}
    agent = Agent(builder, params=params)
    r, g = agent.build_training_graph(env)

    assert isinstance(r["return"], tf.Tensor)
    assert isinstance(r["duration"], tf.Tensor)
    assert isinstance(g, tf.Graph)

    assert builder._build_train.call_count == 1
    assert builder._build_explore.call_count == 1

    assert builder._build_explore.call_args == mock.call(mock.ANY, params=params)
    assert builder._build_train.call_args == mock.call(mock.ANY, params=params)

    # I am not sure how to best test the intricacies of the training loop graph here ...


def test_agent_train_loop():
    # we mock up the training graph, so this test remains quite simple
    def build_training_graph(e):
        with tf.Graph().as_default() as graph:
            return_ = {"return": tf.constant(5.0), "duration": tf.constant(3, tf.int64)}
            return return_, graph

    agent = Agent(MockBuilder())
    agent.build_training_graph = build_training_graph

    returns, durations = agent.train(gym.Env(), 10)
    assert list(returns) == [5.0] * 10
    assert list(durations) == [3] * 10


@in_new_graph
def test_agent_train_session_setup():
    # check that the session is created in the correct graph
    builder = MockBuilder()
    agent = Agent(builder)

    agent.build_training_graph = mock.Mock()

    target_graph = tf.Graph()
    mock_session = mock.MagicMock()
    def check_is_target(*args, **kwargs):
        assert tf.get_default_graph() == target_graph
        mocked = mock.MagicMock()
        mocked.__enter__.return_value = mock_session
        return mocked

    train_ops = mock.Mock()
    agent.build_training_graph.return_value = ([train_ops], target_graph)

    with mock.patch("tfdeeprl.agent.tf.train.SingularMonitoredSession", side_effect=check_is_target) as mock_session_fn:
        mock_session.run.return_value = {"return": 8.0, "duration": 5}
        returns, durations = agent.train(mock.create_autospec(gym.Env()), 1)

        # now the checks
        mock_session_fn.assert_called_once_with(checkpoint_dir=agent.model_dir, hooks=mock.ANY)
        mock_session.run.assert_called_once_with([train_ops])

        assert returns == [8.0]
        assert durations == [5]
