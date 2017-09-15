import gym
import gym.spaces
import numpy as np
import pytest
import tensorflow as tf

from .utils import in_new_graph
from .rollout_ops import gym_reset, gym_step, rollout


class MockEnv(gym.Env):
    action_space = gym.spaces.Discrete(5)
    observation_space = gym.spaces.Box(0, 1, shape=(2, 2))

    def __init__(self, observations, rewards):
        self._reset_called = 0
        self._step_called = 0
        self._action_log = []

        assert len(rewards) == len(observations)
        for obs in observations:
            assert obs.shape == MockEnv.observation_space.shape

        self._observations = observations
        self._rewards = rewards
        self._step_count = 0

    def _step(self, action):
        self._step_called += 1
        self._step_count += 1
        self._action_log += [action]
        if self._step_count == len(self._observations):
            return self._observations[-1], self._rewards[self._step_count-1], True, {}
        return self._observations[self._step_count], self._rewards[self._step_count-1], False, {}

    def _render(self, mode='human', close=False):
        pass

    def _reset(self):
        self._reset_called += 1
        self._step_count = 0
        return self._observations[0]

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass


@in_new_graph
def test_reset_op():
    mock = MockEnv([np.ones((2, 2))], [1.0])
    reset_op = gym_reset(mock)

    assert mock._reset_called == 0
    assert mock._step_called == 0

    init_obs = reset_op.eval()

    assert mock._reset_called == 1
    assert (init_obs == np.ones((2, 2))).all()


@in_new_graph
def test_step_checks():
    mock = MockEnv([np.ones((2, 2))], [1.0])
    action = tf.placeholder(tf.int32, (10,))
    with pytest.raises(ValueError):
        gym_step(mock, action)

    action = tf.placeholder(tf.float32, (1,))
    with pytest.raises(TypeError):
        gym_step(mock, action)

    # TODO add tests for other action spaces


@in_new_graph
def test_step_op():
    mock = MockEnv([np.ones((2, 2)), 2*np.ones((2, 2)), 3*np.ones((2, 2))], [1.0, 2.0, 3.0])
    reset_op = gym_reset(mock)
    action = tf.placeholder(tf.int32, (1,))
    step_op = gym_step(mock, action)

    # first need to reset, then we can step
    reset_op.eval()
    step_result = tf.get_default_session().run(step_op, {action: [2]})

    assert mock._reset_called == 1
    assert mock._step_called == 1
    assert mock._action_log == [2]

    assert (step_result.observation == 2*np.ones((2, 2))).all()
    assert step_result.reward == 1.0
    assert not step_result.terminal

    step_result = tf.get_default_session().run(step_op, {action: [3]})

    assert mock._reset_called == 1
    assert mock._step_called == 2
    assert mock._action_log == [2, 3]

    assert (step_result.observation == 3*np.ones((2, 2))).all()
    assert step_result.reward == 2.0
    assert not step_result.terminal

    step_result = tf.get_default_session().run(step_op, {action: [4]})

    assert mock._reset_called == 1
    assert mock._step_called == 3
    assert mock._action_log == [2, 3, 4]

    assert (step_result.observation == 3*np.ones((2, 2))).all()
    assert step_result.reward == 3.0
    assert step_result.terminal


@in_new_graph
def test_rollout():
    observations = [np.ones((2, 2)), 2 * np.ones((2, 2)), 3 * np.ones((2, 2))]
    rewards = [1.0, 2.0, 3.0]
    mock = MockEnv(observations, rewards)

    def action_fn(observation):
        observations_t = tf.constant(np.array(observations), shape=(3, 2, 2), dtype=tf.float32)
        counter = tf.get_variable("counter", (), tf.int32, tf.zeros_initializer())
        expected = observations_t[counter, :, :]
        with tf.control_dependencies([tf.Assert(tf.reduce_all(tf.equal(expected, observation)),
                                                [counter, expected, observation])]):
            with tf.control_dependencies([counter.assign_add(1)]):
                return tf.ones((1,), tf.int32)

    def record_fn(S, A, R, T, S2):
        observations_t = tf.constant(np.array(observations), shape=(3, 2, 2), dtype=tf.float32)
        rewards_t = tf.constant(np.array(rewards), shape=(3,), dtype=tf.float32)
        counter = tf.get_variable("counter_record", (), tf.int32, tf.zeros_initializer())

        expected_S = observations_t[counter, :, :]
        expected_R = rewards_t[counter]
        expected_S2 = observations_t[tf.minimum(counter+1, 2), :, :]
        expected_S = tf.Print(expected_S, [counter, S, A, R, T, S2, expected_R, expected_S], message="STATE", summarize=10)

        assert_S = tf.Assert(tf.reduce_all(tf.equal(expected_S, S)), [counter, expected_S, S], name="assert_obs")
        assert_R = tf.Assert(tf.reduce_all(tf.equal(expected_R, R)), [counter, expected_R, R], name="assert_reward")
        assert_A = tf.Assert(tf.reduce_all(tf.equal(1, A)), [counter, A], name="assert_action")
        assert_T = tf.Assert(tf.equal(T, tf.equal(counter, 2)), [counter, T], name="assert_terminal")
        assert_S2 = tf.Assert(tf.reduce_all(tf.equal(expected_S2, S2)), [counter, expected_S2, S2], name="assert_next")

        with tf.control_dependencies([assert_S, assert_A, assert_R, assert_S2, assert_T]):
            return counter.assign_add(1)
    result = rollout(mock, action_fn, record_fn)

    tf.get_variable_scope().reuse_variables()
    counter_r = tf.get_variable("counter_record", dtype=tf.int32)

    tf.global_variables_initializer().run()

    assert result.eval() == 6
    assert counter_r.eval() == 3
