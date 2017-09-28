import pytest
from unittest import mock

from .dqn import *
from .helpers.utils import in_new_graph
from .helpers import ops

DUMMY_CONFIG = DQNConfig(lambda x: 1.0, tf.train.AdamOptimizer(), False, True)


def test_dqn_config():
    with pytest.raises(TypeError):
        config = DQNConfig(None, tf.train.AdamOptimizer(), False, False)

    with pytest.raises(TypeError):
        config = DQNConfig(lambda x: 1.0, None, False, True)


def test_init():
    # expected callable
    with pytest.raises(TypeError):
        builder = DQNBuilder(None, [10, 5], DUMMY_CONFIG)

    # expect DQNConfig
    with pytest.raises(TypeError):
        builder = DQNBuilder(lambda: 5, [10, 5], None)

    with mock.patch("tfdeeprl.dqn.make_q_fn") as qfn:
        f = lambda x: 0
        builder = DQNBuilder(f, [10, 5], DUMMY_CONFIG)
        qfn.assert_called_once_with(f, [10, 5])


@in_new_graph
def test_make_q_fun():
    BATCH_SIZE = 12

    feature_fn = mock.Mock(return_value=tf.zeros((BATCH_SIZE, 1)))
    num_actions = [2, 3, 5]

    q_fn = make_q_fn(feature_fn, num_actions)

    state = tf.zeros((BATCH_SIZE, 1))

    with mock.patch("tensorflow.layers.dense", wraps=tf.layers.dense) as dense:
        q_vals = q_fn(state)

        # check the shape of the q_vals
        assert q_vals[0].shape == (BATCH_SIZE, num_actions[0])
        assert q_vals[1].shape == (BATCH_SIZE, num_actions[1])
        assert q_vals[2].shape == (BATCH_SIZE, num_actions[2])

        feature_fn.assert_called_once_with(state)
        dense.assert_called_once_with(feature_fn.return_value, 10, name="all_q_linear")

    with pytest.raises(TypeError):
        make_q_fn(5, [1, 2])


@in_new_graph
def test_assess_state():
    builder = DQNBuilder(lambda x: x, [4, 3], DUMMY_CONFIG)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    assessment = builder.assess_state(ph, "test")

    build_q.assert_called_once_with(ph)
    assert (assessment.eval() == [[5, 4], [6, 9]]).all()


@in_new_graph
def test_assess_action():
    builder = DQNBuilder(lambda x: x, [4, 3], DUMMY_CONFIG)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    assessment = builder.assess_actions(ph, tf.constant([[1, 2], [0, 2]]), "test")

    build_q.assert_called_once_with(ph)
    assert (assessment.eval() == [[1, 2], [0, 2]]).all()


@in_new_graph
def test_act_fn_deterministic():
    builder = DQNBuilder(lambda x: x, [4, 3], DUMMY_CONFIG)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    actions = builder.act_fn(ph, None, "test")

    build_q.assert_called_once_with(ph)
    assert (actions.eval() == [[3, 0], [1, 1]]).all()


@in_new_graph
def test_act_fn_stochastic():
    builder = DQNBuilder(lambda x: x, [4, 3], DUMMY_CONFIG)

    q_vals = [tf.constant([[0, 1, 2, 5], [0, 6, 2, 3]]), tf.constant([[4, 1, 2], [2, 9, 2]])]
    build_q = mock.Mock(return_value=q_vals)
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    ep = tf.placeholder(dtype=tf.float32)
    from tfdeeprl.helpers.ops import epsilon_greedy
    with mock.patch("tfdeeprl.dqn.epsilon_greedy", wraps=epsilon_greedy) as egreedy:
        actions = builder.act_fn(ph, ep, "test")

    build_q.assert_called_once_with(ph)
    assert egreedy.call_args_list[0] == mock.call(q_vals[0], ep, True, dtype=tf.int64)
    assert egreedy.call_args_list[1] == mock.call(q_vals[1], ep, True, dtype=tf.int64)

    # TODO do an actual check with ep=0, and maybe mock egreedy further to check everything.


@in_new_graph
def test_act_fn_type_stability():
    builder = DQNBuilder(lambda x: x, [4, 3], DUMMY_CONFIG)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    actions_det = builder.act_fn(ph, None, "test")  # type: tf.Tensor
    actions_sto = builder.act_fn(ph, ph, "test")    # type: tf.Tensor

    assert actions_det.dtype == actions_sto.dtype
    assert actions_det.shape == actions_sto.shape


@in_new_graph
def test_build_act_shape():
    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3], DUMMY_CONFIG)
    ph = tf.placeholder(dtype=tf.float32, shape=(16,))

    action = builder._build_act(ph, {})
    shape = action.actions.shape  # type: tf.TensorShape
    shape.assert_has_rank(1)


@in_new_graph
def test_build_explore_shape():
    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3],
                         DQNConfig(lambda x: 1.0, tf.train.AdamOptimizer(), False, True))
    ph = tf.placeholder(dtype=tf.float32, shape=(16,))

    action = builder._build_explore(ph, {})
    shape = action.actions.shape  # type: tf.TensorShape
    shape.assert_has_rank(1)


@in_new_graph
def test_build_explore_epsilon():
    step = tf.train.create_global_step()

    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3],
                         DQNConfig(mock.Mock(), tf.train.AdamOptimizer(), False, True))

    eps_t = tf.constant(1.0)
    eps_mock = builder.config.exploration_schedule
    eps_mock.return_value = eps_t

    builder.act_fn = mock.Mock(wraps=builder.act_fn)

    ph = tf.placeholder(dtype=tf.float32, shape=(16,))
    action = builder._build_explore(ph, {})

    eps_mock.assert_called_once_with(step)
    builder.act_fn.assert_called_once_with(mock.ANY, epsilon=eps_mock.return_value, scope=mock.ANY, reuse=False)


@in_new_graph
def test_build_train_shape():
    params = {
        "gamma": tf.constant(0.5)
    }

    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3],
                         DQNConfig(lambda x: 1.0, tf.train.AdamOptimizer(), False, True))

    data = {
        "observation": tf.constant([[5.0], [3], [4]]),
        "next_observation": tf.constant([[6.0], [4], [5]]),
        "action": tf.constant([[2, 1], [0, 0], [1, 1]]),
        "reward": tf.constant([1.0, 2.0, 1.0]),
        "terminal": tf.constant([False, False, True])
    }

    # mock the helper functions
    assess_state = builder.assess_state = mock.Mock(wraps=builder.assess_state)
    assess_actions = builder.assess_actions = mock.Mock(wraps=builder.assess_actions)

    with mock.patch("tfdeeprl.dqn.td_update", wraps=ops.td_update) as td_update:
        builder._build_train.__wrapped__(builder, data, params)

    # check that we assess actions in current state
    assess_state.assert_called_once_with(data["next_observation"], mock.ANY, reuse=True)
    assert assess_state.call_args[0][1].name == builder._target_var_scope.name

    # check that we assess the future state
    assess_actions.assert_called_once_with(data["observation"], data["action"], scope=mock.ANY, reuse=False)
    assert assess_actions.call_args[1]["scope"].name == builder._network_var_scope.name

    # check for correct td update
    # TODO future value should be return from assess_state
    td_update.assert_called_once_with(data["reward"], data["terminal"], mock.ANY, params["gamma"],
                                      name=mock.ANY)

    # TODO some more checks
    # TODO we cannot really test the algorithm like this, but we should check correct branching