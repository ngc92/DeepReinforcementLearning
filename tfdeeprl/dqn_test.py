import pytest
from unittest import mock

from .dqn import *
from .helpers.utils import in_new_graph


def test_init():
    # expected callable
    with pytest.raises(TypeError):
        builder = DQNBuilder(None, [10, 5], None)


    # TODO config test


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
    builder = DQNBuilder(lambda x: x, [4, 3], None)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    assessment = builder.assess_state(ph, "test")

    build_q.assert_called_once_with(ph)
    assert (assessment.eval() == [[5, 4], [6, 9]]).all()


@in_new_graph
def test_assess_action():
    builder = DQNBuilder(lambda x: x, [4, 3], None)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    assessment = builder.assess_actions(ph, tf.constant([[1, 2], [0, 2]]), "test")

    build_q.assert_called_once_with(ph)
    assert (assessment.eval() == [[1, 2], [0, 2]]).all()


@in_new_graph
def test_act_fn_deterministic():
    builder = DQNBuilder(lambda x: x, [4, 3], None)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    actions = builder.act_fn(ph, None, "test")

    build_q.assert_called_once_with(ph)
    assert (actions.eval() == [[3, 0], [1, 1]]).all()


@in_new_graph
def test_act_fn_stochastic():
    builder = DQNBuilder(lambda x: x, [4, 3], None)

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
    builder = DQNBuilder(lambda x: x, [4, 3], None)

    build_q = mock.Mock(return_value=[[[0, 1, 2, 5], [0, 6, 2, 3]], [[4, 1, 2], [2, 9, 2]]])
    builder._build_q = build_q
    ph = tf.placeholder(dtype=tf.float32)
    actions_det = builder.act_fn(ph, None, "test")  # type: tf.Tensor
    actions_sto = builder.act_fn(ph, ph, "test")    # type: tf.Tensor

    assert actions_det.dtype == actions_sto.dtype
    assert actions_det.shape == actions_sto.shape


@in_new_graph
def test_build_act_shape():
    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3], None)
    ph = tf.placeholder(dtype=tf.float32, shape=(16,))

    action = builder._build_act(ph, {})
    shape = action.actions.shape  # type: tf.TensorShape
    shape.assert_has_rank(1)


@in_new_graph
def test_build_explore_shape():
    builder = DQNBuilder(lambda x: tf.layers.dense(x, 5), [5, 3],
                         DQNConfig(False, True, tf.train.AdamOptimizer(), lambda x: 1.0))
    ph = tf.placeholder(dtype=tf.float32, shape=(16,))

    action = builder._build_explore(ph, {})
    shape = action.actions.shape  # type: tf.TensorShape
    shape.assert_has_rank(1)

