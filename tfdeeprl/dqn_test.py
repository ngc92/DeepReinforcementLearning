import pytest
from unittest import mock

from .dqn import *
from .helpers.utils import in_new_graph


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
