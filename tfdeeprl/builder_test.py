import pytest
import tensorflow as tf
from .helpers.utils import in_new_graph
from .builder import *
from unittest import mock


class MockBuilder(AgentBuilder):
    def __init__(self):
        super(MockBuilder, self).__init__()
        self._build_explore = mock.Mock()
        self._build_act = mock.Mock()
        self._build_train = mock.Mock()


@in_new_graph
def test_act_spec_type_checks():
    with pytest.raises(TypeError):
        a = AgentActSpec(actions="test", metrics={}, is_exploring=False)
    with pytest.raises(TypeError):
        a = AgentActSpec(actions=None, metrics={}, is_exploring=False)
    with pytest.raises(TypeError):
        a = AgentActSpec(actions=tf.placeholder(tf.float32), metrics={}, is_exploring="NOT A BOOL")
    with pytest.raises(TypeError):
        a = AgentActSpec(actions=tf.placeholder(tf.float32), metrics=True, is_exploring=False)
    # not setting metrics should be fine
    a = AgentActSpec(actions=tf.placeholder(tf.float32), is_exploring=True)
    assert a.metrics == {}


@in_new_graph
def test_train_spec_type_checks():
    with pytest.raises(TypeError):
        a = AgentTrainSpec(loss="test", train_op=tf.no_op(), metrics={})
    with pytest.raises(TypeError):
        a = AgentTrainSpec(loss=tf.zeros(()), train_op="test", metrics={})
    with pytest.raises(TypeError):
        a = AgentTrainSpec(loss=tf.zeros(()), train_op=tf.no_op(), metrics="")
    with pytest.raises(TypeError):
        a = AgentTrainSpec(loss=tf.zeros((), dtype=tf.complex64), train_op=tf.no_op(), metrics="")
    with pytest.raises(ValueError):
        a = AgentTrainSpec(loss=tf.zeros((10, 5)), train_op=tf.no_op(), metrics={})
    # not setting metrics or loss should be fine
    a = AgentTrainSpec(train_op=tf.no_op())
    assert a.metrics == {}
    assert a.loss is None


@in_new_graph
def test_build_act_errors():
    builder = MockBuilder()
    aph = tf.placeholder(tf.float32)

    with pytest.raises(TypeError):
        builder.act([], {})

    builder._build_act = mock.Mock(return_value=AgentActSpec(actions=aph, metrics={}, is_exploring=False))
    # this works
    builder.act([], {})

    builder._build_act = mock.Mock(return_value=AgentActSpec(actions=aph, metrics={}, is_exploring=True))
    # we expect that exploring is disabled here.
    with pytest.raises(ValueError):
        builder.act([], {})


@in_new_graph
def test_build_explore_errors():
    builder = MockBuilder()
    aph = tf.placeholder(tf.float32)

    with pytest.raises(TypeError):
        builder.explore([], {})

    builder._build_explore = mock.Mock(return_value=AgentActSpec(actions=aph, is_exploring=True))
    # this works
    builder.explore([], {})

    builder._build_explore = mock.Mock(return_value=AgentActSpec(actions=aph, is_exploring=False))
    # we expect that exploring is disabled here.
    with pytest.raises(ValueError):
        builder.explore([], {})


@in_new_graph
def test_build_train_errors():
    builder = MockBuilder()

    with pytest.raises(TypeError):
        builder.train({}, {})

    with mock.patch.object(builder, "_build_train") as build_train:
        build_train.return_value = AgentTrainSpec(loss=None, train_op=tf.no_op(), metrics={})

        # this works
        builder.train({}, {})

        # but without training it should not work
        with pytest.raises(ValueError):
            build_train.return_value = AgentTrainSpec(loss=None, train_op=None, metrics={})
            builder.train({}, {})

