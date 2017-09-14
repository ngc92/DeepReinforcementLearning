import pytest
import numpy as np

from .ops import *
from .utils import in_new_graph


@in_new_graph
def test_choose_from_array():
    source = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    choice = choose_from_array(source, [0, 1, 2, 1])

    assert list(choice.eval()) == [1, 5, 9, 11]


@in_new_graph
def test_epsilon_greedy_verification():
    epsilon_greedy(np.ones((4, 4)), 1, True)

    with pytest.raises(ValueError):
        epsilon_greedy([1, 2, 3], 0.5, False)

    with pytest.raises(ValueError):
        epsilon_greedy(np.ones((4, 4)), [1, 3], False)

    with pytest.raises(ValueError):
        epsilon_greedy(np.ones((4, 4)), 1.0, [False, True])

    with pytest.raises(TypeError):
        epsilon_greedy(np.ones((4, 4)), 1.0, 1)

    with pytest.raises(TypeError):
        epsilon_greedy(np.ones((4, 4)), "1.0", 1)


@in_new_graph
def test_epsilon_greedy_greedyness():
    eg = epsilon_greedy([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], 0.0, True)
    assert list(eg.eval()) == [1, 3, 0, 1]

    eg = epsilon_greedy([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], 1.0, False)
    assert list(eg.eval()) == [1, 3, 0, 1]

# TODO test the random part of E-Greedy


def target_update_fixture(f):
    from functools import wraps
    @wraps(f)
    def test(**kwargs):
        with tf.variable_scope("source"):
            tf.get_variable("a", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(1))

        with tf.variable_scope("target"):
            tf.get_variable("a", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(3))

        f(**kwargs)
    return test


@in_new_graph
@target_update_fixture
def test_target_update_checks_both_args():
    # specifying both tau and interval is an error
    with pytest.raises(ValueError):
        target_net_update("source", "target", 0.5, 100)


@in_new_graph
@target_update_fixture
def test_target_update_checks_same_scope():
    # cannot update to same scope
    with pytest.raises(ValueError):
        target_net_update("source", "source", 0.5, None)


@in_new_graph
@target_update_fixture
def test_target_update_checks_invalid_tau():
    # tau needs to be in range [0, 1]
    with pytest.raises(ValueError):
        target_net_update("source", "target", 1.5, None)

    with pytest.raises(ValueError):
        target_net_update("source", "target", -0.5, None)

    # this one does not fire at construction, only at runtime
    tnu = target_net_update("source", "target", tf.constant(-0.5), None)

    tf.global_variables_initializer().run()

    with pytest.raises(tf.errors.InvalidArgumentError):
        tnu.run()


@in_new_graph
@target_update_fixture
def test_target_update_checks_invalid_interval():
    # interval needs to be positive integer
    with pytest.raises(ValueError):
        target_net_update("source", "target", None, -5)

    with pytest.raises(TypeError):
        target_net_update("source", "target", None, 5.5)

    with pytest.raises(TypeError):
        target_net_update("source", "target", None, tf.constant(5.5))

    tno_0 = target_net_update("source", "target", None, tf.constant(0))

    assert isinstance(tno_0, tf.Operation)

    tf.global_variables_initializer().run()
    with pytest.raises(tf.errors.InvalidArgumentError):
        tno_0.run()


@in_new_graph
@target_update_fixture
def test_target_update_soft(monkeypatch):
    # monkeyparch the update_from_scope function
    check = tf.get_variable("check", (), tf.bool, initializer=tf.constant_initializer(False))

    def fake_update_from_scope(source_scope, target_scope, rate, name=None):
        assert source_scope == "source"
        assert target_scope == "target"
        assert rate.eval() == 0.5
        return check.assign(tf.constant(True))

    monkeypatch.setattr("tfdeeprl.helpers.scoping.update_from_scope", fake_update_from_scope)

    tno = target_net_update("source", "target", 0.5, None)
    tf.global_variables_initializer().run()
    tno.run()

    assert check.eval()


@in_new_graph
@target_update_fixture
def test_target_update_hard(monkeypatch):
    # monkeyparth the assign_from_scope function
    counter = tf.get_variable("counter", (), tf.int32, initializer=tf.constant_initializer(0))

    def fake_assign_from_scope(source_scope, target_scope, name=None):
        assert source_scope == "source"
        assert target_scope == "target"
        with tf.control_dependencies([counter.assign_add(1)]):
            return tf.no_op()

    monkeypatch.setattr("tfdeeprl.helpers.scoping.assign_from_scope", fake_assign_from_scope)
    global_step = tf.train.create_global_step()

    tno = target_net_update("source", "target", None, 10)
    tf.global_variables_initializer().run()
    assert counter.eval() == 0
    tno.run()
    assert counter.eval() == 1
    global_step.assign(1).eval()
    tno.run()
    assert counter.eval() == 1
    global_step.assign(10).eval()
    tno.run()
    assert counter.eval() == 2


