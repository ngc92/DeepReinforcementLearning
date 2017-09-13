import pytest

from .scoping import *
from .scoping import _get_variables_from_scope
from .utils import in_new_graph


@in_new_graph
def test_scope_name():
    assert scope_name() == ""

    with tf.variable_scope("a"):
        assert scope_name() == "a"

    with tf.name_scope("b"):
        assert scope_name() == ""


@in_new_graph
def test_absolute_scope():
    with tf.variable_scope("a"):
        assert absolute_scope_name("b") == "a/b"

    assert absolute_scope_name("b") == "b"

    with tf.name_scope("c"):
        assert absolute_scope_name("b") == "b"


@in_new_graph
def test_copy_vars():
    with tf.variable_scope("a") as scope:
        a = tf.get_variable("A", shape=(13,), dtype=tf.float32, initializer=tf.constant_initializer(5))
        b = tf.get_variable("B", shape=(54,), dtype=tf.int64, initializer=tf.constant_initializer(42))
        c = tf.get_variable("C", shape=(18,), dtype=tf.float32, initializer=tf.constant_initializer(42),
                            collections=[tf.GraphKeys.LOCAL_VARIABLES])

    copy_variables_to_scope(scope, "b", True)

    with tf.variable_scope("b", reuse=True):
        a_ = tf.get_variable("A")
        b_ = tf.get_variable("B", dtype=tf.int64)
        # check that only GLOBAL variables are copied
        with pytest.raises(ValueError):
            c_ = tf.get_variable("C")

    assert a_.dtype == a.dtype
    assert b_.dtype == b.dtype
    assert a_.shape == a.shape
    assert b_.shape == b.shape
    assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "b")) == 2

    # now check init
    tf.global_variables_initializer().run()

    assert (a_.eval() == a.eval()).all()
    assert (b_.eval() == b.eval()).all()


@in_new_graph
def test_vars_from_scope():
    def make_var(name):
        return tf.get_variable(name, shape=(), dtype=tf.float32)

    with tf.variable_scope("a"):
        a = make_var("A")
        op = tf.constant(5, name="const")
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, op)
        with tf.variable_scope("b") as b_scope:
            b = make_var("B")

    bait = make_var("bait")
    assert bait in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "b")

    with tf.variable_scope("b") as c_scope:
        c = make_var("C")

    assert _get_variables_from_scope("a", tf.GraphKeys.GLOBAL_VARIABLES) == {a, b}
    assert _get_variables_from_scope("b", tf.GraphKeys.GLOBAL_VARIABLES) == {c}
    assert _get_variables_from_scope(b_scope, tf.GraphKeys.GLOBAL_VARIABLES) == {b}

    with tf.variable_scope("a"):
        assert _get_variables_from_scope("b", tf.GraphKeys.GLOBAL_VARIABLES) == {b}
        assert _get_variables_from_scope(c_scope, tf.GraphKeys.GLOBAL_VARIABLES) == {c}


@in_new_graph
def test_assign_from_scope():
    with tf.variable_scope("a"):
        with tf.variable_scope("b") as scope:  # type: tf.VariableScope
            a = tf.get_variable("A", shape=(13,), dtype=tf.float32, initializer=tf.constant_initializer(5))
        bait = tf.get_variable("bait", shape=(16,), dtype=tf.float32, initializer=tf.constant_initializer(8))
    with tf.variable_scope("c") as target_scope:
        a_ = tf.get_variable("A", shape=(13,), dtype=tf.float32, initializer=tf.constant_initializer(12))
        bait_ = tf.get_variable("bait", shape=(16,), dtype=tf.float32, initializer=tf.constant_initializer(10))

    tf.global_variables_initializer().run()

    assert (a.eval() != a_.eval()).any()
    assert (bait.eval() != bait_.eval()).any()

    asg = assign_from_scope("a/b", "c", name="test_assign")
    asg.run()

    assert (a.eval() == a_.eval()).all()
    assert (a.eval() == 5).all()
    assert (bait.eval() != bait_.eval()).any()

    # check that everything works even if we include non-variables in the collection
    with tf.variable_scope(target_scope):
        constant_ = tf.get_variable("constant", shape=(13,), dtype=tf.float32, initializer=tf.constant_initializer(12))  # type: tf.Variable

    tf.global_variables_initializer().run()

    with tf.name_scope(scope.original_name_scope):
        op = tf.constant(5, name="constant")
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, op)

    # do not assign from non_variables
    assign_from_scope(scope, target_scope, name="test_assign").run()
    assert (constant_.eval() == 12).all()


@in_new_graph
def test_update_from_scope():
    with tf.variable_scope("a"):
        with tf.variable_scope("b") as scope:  # type: tf.VariableScope
            a = tf.get_variable("A", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(6))
        bait = tf.get_variable("bait", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(8))
    with tf.variable_scope("c") as target_scope:
        a_ = tf.get_variable("A", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(12))
        bait_ = tf.get_variable("bait", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(10))

    tf.global_variables_initializer().run()

    assert (a.eval() != a_.eval()).any()
    assert (bait.eval() != bait_.eval()).any()

    asg = update_from_scope("a/b", "c", name="test_update", rate=0.5)
    asg.run()

    assert (a.eval() == 6).all()
    assert (a_.eval() == 9).all()

    asg = update_from_scope("a/b", "c", name="test_update", rate=1.0)
    asg.run()

    assert (a.eval() == 6).all()
    assert (a_.eval() == 6).all()
