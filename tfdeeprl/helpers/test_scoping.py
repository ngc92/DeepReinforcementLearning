import pytest
from .scoping import *
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
