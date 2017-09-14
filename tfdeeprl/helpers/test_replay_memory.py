import numpy as np
import pytest
import tensorflow as tf

from .utils import in_new_graph
from .replay_memory import Memory, QSample


@in_new_graph
def test_init_args_check():
    with pytest.raises(ValueError):
        Memory(-1, (10,), (10,), tf.int64)

    with pytest.raises(ValueError):
        Memory(10, (-5,), (10,), tf.int64)

    with pytest.raises(ValueError):
        Memory(10, (10,), (0,), tf.int64)

    with pytest.raises(ValueError):
        Memory(10, (10,), (10,), tf.string)

    with pytest.raises(TypeError):
        Memory(10, (10,), (10,), "?")


@in_new_graph
def test_append_args_check():
    memory = Memory(10, (10,), (1,), tf.int64)

    # shape checks
    with pytest.raises(ValueError):
        memory.append(np.zeros(4), 1, 5.6, False, np.zeros(10))

    with pytest.raises(ValueError):
        memory.append(np.zeros(10), [1, 2], 5.6, False, np.zeros(10))

    with pytest.raises(ValueError):
        memory.append(np.zeros(10), [1], [5.6, 1.4], False, np.zeros(10))

    with pytest.raises(ValueError):
        memory.append(np.zeros(10), [1], 5.6, [False, False], np.zeros(10))

    with pytest.raises(ValueError):
        memory.append(np.zeros(10), [1], 5.6, False, np.zeros(15))

    # type checks 
    with pytest.raises(TypeError):
        memory.append(np.zeros(10), [1], 5.6, 1.7, np.zeros(10))

    with pytest.raises(TypeError):
        memory.append(np.zeros(10), [1.9], 5.6, True, np.zeros(10))


@in_new_graph
def test_sample_empty():
    memory = Memory(10, (10,), (1,), tf.int32)

    sample = memory.sample(5)
    tf.get_default_session().run(tf.global_variables_initializer())

    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.get_default_session().run(sample)


@in_new_graph
def test_length():
    memory = Memory(10, (1,), (1,), tf.int32)
    append = memory.append([1], [1], 1, False, [1])  # type: tf.Operation

    tf.global_variables_initializer().run()

    assert memory.length.eval() == 0
    append.run()
    assert memory.length.eval() == 1
    for i in range(10):
        append.run()
    assert memory.length.eval() == 10


@in_new_graph
def test_sampling():
    memory = Memory(10, (1,), (1,), tf.int32)
    append = memory.append([1], [0], 1, False, [2])  # type: tf.Operation
    sample = memory.sample(1)

    tf.global_variables_initializer().run()

    append.run()
    result = tf.get_default_session().run(sample)
    expected = QSample([1], [0], 1, terminal=False, next=[2])
    assert result == expected


@in_new_graph
def test_length_in_loop():
    memory = Memory(10, (1,), (1,), tf.int32)
    counter = tf.constant(0)

    def body(ctr: tf.Tensor):
        with tf.control_dependencies([memory.append([1], [1], 1, False, [1])]):
            with tf.control_dependencies([tf.assert_equal(memory.length, ctr+1)]):
                return ctr + 1

    loop = tf.while_loop(lambda x: x < 10, body, [counter])
    tf.global_variables_initializer().run()
    assert loop.eval() == 10
