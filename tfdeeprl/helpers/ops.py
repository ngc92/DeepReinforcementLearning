import tensorflow as tf


def choose_from_array(source, indices, name="choose_from_array") -> tf.Tensor:
    """
    returns [source[i, indices[i]] for i in 1:len(indices)]
    """
    indices = tf.convert_to_tensor(indices, name="indices")
    source = tf.convert_to_tensor(source, name="indices")

    with tf.name_scope(name, values=[indices, source]):
        num_samples = tf.shape(indices)[0]
        indices     = tf.transpose(tf.stack([tf.cast(tf.range(0, num_samples), indices.dtype), indices]))
        values      = tf.gather_nd(source, indices)
    return values


def epsilon_greedy(values, epsilon, stochastic, name=None, dtype=tf.int32) -> tf.Tensor:
    """
    This op performs epsilon greedy selection based on `values`, using a random index with probability
    `epsilon`.
    :param values: A Tensor that contains the values based on which to perform greedy selection.
                   Expected to have shape `(BATCH_SIZE, NUM_ACTIONS)`
    :param epsilon: Probability for choosing a random action in [0, NUM_ACTIONS)
    :param stochastic: If this is set to False, only greedy actions will be selected.
    :param name: Name for the operation. Defaults to "epsilon_greedy"
    :param dtype: Data type of the resulting action tensor.
    :return: The selected actions. Tensor of shape (BATCH_SIZE) and type `dtype`.
    :raises: ValueError, if values do not have rank two.
    :raises: TypeError, if stochastic is not of boolean type.
    """

    values = tf.convert_to_tensor(values, name="values")  # type: tf.Tensor
    values.shape.assert_has_rank(2)

    epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32, name="epsilon")  # type: tf.Tensor
    epsilon.shape.assert_has_rank(0)

    stochastic = tf.convert_to_tensor(stochastic, name="stochastic")  # type: tf.Tensor
    stochastic.shape.assert_has_rank(0)
    if stochastic.dtype is not tf.bool:
        raise TypeError("stochastic has to be a boolean.")

    with tf.name_scope(name, default_name="epsilon_greedy", values=[values, epsilon, stochastic]):
        with tf.name_scope("batch_size"):
            batch_size = tf.shape(values)[0]

        greedy = tf.cast(tf.argmax(values, axis=1), dtype)

        def eps_greedy():
            do_greedy = tf.random_uniform((batch_size,), 0.0, 1.0, dtype=tf.float32) > epsilon
            random = tf.random_uniform((batch_size,), minval=0, maxval=values.shape[1], dtype=dtype)
            action = tf.where(condition=do_greedy,
                              x=greedy,
                              y=random)
            return action
        return tf.cond(stochastic, eps_greedy, lambda: tf.identity(greedy))
