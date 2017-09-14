from typing import Optional

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


def target_net_update(source_scope, target_scope, tau: Optional, interval: Optional, name: Optional = None,
                      collection=(tf.GraphKeys.UPDATE_OPS,)) -> tf.Operation:
    """
    This function creates an Op for updating the target variables in a deep reinforcement learning algorithm.
    Either overwriting them completely every `interval` steps, or partially with interpolation factor `tau`
    every step. The passing of time steps is determined by the `global_step` tensor.
    :param source_scope: Scope of the active net parameters.
    :param target_scope: Scope of the target net parameters.
    :param tau: Soft update interpolation factor, or None if hard update is preferred.
    :param interval: Time interval between target updates, of None if soft updates are preferred.
    :param name: Name for the op. Defaults to update_target_net.
    :param collection: Adds the resulting operation to these collections. Defaults to `UPDATE_OPS`.
    :return: The update operation.
    """
    from .scoping import update_from_scope, assign_from_scope
    if tau is not None and interval is not None:
        raise ValueError("Cannot specify tau (%s) and interval (%s) at the same time" % (tau, interval))

    with tf.name_scope(name, "update_target_net"):
        if tau is not None:
            tau = tf.convert_to_tensor(tau, tf.float32, name="tau")
            update_target_vars = update_from_scope(source_scope, target_scope, tau)
        else:
            interval = tf.convert_to_tensor(interval, name="interval")
            if not interval.dtype.is_integer:
                raise TypeError("Update interval has to be of integral type, got {}".format(interval))
            interval = tf.cast(interval, tf.int64)

            def make_target_update():
                return assign_from_scope(source_scope, target_scope)

            step = tf.train.get_or_create_global_step()

            update_target_vars = tf.cond(tf.equal(step % interval, 0),
                                         true_fn=make_target_update, false_fn=lambda: tf.no_op())

    for col in collection:
        tf.add_to_collection(col, update_target_vars)

    return update_target_vars
