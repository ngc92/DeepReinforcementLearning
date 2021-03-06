from numbers import Number
from typing import Optional

import tensorflow as tf


def choose_from_array(source, indices, name="choose_from_array") -> tf.Tensor:
    """
    returns [source[i, indices[i]] for i in 1:len(indices)]
    """
    with tf.name_scope(name, values=[indices, source]):
        indices = tf.convert_to_tensor(indices, name="indices")
        source = tf.convert_to_tensor(source, name="source")

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

    with tf.name_scope(name, default_name="epsilon_greedy", values=[values, epsilon, stochastic]):
        values = tf.convert_to_tensor(values, name="values")  # type: tf.Tensor
        values.shape.assert_has_rank(2)

        epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32, name="epsilon")  # type: tf.Tensor
        epsilon.shape.assert_has_rank(0)

        stochastic = tf.convert_to_tensor(stochastic, name="stochastic")  # type: tf.Tensor
        stochastic.shape.assert_has_rank(0)
        if stochastic.dtype is not tf.bool:
            raise TypeError("stochastic has to be a boolean.")

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
    from .scoping import update_from_scope, assign_from_scope, absolute_scope_name
    if tau is not None and interval is not None:
        raise ValueError("Cannot specify tau (%s) and interval (%s) at the same time" % (tau, interval))

    if absolute_scope_name(source_scope) == absolute_scope_name(target_scope):
        raise ValueError("Source and target scope are identical ({})".format(absolute_scope_name(target_scope)))

    with tf.name_scope(name, "update_target_net", [tau, interval]):
        if tau is not None:
            if isinstance(tau, Number):
                if tau < 0 or tau > 1:
                    raise ValueError("Interpolation parameter must be between 0 and 1, got {}".format(tau))

            tau = tf.convert_to_tensor(tau, tf.float32, name="tau")
            with tf.control_dependencies([tf.Assert(tf.logical_and(0.0 <= tau, 1.0 >= tau), data=[tau])]):
                update_target_vars = update_from_scope(source_scope, target_scope, tau)
        else:
            if isinstance(interval, Number):
                if interval < 1:
                    raise ValueError("Invalid update interval {} specified".format(interval))

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

    # this ensures that we always return a tf.Operation
    with tf.control_dependencies([update_target_vars]):
        return tf.no_op()


def td_update(reward, terminal, future_value, discount, name=None) -> tf.Tensor:
    """
    Calculates the TD update for a batch of data. This is `reward` for terminal states,
    and `reward + discount * future_value` for non-terminals.
    :param reward: A vector. The reward at the corresponding time steps.
    :param terminal: A vector. Whether the states were terminal.
    :param future_value: A vector or matrix. The value predicted at the following time step. Use a
                         vector if the future values are only scalars, and a matrix of shape [BATCH, ACTION_DIMS]
                         otherwise.
    :param discount: A scalar. Factor used for discounting `future_value`.
    :param name: Name of the op.
    :return: The new value according to the TD update. Has the same shape as `future_value`.
    """
    with tf.name_scope(name, default_name="td_update", values=[reward, terminal, future_value, discount]):
        # static args check
        if isinstance(discount, Number) and not (0.0 <= discount <= 1.0):
            raise ValueError("discount factor {} not in [0, 1]".format(discount))

        # args check
        future_value = tf.convert_to_tensor(future_value)  # type: tf.Tensor
        if not future_value.dtype.is_floating:
            raise TypeError("value function {} is not of floating point type".format(future_value))

        reward = tf.convert_to_tensor(reward, dtype=future_value.dtype)
        terminal = tf.convert_to_tensor(terminal, dtype=tf.bool)
        discount = tf.convert_to_tensor(discount, dtype=future_value.dtype)

        discount.shape.assert_has_rank(0)
        terminal.shape.assert_has_rank(1)

        if not terminal.shape.is_compatible_with(reward.shape):
            raise ValueError("reward {} and terminal {} have incompatible shapes".format(reward, terminal))

        with tf.control_dependencies([tf.assert_non_negative(discount)]):
            future = tf.where(terminal, x=tf.zeros_like(future_value), y=future_value, name="select_nonterminal")  # type: tf.Tensor

            # OK, now we have to be careful about dimensions. If we act on vector future_value, reward has to be a
            # vector, otherwise we need to match ranks
            if future.shape.ndims == 1:
                pass
            elif future.shape.ndims == 2:
                reward = tf.expand_dims(reward, 1)
            else:
                raise ValueError("Invalid rank of future_value {}".format(future_value))
            return reward + discount * future
