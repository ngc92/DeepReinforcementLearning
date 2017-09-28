import functools

import tensorflow as tf


def clipping_optimizer(optimizer: tf.train.Optimizer, clip_fn):
    class ClippingOptimizer(tf.train.Optimizer):
        def __init__(self, base_optimizer: tf.train.Optimizer, clipper):
            super(ClippingOptimizer, self).__init__(base_optimizer._use_locking, "ClippingOptimizer")
            self._base_optimizer = base_optimizer
            self._clipper = clipper

        def __getattr__(self, item):
            return getattr(self._base_optimizer, item)

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            # clipping
            with tf.name_scope(name+"/gradient_clipping"):
                for i in range(len(grads_and_vars)):
                    g, v = grads_and_vars[i]
                    if g is not None:
                        grads_and_vars[i] = (self._clipper(g), v)

            return self._base_optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                                                        name=name)

    return ClippingOptimizer(optimizer, clip_fn)


def in_new_graph(original):
    @functools.wraps(original)
    def in_session(**kwargs):
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as session:
                original(**kwargs)

    return in_session


def add_replay_memory(batch_size=None, memory_size=None):
    def decorator(builder_class: type):
        from tfdeeprl.helpers import Memory
        from tfdeeprl.builder import AgentBuilder

        if not issubclass(builder_class, AgentBuilder):
            raise TypeError("Expected subclass of AgentBuilder, got {}".format(builder_class))

        default_memory_size = memory_size
        default_batch_size = batch_size

        # noinspection PyProtectedMember
        old_build_train = builder_class._build_train

        @functools.wraps(old_build_train)
        def build_train(self: builder_class, transitions, params):
            state = transitions["observation"]
            action = transitions["action"]
            reward = transitions["reward"]
            next = transitions["next_observation"]
            terminal = transitions["terminal"]

            memory_size = params.get("memory_size", default_memory_size)
            batch_size = params.get("batch_size", default_batch_size)

            memory = Memory(memory_size, tuple(state.shape.as_list()), tuple(action.shape.as_list()), action.dtype)
            append_op = memory.append(state=state, action=action, reward=reward,
                                      terminal=terminal, next_state=next)

            # ensure they are synchronized
            with tf.control_dependencies([append_op]):
                sample_op = memory.sample(batch_size)

            sampled_transition = {
                "observation": sample_op.current,
                "next_observation": sample_op.next,
                "action": sample_op.action,
                "reward": sample_op.reward,
                "terminal": sample_op.terminal
            }

            return old_build_train(self, transition=sampled_transition, params=params)

        builder_class._build_train = build_train

        return builder_class

    return decorator


def linear_schedule(initial_value, final_value, start_step, end_step, name=None):
    """
    Returns a function that maps a step tensor to a linear interpolation between
    `initial_value` and `final_value`. Before `start_step` and after `end_step`
    the value is clipped between `initial_value` and `final_value` respectively.
    :param initial_value: The value to return before `start_step`.
    :param final_value: The value to return after `end_step`.
    :param start_step: The first step at which the value starts interpolating.
    :param end_step: The lest step of the interpolation.
    :param name: Name for the op. Defaults to `linear_schedule`.
    :return: A callable that performs the interpolation.
    """
    def schedule(global_step: tf.Tensor) -> tf.Tensor:
        with tf.name_scope(name, default_name="linear_schedule", values=[global_step, initial_value, final_value,
                                                                         start_step, end_step]):
            # convert things to tensors
            global_step = tf.convert_to_tensor(global_step, name="global_step")
            start_step_t = tf.convert_to_tensor(start_step, name="start_step", dtype=global_step.dtype)
            end_step_t = tf.convert_to_tensor(end_step, name="end_step", dtype=global_step.dtype)
            initial_value_t = tf.convert_to_tensor(initial_value, name="initial_value")
            final_value_t = tf.convert_to_tensor(final_value, name="final_value")

            duration = tf.cast(end_step_t - start_step_t, final_value_t.dtype)
            relative = tf.cast(global_step - start_step_t, final_value_t.dtype) / duration
            relative = tf.clip_by_value(relative, 0.0, 1.0)

            return initial_value_t + (final_value_t - initial_value_t) * relative

    return schedule
