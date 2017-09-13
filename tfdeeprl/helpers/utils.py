import functools
from typing import Dict, Any, Optional

import tensorflow as tf

from tfdeeprl.agent import AgentModes
from tfdeeprl.helpers.replay_memory import Memory


def choose_from_array(source, indices, name="choose_from_array"):
    """ returns [source[i, indices[i]] for i in 1:len(indices)] """
    with tf.name_scope(name):
        num_samples = tf.shape(indices)[0]
        indices     = tf.transpose(tf.stack([tf.cast(tf.range(0, num_samples), indices.dtype), indices]))
        values      = tf.gather_nd(source, indices)
    return values


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
    def in_session():
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as session:
                original()

    return in_session


def add_replay_memory_training(base_fn, memory_size, batch_size):
    def model_fn(data, mode, params: Dict[str, Any]):
        if mode == AgentModes.TRAIN:
            state = data["observation"]
            action = data["action"]
            reward = data["reward"]
            next = data["next_observation"]
            terminal = data["terminal"]

            print("REWARD:")
            print(reward)

            memory = Memory(memory_size, tuple(state.shape.as_list()), tuple(action.shape.as_list()), action.dtype)
            append_op = memory.append(state=state, action=action, reward=reward,
                                      terminal=terminal, next_state=next)

            # ensure they are synchronized
            with tf.control_dependencies([append_op]):
                sample_op = memory.sample(batch_size)

            data["observation"] = sample_op.current
            data["next_observation"] = sample_op.next
            data["action"] = sample_op.action
            data["reward"] = sample_op.reward
            data["terminal"] = sample_op.terminal

        return base_fn(data, mode, params=params)

    return model_fn


def epsilon_greedy(values: tf.Tensor, epsilon: tf.Tensor, stochastic: tf.Tensor, name=None):
    assert len(values.shape) == 2
    with tf.name_scope(name, default_name="epsilon_greedy", values=[values, epsilon, stochastic]):
        with tf.name_scope("batch_size"):
            batch_size = tf.shape(values)[0]

        greedy = tf.cast(tf.argmax(values, axis=1), tf.int32)

        def eps_greedy():
            do_greedy = tf.random_uniform((batch_size,), 0.0, 1.0) < epsilon
            random = tf.random_uniform((batch_size,), minval=0, maxval=values.shape[1], dtype=tf.int32)
            action = tf.where(condition=do_greedy,
                              x=greedy,
                              y=random)
            return action
        return tf.cond(stochastic, eps_greedy, lambda: tf.identity(greedy))
