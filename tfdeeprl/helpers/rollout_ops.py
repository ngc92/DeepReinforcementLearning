from typing import Optional

import gym
import gym.spaces
import tensorflow as tf
import numpy as np
from collections import namedtuple


GymStepResult = namedtuple("GymStepResult", ("observation", "reward", "terminal"))
GymLoopData = namedtuple("GymLoopData", ("observation", "reward", "terminal", "step_count"))


def rollout(env: gym.Env, policy_fn, record_fn: Optional = None):
    state = gym_reset(env)
    reward = tf.zeros((), name="reward")
    done = tf.zeros((), dtype=tf.bool)

    loop_data = GymLoopData(state, reward, done, tf.constant(0, dtype=tf.int64))

    def cond(obs, rew, done, count):
        return tf.logical_not(done)

    def step(obs, rew, done, count):
        action = policy_fn(obs)
        result = gym_step(env, action)
        deps = []
        # the recording gets a control dependency on
        with tf.control_dependencies([rew]):
            if record_fn:
                record_op = record_fn(obs, action, result.reward, result.terminal, result.observation)
                deps.append(record_op)

        with tf.control_dependencies(deps):
            return GymLoopData(result.observation, rew + result.reward, result.terminal, count + 1)

    _, reward, _, count = tf.while_loop(cond=cond, body=step, loop_vars=loop_data,
                                        back_prop=False, parallel_iterations=10)
    return reward, count


def gym_reset(env: gym.Env) -> tf.Tensor:
    def reset():
        return np.array(env.reset()).astype(np.float32)

    reset_op = tf.py_func(reset, [], tf.float32, True, name="reset")  # type: tf.Tensor
    reset_op.set_shape(env.observation_space.shape)
    return reset_op


def gym_step(env: gym.Env, action: tf.Tensor):
    # check type compatibility
    action_space = env.action_space
    observation_space = env.observation_space
    if isinstance(action_space, gym.spaces.Box):
        if not action.dtype.is_floating:
            raise TypeError("Expected a floating point action, got {}".format(action))

        action.shape.assert_is_compatible_with(tf.TensorShape(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        if not action.dtype.is_integer:
            raise TypeError("Expected a discrete action, got {}".format(action))
        # TODO treat scalars as 1 element vectors
        action.shape.assert_is_compatible_with([1])

    def step(action):
        obs, rew, done, info = env.step(action)
        return np.array(obs).astype(np.float32), np.float32(rew), bool(done)

    result = tf.py_func(step, [action], [tf.float32, tf.float32, tf.bool], True, "step_env")
    observation = result[0]  # type: tf.Tensor
    reward = result[1]       # type: tf.Tensor
    terminal = result[2]     # type: tf.Tensor

    if isinstance(observation_space, gym.spaces.Box):
        observation.set_shape(observation_space.shape)
    else:
        raise NotImplementedError()

    reward.set_shape(())
    terminal.set_shape(())

    return GymStepResult(observation, reward, terminal)
