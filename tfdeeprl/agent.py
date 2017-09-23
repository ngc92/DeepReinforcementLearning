import copy
import tempfile

import gym
import gym.spaces
import tensorflow as tf

from .builder import AgentBuilder
from .helpers.rollout_ops import rollout


class Agent:
    """
    This is a much simplified version of the tf.estimator.Estimator for reinforcement
    learning gym environments.
    """
    def __init__(self, builder: AgentBuilder, model_dir=None, params=None):
        self._model_dir = model_dir
        if self._model_dir is None:
            self._model_dir = tempfile.mkdtemp()
            tf.logging.warning('Using temporary folder as model directory: %s',
                               self._model_dir)

        if not isinstance(builder, AgentBuilder):
            raise TypeError('builder must be an AgentBuilder, got {}.'.format(builder))

        self._builder = builder
        self._params = copy.deepcopy(params) or {}

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def params(self):
        return copy.deepcopy(self._params)

    def build_training_graph(self, env: gym.Env):
        with tf.Graph().as_default() as g:
            def explore_fn(observation):
                act_spec = self._builder.explore(observation, params=self.params)
                return act_spec.actions

            def record_fn(state, action, reward, terminal, state_tp1):
                data = {"observation": state,
                        "action": action,
                        "reward": reward,
                        "terminal": terminal,
                        "next_observation": state_tp1
                        }
                spec = self._builder.train(data, params=self.params)
                return spec.train_op

            rollout_op = rollout(env, explore_fn, record_fn)

        result = {"return": rollout_op[0], "duration": rollout_op[1]}
        return result, g

    def train(self, env: gym.Env, max_episodes: int):
        returns = []
        durations = []
        result_op, g = self.build_training_graph(env)

        # TODO make this frequency configurable
        episode_logger = tf.train.LoggingTensorHook(result_op, 10)
        checkpoint_saver = tf.train.CheckpointSaverHook(checkpoint_dir=self.model_dir, save_secs=60)
        with g.as_default():
            with tf.train.SingularMonitoredSession(checkpoint_dir=self.model_dir,
                                                   hooks=[episode_logger, checkpoint_saver]) as session:
                for i in range(max_episodes):
                    # get action
                    result_val = session.run(result_op)

                    returns.append(result_val["return"])
                    durations.append(result_val["duration"])
            return returns, durations
