from collections import namedtuple
from numbers import Real, Integral
from typing import Callable, List, Optional, Dict, Any, Union, Iterable
import copy

import numpy as np
import tensorflow as tf

from .helpers.scoping import copy_variables_to_scope
from .helpers.utils import add_replay_memory
from .helpers.ops import choose_from_array, epsilon_greedy, target_net_update
from .builder import AgentBuilder, AgentActSpec, AgentTrainSpec

# mapping state -> features
FeaturesFunc = Callable[[tf.Tensor], tf.Tensor]
QFunc = Callable[[tf.Tensor], List[tf.Tensor]]

DQNConfig = namedtuple("DQNConfig", ["double_q", "soft_target", "optimizer"])


def make_q_fn(features_fn: FeaturesFunc, num_discrete_actions: Iterable[Integral]) -> QFunc:
    if not callable(features_fn):
        raise TypeError("features_fn {} is not callable!".format(features_fn))

    flat_actions = np.array(num_discrete_actions).flatten()
    partial_sums = np.cumsum(flat_actions)
    total_actions = flat_actions.sum()

    def q_fn(state: tf.Tensor) -> List[tf.Tensor]:
        features = features_fn(state)
        q_values = tf.layers.dense(features, total_actions, name="all_q_linear")
        last = 0
        split = []
        for i in range(len(flat_actions)):
            split.append(q_values[:, last:partial_sums[i]])
            last = partial_sums[i]
        return split

    return q_fn


def scalar_param(name: str, dict_: dict, default):
    value = tf.convert_to_tensor(dict_.get(name, default))
    value.shape.assert_has_rank(0)
    return value


@add_replay_memory(64, 100000)
class DQNBuilder(AgentBuilder):
    def __init__(self, feature_func, num_actions, config):
        super(DQNBuilder, self).__init__()
        self._q_function = make_q_fn(feature_func, num_actions)
        self._target_var_scope = None
        self._network_var_scope = None
        self.config = copy.deepcopy(config)

    def _prepare_scopes(self):
        with tf.variable_scope("target_variables") as scope:
            self._target_var_scope = scope

        with tf.variable_scope("deepq") as scope:
            self._network_var_scope = scope

    def _build_q(self, state: tf.Tensor) -> List[tf.Tensor]:
        with tf.variable_scope("q_values"):
            return self._q_function(state)

    def act_fn(self, state: tf.Tensor, epsilon: Optional[tf.Tensor], scope, reuse=None) -> tf.Tensor:
        """
        Calculates the actions following greedy or epsilon-greedy action selection, given the
        Q values calculate from `state`. This method expects `state` to be presented in batch form.
        :param state: Current states. First dimension is batch dimension.
        :param epsilon: Epsilon-Greedy parameter, a scalar tensor, or None for greedy action selection.
        :param scope: Variable scope in which the q function is calculated.
        :param reuse: Whether the VariableScope is in reuse mode.
        :return tf.Tensor: The actions corresponding to the states.
        """
        # ensure that we have a batch dimension
        state.shape.with_rank_at_least(2)

        with tf.variable_scope(scope, reuse=reuse):
            return self._act_fn(state, epsilon)

    def _act_fn(self, state: tf.Tensor, epsilon: Optional[tf.Tensor]):
        # epsilon greedy action selection
        split_q = self._build_q(state)
        if epsilon is None:
            actions = [tf.argmax(q, axis=1, name="greedy_action") for q in split_q]
        else:
            actions = [epsilon_greedy(q, epsilon, True, dtype=tf.int64) for q in split_q]
        actions = tf.stack(actions, axis=1, name="concat_actions")
        return actions

    def assess_actions(self, state: tf.Tensor, actions: tf.Tensor, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            split_q = self._build_q(state)
            q_values = [choose_from_array(q, actions[:, i]) for i, q in enumerate(split_q)]
            q_values = tf.stack(q_values, axis=1)
            return q_values

    def assess_state(self, state: tf.Tensor, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            split_q = self._build_q(state)
            q_values = [tf.reduce_max(q, axis=1) for q in split_q]
            q_values = tf.stack(q_values, axis=1)
            return q_values

    ####################################################################################################################
    def _build_act(self, observation, params):
        self._prepare_scopes()

        # add a fake batch dimension
        state_t = tf.expand_dims(observation, 0)
        action = self.act_fn(state_t, None, scope=self._network_var_scope)
        action.shape.is_compatible_with(tf.TensorShape([1, None]))
        action = tf.squeeze(action, axis=0)
        return AgentActSpec(actions=action, metrics={}, is_exploring=False)

    def _build_explore(self, observation, params):
        self._prepare_scopes()

        # add a fake batch dimension
        state_t = tf.expand_dims(observation, 0)
        action = self.act_fn(state_t, None, scope=self._network_var_scope)
        action.shape.is_compatible_with(tf.TensorShape([1, None]))
        action = tf.squeeze(action, axis=0)
        return AgentActSpec(actions=action, metrics={}, is_exploring=True)

    def _build_train(self, transition: Dict[str, tf.Tensor], params):
        self._prepare_scopes()

        state_t = transition["observation"]  # shape: (Batch, ObservationShape)
        state_tp1 = transition["next_observation"]  # shape: (Batch, ObservationShape)
        action_t = transition["action"]  # shape: (Batch, ActionShape)
        reward_t = transition["reward"]  # shape: (Batch)
        terminal_t = transition["terminal"]  # shape: (Batch)

        # get relevant data from params
        gamma = scalar_param("gamma", params, 0.99)

        # calculate the Q values using the current weights
        q_t = self.assess_actions(state_t, action_t, scope=self._network_var_scope, reuse=True)

        # prepare target network
        target_scope, target_vars = copy_variables_to_scope(self._network_var_scope, self._target_var_scope,
                                                            trainable=False)

        if self.config.double_q:
            raise NotImplementedError()
        else:
            q_tp1 = self.assess_state(state_tp1, target_scope, reuse=True)

        return_t = reward_t + \
                   gamma * tf.where(terminal_t, x=tf.zeros_like(q_tp1), y=q_tp1, name="select_nonterminal")

        return_t = tf.stop_gradient(return_t, "return_t")  # type: tf.Tensor

        return_t.shape.assert_is_compatible_with(q_t.shape)

        loss = tf.losses.huber_loss(q_t, tf.stop_gradient(return_t), reduction=tf.losses.Reduction.MEAN)

        optimizer = self.config.optimizer  # type: tf.train.Optimizer
        train_step = optimizer.minimize(loss, tf.train.get_global_step(tf.get_default_graph()))

        if self.config.soft_target:
            tau = scalar_param("tau", params, 1e-3)
            interval = None
        else:
            interval = scalar_param("update_interval", params, 1000)
            tau = None

        with tf.control_dependencies([train_step]):
            update_target_vars = target_net_update(self._network_var_scope, target_scope, tau, interval)

        return AgentTrainSpec(loss=loss, train_op=update_target_vars, metrics={})
