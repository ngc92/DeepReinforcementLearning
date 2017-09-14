import abc
from collections import namedtuple
from typing import Optional, Dict
import tensorflow as tf

AgentTrainSpec_ = namedtuple("AgentTrainSpec_", ["loss", "train_op", "metrics"])
AgentActSpec_ = namedtuple("AgentActSpec_", ["actions", "metrics", "is_exploring"])


class AgentActSpec(AgentActSpec_):
    def __new__(cls, *, actions, metrics, is_exploring):
        if metrics is None:
            metrics = {}
        if not isinstance(actions, tf.Tensor):
            raise TypeError("actions has to be a Tensor, got {}".format(actions))
        if not isinstance(is_exploring, bool):
            raise TypeError("is_exploring has to be a boolean, got {}".format(is_exploring))
        if not isinstance(metrics, Dict):
            raise TypeError("metrics have to be a dict, got {}".format(metrics))

        new = super(AgentActSpec, cls).__new__(cls, actions=actions, metrics=metrics, is_exploring=is_exploring)
        return new


class AgentTrainSpec(AgentTrainSpec_):
    def __new__(cls, *, train_op, metrics=None, loss=None):
        if metrics is None:
            metrics = {}
        if loss is not None:
            if not isinstance(loss, tf.Tensor):
                raise TypeError("loss has to be a Tensor, got {}".format(loss))
            loss.shape.assert_has_rank(0)
            if not loss.dtype.is_floating:
                raise TypeError("loss has to be a real scalar tensor, got {}".format(loss))
        if train_op is not None and not isinstance(train_op, tf.Operation):
            raise TypeError("train_op has to be an Operation, got {}".format(train_op))
        if not isinstance(metrics, Dict):
            raise TypeError("metrics have to be a dict, got {}".format(metrics))

        new = super(AgentTrainSpec, cls).__new__(cls, loss=loss, train_op=train_op, metrics=metrics)
        return new


class AgentBuilder:
    def __init__(self):
        pass

    def act(self, observation, params: Optional) -> AgentActSpec:
        actor = self._build_act(observation, params=params)
        if not isinstance(actor, AgentActSpec):
            raise TypeError("_build_act is supposed to return AgentActSpec, got {}".format(actor))

        if actor.is_exploring:
            raise ValueError("Exploration enabled in act mode.")

        return actor

    def explore(self, observation, params: Optional) -> AgentActSpec:
        actor = self._build_explore(observation, params=params)
        if not isinstance(actor, AgentActSpec):
            raise TypeError("_explore is supposed to return AgentActSpec, got {}".format(actor))

        if actor.actions is None:
            raise ValueError("No actions specified in AgentActSpec")
        if not actor.is_exploring:
            raise ValueError("Exploration disabled in explore mode.")
        return actor

    def train(self, transition, params: Optional) -> AgentTrainSpec:
        trainer = self._build_train(transition, params=params)
        if not isinstance(trainer, AgentTrainSpec):
            raise TypeError("_train is supposed to return AgentTrainSpec")

        if trainer.train_op is None:
            raise ValueError("No train_op specified in AgentTrainSpec")

        return trainer

    # functions to be implemented by derived classes
    @abc.abstractmethod
    def _build_act(self, observation, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_explore(self, observation, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_train(self, transition, params):
        raise NotImplementedError()
