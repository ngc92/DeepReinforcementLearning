from collections import namedtuple


AgentActSpec = namedtuple("AgentActSpace", ["actions", "metrics", "is_exploring"])
AgentTrainSpec = namedtuple("AgentTrainSpec", ["loss", "train_op", "metrics"])


class AgentBuilder:
    def __init__(self):
        pass

    def act(self, observation, params) -> AgentActSpec:
        actor = self._build_act(observation, params)
        if not isinstance(actor, AgentActSpec):
            raise TypeError("_act is supposed to return AgentActSpec")

        if actor.actions is None:
            raise ValueError("No actions specified in AgentActSpec")

        return actor

    def explore(self, observation, params) -> AgentActSpec:
        actor = self._build_explore(observation, params)
        if not isinstance(actor, AgentActSpec):
            raise TypeError("_explore is supposed to return AgentActSpec")

        if actor.actions is None:
            raise ValueError("No actions specified in AgentActSpec")

        return actor

    def train(self, transition, params) -> AgentTrainSpec:
        trainer = self._build_train(transition, params)
        if not isinstance(trainer, AgentTrainSpec):
            raise TypeError("_train is supposed to return AgentTrainSpec")

        if trainer.train_op is None:
            raise ValueError("No train_op specified in AgentTrainSpec")

        return trainer

    # functions to be implemented by derived classes
    def _build_act(self, observation, params):
        raise NotImplementedError()

    def _build_explore(self, observation, params):
        raise NotImplementedError()

    def _build_train(self, transition, params):
        raise NotImplementedError()
