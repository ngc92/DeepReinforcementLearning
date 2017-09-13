from collections import namedtuple
from numbers import Integral, Number
from typing import Tuple, Union, List
import tensorflow as tf
import numpy as np


QSample = namedtuple('QSample', ['current', 'action', "reward", "next", "terminal"])
TensorLike = Union[tf.Tensor, np.ndarray, Number, List[Number]]


def _verify_shape(shape):
    for s in shape:
        if s <= 0:
            raise ValueError("Invalid shape {}!".format(shape))
    return tuple(shape)


def _test_static_shape(tensor, required_shape):
    static_shape = tensor.shape
    for (s, r) in zip(static_shape, required_shape):
        if s is not None and s != r:
            raise ValueError("Shape mismatch of {}, expected {} but got {}.".format(tensor,
                                                                                    required_shape, static_shape))


class Memory:
    """
    This class manages a ReplayMemory completely in tensorflow variables.
    >>> sess = tf.InteractiveSession()
    >>> memory = Memory(100, (10,), (1,), tf.int32)
    >>> append_op = memory.append(np.zeros(10), 5, 1.5, False, np.ones(10))
    >>> sample_op = memory.sample(10)
    >>> sess.run(append_op)
    >>> sess.run(sample_op)
    """
    def __init__(self, size: int, state_shape: Tuple[int, ...], action_shape: Tuple[int, ...],
                 action_type: tf.DType=tf.int64, scope=None):
        """
        Create a new ReplayMemory.
        :param size: Number of slots in the replay memory.
        :param state_shape: Shape of the observation.
        :param action_shape: Shape of the action.
        :param action_type: Either tf.float32 for continuous actions, or tf.int32/64 for discrete actions.
                            Mixing is currently not supported.
        :param scope: Name scope in which to place the variables.
        """

        # check argument validity
        if size < 1:
            raise ValueError("Cannot specify negative size for ReplayMemory. Got {}".format(size))

        action_type = tf.as_dtype(action_type)
        if action_type not in {tf.float32, tf.int64, tf.int32}:
            raise ValueError("Unsupported action type {}. Use tf.float32 or tf.int32/64!".format(action_type))

        self._size = size
        self._state_shape = _verify_shape(state_shape)
        self._action_shape = _verify_shape(action_shape)
        self._action_type = action_type

        # cached tf variables
        self._states_v = None
        self._actions_v = None
        self._rewards_v = None
        self._next_v = None
        self._term_v = None
        self._write_pointer_v = None
        self._total_v = None

        self._length = None

        self._scope = scope
        self._has_vars = False

    def _prepare_variables(self):
        """
        Create all Variables that are needed for the replay memory.
        Does nothing if these variables already exist.
        :return: None
        """

        # skip if we already have the vars.
        if self._has_vars:
            return

        # otherwise, build them in our scope
        with tf.variable_scope(self._scope, "ReplayMemory", reuse=False) as scope, tf.device("/cpu:0"):
            ms = (self._size, )
            state_shape = ms + self._state_shape
            action_shape = ms + self._action_shape

            kwargs = {"initializer": tf.zeros_initializer(), "trainable": False, "collections": ["REPLAY_MEMORY",
                                                                                                 tf.GraphKeys.GLOBAL_VARIABLES]}

            # get the data varaibles
            self._states_v = tf.get_variable("observation_t", shape=state_shape, dtype=tf.float32, **kwargs)
            self._actions_v = tf.get_variable("action_t", shape=action_shape, dtype=self._action_type, **kwargs)
            self._rewards_v = tf.get_variable("rewards_t", shape=ms, dtype=tf.float32, **kwargs)
            self._next_v = tf.get_variable("observation_tp1", shape=state_shape, dtype=tf.float32, **kwargs)
            self._term_v = tf.get_variable("term_t", shape=ms, dtype=tf.bool, **kwargs)

            # get the control variables
            self._write_pointer_v = tf.get_variable("write_pointer", shape=(), dtype=tf.int32, **kwargs)
            self._total_v = tf.get_variable("total_size", shape=(), dtype=tf.int32, **kwargs)

            self._length = tf.minimum(self._size, self._total_v)

            self._scope = scope
            self._has_vars = True

    def append(self, state: TensorLike, action: TensorLike, reward: TensorLike, terminal: TensorLike,
               next_state: TensorLike, name=None):
        """
        Append a new (state, action, reward, state) tuple to the replay memory.
        :return: The update operation.
        """

        # ensure the variables exists
        self._prepare_variables()

        state = tf.convert_to_tensor(state, tf.float32, name="state")
        action = tf.convert_to_tensor(action, self._action_type, name="action")
        reward = tf.convert_to_tensor(reward, tf.float32, name="reward")
        terminal = tf.convert_to_tensor(terminal, tf.bool, name="terminal")
        next_state = tf.convert_to_tensor(next_state, tf.float32, name="state")

        _test_static_shape(state, self._state_shape)
        _test_static_shape(action, self._action_shape)
        _test_static_shape(reward, ())
        _test_static_shape(terminal, ())
        _test_static_shape(next_state, self._state_shape)

        # put all ops in a name scope, and verify the input graphs
        with tf.name_scope(name, "append", (state, action, reward, terminal, next_state)), tf.device("/cpu:0"):
            return self._append(state, action, reward, terminal, next_state)

    def _append(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, terminal: tf.Tensor,
                next_state: tf.Tensor):
        """
        Do the actual appending, after all checks have passend and scopes are set
        """
        # perform the simple variable updates. For simplicity we update next_state even
        # if the state is terminal.
        i = self._write_pointer_v  # type: tf.Variable
        state_update = tf.scatter_update(self._states_v, i, state, name="update_states")
        actions_update = tf.scatter_update(self._actions_v, i, action, name="update_actions")
        rewards_update = tf.scatter_update(self._rewards_v, i, reward, name="update_rewards")
        terminal_update = tf.scatter_update(self._term_v, i, terminal, name="update_terminals")
        next_update = tf.scatter_update(self._next_v, i, next_state, name="update_next_states")

        update_vars = tf.group(state_update, actions_update, rewards_update, terminal_update, next_update,
                               name="update_variables")

        with tf.name_scope("increment_counters"), tf.control_dependencies([update_vars]):
            inc_i = i.assign_add(1)
            inc_t = self._total_v.assign_add(1)

            with tf.control_dependencies([inc_i]):
                inc_i = tf.cond(tf.equal(i, self._size), true_fn=lambda: i.assign(0), false_fn=lambda: i)

            # now, this op depends on everything that came before
            return tf.group(inc_i, inc_t)

    def sample(self, amount: Union[Integral, tf.Tensor], name=None) -> QSample:
        """
        Sample `amount` many state transitions from the replay memory. Samples are chosen uniformly
        from the available data in the ReplayMemory.
        :param amount: How many transitions to sample.
        :param name: Name of the operation.
        :return: A QSample object containing the tensors with the sampled values.
        """
        # ensure the variables exist
        self._prepare_variables()

        # if we got a constant amount, we can statically check validity
        if isinstance(amount, Integral):
            if amount <= 0:
                raise ValueError("Sampling amount has to be a positive integer, got {}".format(amount))

        with tf.name_scope(name, "sample_replay_memory"), tf.device("/cpu:0"):
            amount_tensor = tf.convert_to_tensor(amount, dtype=tf.int64)

            assert_nonempty = tf.assert_greater(self.length, 0, name="assert_non_empty",
                                                message="Trying to sample from empty ReplayMemory")
            assert_positive = tf.assert_positive(amount_tensor, message="Sampling amount has to be positive.",
                                                 name="assert_positive")

            if amount_tensor.shape != ():
                raise ValueError("Sampling amount has to be am integral scalar, got {} instead.".format(amount))

            # after all the checks, do the actual sampling
            with tf.control_dependencies([assert_nonempty, assert_positive]):
                return self._sample(amount_tensor)

    def _sample(self, amount: tf.Tensor):
        """
        This function implements the actual sampling logic.
        """
        indices = tf.random_uniform((amount,), 0, self.length, dtype=tf.int32, name="random_sampling_indices")
        state = tf.gather(self._states_v, indices, name="gather_state")
        action = tf.gather(self._actions_v, indices, name="gather_action")
        reward = tf.gather(self._rewards_v, indices, name="gather_reward")
        terminal = tf.gather(self._term_v, indices, name="gather_terminal")
        next_state = tf.gather(self._next_v, indices, name="gather_next_state")

        return QSample(state, action, reward, next_state, terminal)

    @property
    def length(self) -> tf.Tensor:
        """
        :return: The "length" of the memory, i.e. the number of elements that are currently saved
                 inside it.
        """
        self._prepare_variables()
        return self._length

    @property
    def capacity(self):
        """
        :return: The maximum capacity of this ReplayMemory.
        """
        return self._size
