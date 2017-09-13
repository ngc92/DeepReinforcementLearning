import pytest
import numpy as np

from .ops import *
from .utils import in_new_graph


@in_new_graph
def test_choose_from_array():
    source = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    choice = choose_from_array(source, [0, 1, 2, 1])

    assert list(choice.eval()) == [1, 5, 9, 11]


@in_new_graph
def test_epsilon_greedy_verification():
    epsilon_greedy(np.ones((4, 4)), 1, True)

    with pytest.raises(ValueError):
        epsilon_greedy([1, 2, 3], 0.5, False)

    with pytest.raises(ValueError):
        epsilon_greedy(np.ones((4, 4)), [1, 3], False)

    with pytest.raises(ValueError):
        epsilon_greedy(np.ones((4, 4)), 1.0, [False, True])

    with pytest.raises(TypeError):
        epsilon_greedy(np.ones((4, 4)), 1.0, 1)

    with pytest.raises(TypeError):
        epsilon_greedy(np.ones((4, 4)), "1.0", 1)


@in_new_graph
def test_epsilon_greedy_greedyness():
    eg = epsilon_greedy([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], 0.0, True)
    assert list(eg.eval()) == [1, 3, 0, 1]

    eg = epsilon_greedy([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], 1.0, False)
    assert list(eg.eval()) == [1, 3, 0, 1]

# TODO test the random part of E-Greedy
