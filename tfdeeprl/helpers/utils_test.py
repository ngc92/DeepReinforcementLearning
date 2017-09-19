import pytest
from .utils import *


@in_new_graph
def test_linear_schedule():
    sched = linear_schedule(1.0, 0.0, 0, 1000)

    assert sched(100).eval() == pytest.approx(0.9)
    assert sched(500).eval() == pytest.approx(0.5)
    assert sched(1500).eval() == pytest.approx(0.0)

    sched = linear_schedule(1.0, 0.0, 1000, 2000)
    assert sched(100).eval() == pytest.approx(1.0)
    assert sched(1100).eval() == pytest.approx(0.9)
    assert sched(1500).eval() == pytest.approx(0.5)

    sched = linear_schedule(0.7, 0.2, 0, 1000)
    assert sched(-100).eval() == pytest.approx(0.7)
    assert sched(500).eval() == pytest.approx((0.7 + 0.2) / 2)
    assert sched(1100).eval() == pytest.approx(0.2)
