import pytest
from bang.core.PBN import PBN
import numpy as np

def test_fixpoint_async_step():
    pbn1 = PBN(2,[1,1],[2,2],[[True, True, True, False],[True, False, True, True]], [[0,1],[0,1]],[[1.],[1.]],0.,[2],n_parallel=2, update_type="asynchronous_random_order")
    pbn1.set_states([[True, False], [False, False]])

    pbn1.simple_steps_cpu(101)

    assert np.array_equal([[1], [3]], pbn1.latest_state), pbn1.latest_state


def test_fixpoint_sync_step():
    pbn1 = PBN(2, [1,1],[2,2],[[True, True, True, False],[True, False, True, True]], [[0,1],[0,1]],[[1.],[1.]],0.,[2],n_parallel=2, update_type="synchronous")
    pbn1.set_states([[True, False], [True, False]])

    pbn1.simple_steps_cpu(21)

    assert np.array_equal([[1], [1]], pbn1.latest_state), pbn1.latest_state


@pytest.mark.parametrize("n_parallel", [16, 32, 64, 128, 256, 512])
def test_large_n_parallel_sync(n_parallel):
    pbn1 = PBN(2, [1,1],[2,2],[[True, True, True, False],[True, False, True, True]], [[0,1],[0,1]],[[1.],[1.]],0.,[2], n_parallel, update_type="synchronous")
    pbn1.set_states([[False, False] for _ in range(n_parallel)])

    pbn1.simple_steps_cpu(21)

    assert np.array_equal([[3] for _ in range(n_parallel)], pbn1.latest_state), pbn1.latest_state

@pytest.mark.parametrize("n_parallel", [16, 32, 64, 128, 256, 512])
def test_large_n_parallel_async(n_parallel):
    pbn1 = PBN(2, [1,1],[2,2],[[True, True, True, False],[True, False, True, True]], [[0,1],[0,1]],[[1.],[1.]],0.,[2], n_parallel, update_type="asynchronous_random_order")
    pbn1.set_states([[False, False] for _ in range(n_parallel)])

    pbn1.simple_steps_cpu(21)

    assert np.array_equal([[3] for _ in range(n_parallel)], pbn1.latest_state), pbn1.latest_state