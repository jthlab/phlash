import numpy as np
import pytest

import phlash
import phlash.sim
from phlash.data import RawContig
from phlash.size_history import DemographicModel


@pytest.mark.slow
def test_functional1():
    sim = phlash.sim.stdpopsim_dataset(
        "HomSap", "Zigzag_1S14", {"generic": 20}, options={"length_multiplier": 0.01}
    )
    res = phlash.fit(list(sim["data"].values()), niter=5, num_particles=123)
    assert isinstance(res, list)
    assert len(res) == 123
    assert isinstance(res[0], DemographicModel)


def test_functional2():
    het = np.array([[0, 1, 0, 1, 1]], dtype=np.int8)
    afs = np.array([1])
    ctg = RawContig(het, afs, 100)
    res = phlash.fit([ctg], niter=2, num_particles=5, chunk_size=1, overlap=1)
    assert isinstance(res, list)
    assert len(res) == 5
    assert isinstance(res[0], DemographicModel)


def test_psmc(psmcfa_file):
    phlash.psmc([psmcfa_file] * 3, niter=2, num_particles=5, chunk_size=1, overlap=1)
