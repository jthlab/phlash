import phlash.mcmc
import phlash.sim
from phlash.size_history import DemographicModel


def test_functional1():
    truth, chd = phlash.sim.stdpopsim_dataset("HomSap", "Zigzag_1S14", {"generic": 20})
    res = phlash.mcmc.fit(chd, options=dict(niter=5))
    assert isinstance(res, list)
    assert len(res) == 500
    assert isinstance(res[0], DemographicModel)
