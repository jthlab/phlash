import phlash.mcmc
import phlash.sim
from phlash.size_history import DemographicModel


def test_functional1():
    sim = phlash.sim.stdpopsim_dataset(
        "HomSap", "Zigzag_1S14", {"generic": 20}, options={"length_multiplier": 0.01}
    )
    res = phlash.mcmc.fit(list(sim["data"].values()), niter=5, num_particles=123)
    assert isinstance(res, list)
    assert len(res) == 123
    assert isinstance(res[0], DemographicModel)
