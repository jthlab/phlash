import numpy as np

from phlash.hmm import matvec_smc
from phlash.params import PSMCParams
from phlash.size_history import DemographicModel
from phlash.transition import transition_matrix


def test_matvec(rng):
    dm = DemographicModel.default(pattern="16*1", theta=1e-2, rho=1e-2)
    A = transition_matrix(dm)
    v = rng.uniform(size=16)
    v /= v.sum()
    v1 = v @ A
    pp = PSMCParams.from_dm(dm)
    v2 = matvec_smc(v, pp)
    np.testing.assert_allclose(v1, v2)
