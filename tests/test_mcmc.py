import numpy as np

import eastbay.data
import eastbay.mcmc


def test_sp(rng):
    x = rng.normal(size=100)
    y = _squareplus(x)
    np.testing.assert_allclose(x, _squareplus_inv(y))


def test_functional1():
    truth, chd = eastbay.data.stdpopsim_dataset(
        "SouthMiddleAtlas_1D17", "SouthMiddleAtlas"
    )
    eastbay.mcmc.fit(chd, options=dict(niter=5))
