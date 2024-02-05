import numpy as np

import phlash.data
import phlash.mcmc


def test_sp(rng):
    x = rng.normal(size=100)
    y = _squareplus(x)
    np.testing.assert_allclose(x, _squareplus_inv(y))


def test_functional1():
    truth, chd = phlash.data.stdpopsim_dataset(
        "SouthMiddleAtlas_1D17", "SouthMiddleAtlas"
    )
    phlash.mcmc.fit(chd, options=dict(niter=5))
