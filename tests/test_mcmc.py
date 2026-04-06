import numpy as np
import pytest

import phlash
import phlash.mcmc
import phlash.sim
from phlash.data import MemoryContig
from phlash.size_history import DemographicModel


@pytest.mark.slow
def test_functional1():
    sim = phlash.sim.stdpopsim_dataset(
        "HomSap", "Zigzag_1S14", {"generic": 20}, options={"length_multiplier": 0.01}
    )
    res = phlash.fit(
        list(sim["data"].values()), niter=5, num_particles=123, chunk_size=456
    )
    assert isinstance(res, list)
    assert len(res) == 123
    assert isinstance(res[0], DemographicModel)


def test_functional2():
    het = np.array([[0, 1, 0, 1, 1]], dtype=np.int8)
    afs = np.array([1])
    ctg = MemoryContig.from_data(het, afs, 100)
    res = phlash.fit([ctg], niter=2, num_particles=5, chunk_size=1, overlap=1)
    assert isinstance(res, list)
    assert len(res) == 5
    assert isinstance(res[0], DemographicModel)


def test_psmc(psmcfa_file):
    phlash.psmc([psmcfa_file] * 3, niter=2, num_particles=5, chunk_size=1, overlap=1)


def test_fit_interprets_t1_in_generations_with_mutation_rate(monkeypatch):
    captured = {}

    def fake_init_mcmc_data(*args, **kwargs):
        return np.array([1]), np.array([[0, 1]], dtype=np.int8)

    def fake_from_linear(
        cls, *, pattern, t1, tM, c, theta, rho, alpha=0.0, beta=0.0
    ):
        captured.update(t1=float(t1), tM=float(tM), theta=float(theta))
        raise RuntimeError("stop after init")

    monkeypatch.setattr(phlash.mcmc, "init_mcmc_data", fake_init_mcmc_data)
    monkeypatch.setattr(
        phlash.mcmc.MCMCParams, "from_linear", classmethod(fake_from_linear)
    )

    with pytest.raises(RuntimeError, match="stop after init"):
        phlash.fit([object()], mutation_rate=1.29e-8, t1=1e3, overlap=1, window_size=1)

    N0 = captured["theta"] / 1.29e-8
    assert captured["t1"] == pytest.approx(1e3 / (2.0 * N0))
    assert captured["tM"] == pytest.approx(1e6 / (2.0 * N0))


def test_fit_defaults_t1_to_1k_generations_with_mutation_rate(monkeypatch):
    captured = {}

    def fake_init_mcmc_data(*args, **kwargs):
        return np.array([1]), np.array([[0, 1]], dtype=np.int8)

    def fake_from_linear(
        cls, *, pattern, t1, tM, c, theta, rho, alpha=0.0, beta=0.0
    ):
        captured.update(t1=float(t1), tM=float(tM), theta=float(theta))
        raise RuntimeError("stop after init")

    monkeypatch.setattr(phlash.mcmc, "init_mcmc_data", fake_init_mcmc_data)
    monkeypatch.setattr(
        phlash.mcmc.MCMCParams, "from_linear", classmethod(fake_from_linear)
    )

    with pytest.raises(RuntimeError, match="stop after init"):
        phlash.fit([object()], mutation_rate=1.29e-8, overlap=1, window_size=1)

    N0 = captured["theta"] / 1.29e-8
    assert captured["t1"] == pytest.approx(1e3 / (2.0 * N0))
    assert captured["tM"] == pytest.approx(1e6 / (2.0 * N0))


def test_fit_validates_generation_time_bounds(monkeypatch):
    def fake_init_mcmc_data(*args, **kwargs):
        return np.array([1]), np.array([[0, 1]], dtype=np.int8)

    monkeypatch.setattr(phlash.mcmc, "init_mcmc_data", fake_init_mcmc_data)

    with pytest.raises(ValueError, match="t1 must be less than tM"):
        phlash.fit(
            [object()],
            mutation_rate=1.29e-8,
            t1=1e6,
            tM=1e3,
            overlap=1,
            window_size=1,
        )
