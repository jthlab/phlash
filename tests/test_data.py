import os

import msprime
import numpy as np

from eastbay.data import TreeSequenceContig, VcfContig, _chunk_het_matrix


def test_chunk(rng):
    H = rng.integers(0, 2, size=(1, 10_000))
    overlap = 123
    chunk_size = 4_567
    ch = _chunk_het_matrix(H, overlap=overlap, chunk_size=chunk_size)
    assert ch.shape == (3, overlap + chunk_size)
    b = 0
    for ch_i in ch:
        q = min(chunk_size + overlap, len(H[0, b:]))
        assert np.all(ch_i[:q] == H[0, b : b + q])
        b += chunk_size


def test_vcf():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    vcf = VcfContig(fn, "1", (25_000_000, 26_000_000), ["NA12878", "NA12889"])
    d = vcf.get_data(100)
    assert d["het_matrix"].max() == 2
    assert d["het_matrix"].sum() == 256
    assert np.all(d["afs"] == [143, 60, 89])


def test_ts():
    sim = msprime.simulate(4, length=1e6, mutation_rate=1e-4, random_seed=1)
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert d["het_matrix"].max() == 3
    assert d["het_matrix"].sum() == 570
    assert np.all(d["afs"] == [507, 172, 63])


def test_ts_mask_missing():
    sim = msprime.simulate(4, length=1e6, mutation_rate=1e-4, random_seed=1)
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"] != -1)
    assert np.all(
        d["afs"]
        == tsc.ts.allele_frequency_spectrum(span_normalise=False, polarised=True)[1:-1]
    )


def test_ts_mask():
    sim = msprime.simulate(4, length=1e6, mutation_rate=1e-4, random_seed=1)
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)], mask=[(250, 1000)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"][:, 2:10] == -1)
