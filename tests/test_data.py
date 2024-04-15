import os
import os.path
import tempfile

import msprime
import numpy as np
import pytest
from pytest import fixture

from phlash.data import RawContig, TreeSequenceContig, VcfContig, _chunk_het_matrix


@fixture
def sim():
    return msprime.simulate(4, length=1e6, mutation_rate=1e-4, random_seed=1)


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


def test_psmcfa():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.psmcfa")
    for contig in "1", 0:
        # allow passing by string name or index
        rc = RawContig.from_psmcfa(fn, contig, 100)
        assert rc.het_matrix.shape == (1, 100)
        assert rc.het_matrix.sum() == 82
        assert rc.window_size == 100
    with pytest.raises(ValueError):
        rc = RawContig.from_psmcfa(fn, "2", 100)


def test_vcf():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    vcf = VcfContig(
        fn,
        contig="1",
        interval=(25_000_000, 26_000_000),
        samples=["NA12878", "NA12889"],
    )
    d = vcf.get_data(100)
    assert d["het_matrix"].max() == 2
    assert d["het_matrix"].sum() == 256
    assert np.all(d["afs"] == [143, 60, 89])


def test_ts(sim):
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert d["het_matrix"].max() == 3
    assert d["het_matrix"].sum() == 570
    assert np.all(d["afs"] == [507, 172, 63])


def test_ts_mask_missing(sim):
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"] != -1)
    assert np.all(
        d["afs"]
        == tsc.ts.allele_frequency_spectrum(span_normalise=False, polarised=True)[1:-1]
    )


def test_ts_mask(sim):
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)], mask=[(250, 1000)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"][:, 2:10] == -1)


def test_equal_ts_vcf(sim):
    tsc = TreeSequenceContig(sim, [(0, 1), (2, 3)])
    data_tsc = tsc.get_data(100)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tmp.vcf")
        with open(path, "w") as f:
            tsc.ts.write_vcf(
                f, ploidy=2, position_transform=lambda x: (1 + np.array(x)).astype(int)
            )
        vcfc = VcfContig(
            path,
            samples=["tsk_0", "tsk_1"],
            contig=None,
            interval=None,
            _allow_empty_region=True,
        )
        data_vcf = vcfc.get_data(100)
    for x in ["het_matrix", "afs"]:
        np.testing.assert_allclose(data_tsc[x], data_vcf[x])
