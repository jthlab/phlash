import os
import os.path
import tempfile

import msprime
import numpy as np
import pytest
import stdpopsim
from pytest import fixture

from phlash.data import Contig, _chunk_het_matrix


@fixture
def sim():
    return msprime.simulate(4, length=1e6, mutation_rate=1e-4, random_seed=1)


@fixture
def large_sim():
    return msprime.simulate(
        100,
        length=2e7,
        mutation_rate=1e-8,
        recombination_rate=1e-8,
        Ne=1e4,
        random_seed=1,
    )


def test_chunk(rng):
    H = rng.integers(0, 2, size=(1, 10_000, 2))
    overlap = 123
    chunk_size = 4_567
    ch = _chunk_het_matrix(H, overlap=overlap, chunk_size=chunk_size)
    assert ch.shape == (1, 3, overlap + chunk_size, 2)
    b = 0
    for ch_i in ch[0]:
        q = min(chunk_size + overlap, len(H[0, b:]))
        assert np.all(ch_i[:q] == H[0, b : b + q])
        b += chunk_size


def test_psmcfa(psmcfa_file):
    rc = Contig.from_psmcfa(psmcfa_file, "1", 100)
    assert rc.window_size == 100
    assert rc.hets[0].shape == (1, 100, 2)
    assert rc.hets[0][0, :, 0].sum() == 100 * 99  # 1 missing entry has been inserted
    assert rc.hets[0][0, :, 1].sum() == 81
    assert len(rc.populations) == 1
    assert (rc.hets[1] == rc.populations[0]).all()


def test_vcf():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    vcfc = Contig.from_vcf(
        vcf_path=fn,
        contig="1",
        interval=(25_000_000, 26_000_000),
        sample_ids=["NA12878", "NA12889", ("NA12878", "NA12889")],
    )
    assert vcfc.hets[0][..., 1].max() == 2
    assert vcfc.hets[0][..., 1].sum() == 380
    assert vcfc.afs.keys() == {(4,)}  # sample size
    assert vcfc.populations == (0,)
    assert np.all(vcfc.afs[(4,)][1:-1] == [143, 60, 89])


def test_vcf_ratemap():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    rm = (
        stdpopsim.get_species("HomSap")
        .get_genetic_map("HapMapII_GRCh37")
        .get_chromosome_map("1")
    )
    # not enough samples
    with pytest.raises(ValueError):
        Contig.from_vcf(
            vcf_path=fn,
            contig="1",
            interval=(25_000_000, 26_000_000),
            sample_ids=["NA12878", "NA12889"],
            genetic_map=rm,
            ld_buckets=np.arange(0.1, 0.4, 0.1),
        )
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample_big.bcf")
    Contig.from_vcf(
        vcf_path=fn,
        contig="1",
        interval=(20_000_000, 24_000_000),
        sample_ids=[f"tsk_{i}" for i in range(4)],
        genetic_map=rm,
        ld_buckets=np.arange(0.1, 0.4, 0.1),
    )


def test_vcf_empty_samples():
    # if samples is an empty list, it should raise an error
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    with pytest.raises(ValueError):
        Contig.from_vcf(
            vcf_path=fn,
            contig="1",
            interval=(25_000_000, 26_000_000),
            sample_ids=[],
        )


def test_ts(sim):
    tsc = Contig.from_ts(ts=sim, nodes=[(0, 1), (2, 3)])
    assert tsc.hets[0][..., 1].max() == 3
    assert tsc.hets[0][..., 1].sum() == 570
    assert (tsc.hets[1] == 0).all()
    assert tsc.afs.keys() == {(4,)}
    assert np.all(tsc.afs[4,] == [0, 507, 172, 63, 0])
    assert np.all(
        tsc.afs[4,]
        == sim.allele_frequency_spectrum(span_normalise=False, polarised=True)
    )


def test_ts_repeated_nodes(sim):
    tsc = Contig.from_ts(ts=sim, nodes=[(0, 1), (1, 2), (0, 0)])
    assert tsc.hets[0][..., 1].max() == 2
    assert tsc.hets[0][..., 1].sum() == 591
    assert tsc.afs.keys() == {(3,)}
    assert np.all(tsc.afs[3,] == [0, 330, 235, 0])


def test_ts_mask(sim):
    tsc = Contig.from_ts(ts=sim, nodes=[(0, 1), (2, 3)], mask=[(250, 1000)])
    assert np.all(tsc.hets[0][:, 2:10, 0] == ([50] + [0] * 7))


def test_equal_ts_vcf(sim):
    tsc = Contig.from_ts(ts=sim, nodes=[(0, 1), (2, 3)])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tmp.vcf")
        with open(path, "w") as f:
            sim.write_vcf(
                f, ploidy=2, position_transform=lambda x: (1 + np.array(x)).astype(int)
            )
        # call bcftools index to create the index file
        os.system(f"bgzip {path}")
        path += ".gz"
        os.system(f"bcftools index {path}")
        vcfc = Contig.from_vcf(
            vcf_path=path,
            sample_ids=["tsk_0", "tsk_1"],
            contig="1",
            interval=(1, 1_000_001),
        )
    np.testing.assert_allclose(tsc.hets[0], vcfc.hets[0])
    np.testing.assert_allclose(tsc.hets[1], vcfc.hets[1])
    assert tsc.afs.keys() == vcfc.afs.keys() == {(4,)}
    np.testing.assert_allclose(tsc.afs[4,], vcfc.afs[4,])


@pytest.mark.slow
def test_ld(large_sim):
    buckets = np.arange(0.5, 10.5, 0.5)
    tsc = Contig.from_ts(
        ts=large_sim,
        genetic_map=1e-8,
        nodes=[(2 * i, 2 * i + 1) for i in range(10)],
        ld_buckets=buckets,
    )
    assert len(tsc.ld) == len(buckets) - 1


def test_multipop_equal_ts_vcf(sim_twopop):
    tsc = Contig.from_ts(ts=sim_twopop, nodes=[(0, 1), (2, 3), (4, 5)])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tmp.vcf")
        with open(path, "w") as f:
            sim_twopop.write_vcf(
                f, position_transform=lambda x: (1 + np.array(x)).astype(int)
            )
        # call bcftools index to create the index file
        os.system(f"bgzip {path}")
        path += ".gz"
        os.system(f"bcftools index {path}")
        vcfc = Contig.from_vcf(
            vcf_path=path,
            sample_ids=["tsk_0", "tsk_1", "tsk_2"],
            sample_pops={"tsk_0": "A", "tsk_1": "B", "tsk_2": "B"},
            contig="1",
            interval=(1, 1_000_001),
        )
    np.testing.assert_allclose(tsc.hets[0], vcfc.hets[0])
    assert tsc.afs.keys() == vcfc.afs.keys() == {(2, 4)}
    # FIXME if the mutation rate is high these will be different because tskit counts
    # sites that have experienced more than 1 mutation, whereas we toss them.
    np.testing.assert_allclose(tsc.afs[2, 4], vcfc.afs[2, 4])
    assert tsc.populations == vcfc.populations == tuple("AB")
