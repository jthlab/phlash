import os
import os.path
import tempfile

import msprime
import numpy as np
import pandas_plink
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
    assert ch.shape == (3, overlap + chunk_size, 2)
    b = 0
    for ch_i in ch:
        q = min(chunk_size + overlap, len(H[0, b:]))
        assert np.all(ch_i[:q] == H[0, b : b + q])
        b += chunk_size


def test_psmcfa(psmcfa_file):
    rc = Contig.from_psmcfa(psmcfa_file, "1", 100)
    assert rc.window_size == 100
    assert rc.hets.shape == (1, 100, 2)
    assert rc.hets[0, :, 0].sum() == 100 * 99  # 1 missing entry has been inserted
    assert rc.hets[0, :, 1].sum() == 81


def test_vcf():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    vcfc = Contig.from_vcf(
        vcf_path=fn,
        contig="1",
        interval=(25_000_000, 26_000_000),
        sample_ids=["NA12878", "NA12889"],
    )
    assert vcfc.hets[..., 1].max() == 2
    assert vcfc.hets[..., 1].sum() == 256
    assert vcfc.afs.keys() == {4}  # sample size
    assert np.all(vcfc.afs[4] == [143, 60, 89])


def test_vcf_ratemap():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    rm = (
        stdpopsim.get_species("HomSap")
        .get_genetic_map("HapMapII_GRCh37")
        .get_chromosome_map("1")
    )
    Contig.from_vcf(
        vcf_path=fn,
        contig="1",
        interval=(25_000_000, 26_000_000),
        sample_ids=["NA12878", "NA12889"],
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
    assert tsc.hets[..., 1].max() == 3
    assert tsc.hets[..., 1].sum() == 570
    assert tsc.afs.keys() == {4}
    assert np.all(tsc.afs[4] == [507, 172, 63])
    assert np.all(
        tsc.afs[4]
        == sim.allele_frequency_spectrum(span_normalise=False, polarised=True)[1:-1]
    )


def test_ts_mask(sim):
    tsc = Contig.from_ts(ts=sim, nodes=[(0, 1), (2, 3)], mask=[(250, 1000)])
    assert np.all(tsc.hets[:, 2:10, 0] == ([50] + [0] * 7))


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
    np.testing.assert_allclose(tsc.hets, vcfc.hets)
    assert tsc.afs.keys() == vcfc.afs.keys() == {4}
    np.testing.assert_allclose(tsc.afs[4], vcfc.afs[4])


@pytest.mark.slow
def test_ld(large_sim):
    buckets = np.arange(0.5, 10.5, 0.5)
    tsc = Contig.from_ts(
        ts=large_sim, genetic_map=1e-8, nodes=[(2 * i, 2 * i + 1) for i in range(10)]
    )
    assert len(tsc.ld) == len(buckets) - 1


def test_ld_vs_hapne(test_assets):
    bim, bed, fam = pandas_plink.read_plink(
        str(test_assets / "ld" / "chr1.from752721.to121475791")
    )
    # sort bim in ascending order of pos
    bim = bim.sort_values("pos")
    # some sites are duplicated, remove them
    bim = bim.drop_duplicates("pos")
    # assert that cm is ascending
    assert np.all(np.diff(bim["cm"]) >= 0)
    # assert that pos is ascending
    assert np.all(np.diff(bim["pos"]) > 0)
    pos, cm = (np.insert(bim[x].values, 0, 0) for x in ["pos", "cm"])
    rate = np.diff(cm) / np.diff(pos) / 100
    rm = msprime.RateMap(position=pos, rate=rate)
    genotype_matrix = fam.compute().astype(np.uint8)
    GenotypeMatrixContig(
        positions=bim["pos"].values, genotype_matrix=genotype_matrix, genetic_map=rm
    )
