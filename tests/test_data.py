import os
import os.path
import tempfile

import msprime
import numpy as np
import pytest
import sgkit
from pytest import fixture
from sgkit.io.dataset import save_dataset

from phlash.data import (
    MemoryContig,
    VczContig,
    _chunk_het_matrix,
    contig,
)


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


def test_psmcfa(psmcfa_file):
    # allow passing by string name or index
    rc = list(MemoryContig.from_psmcfa_iter(psmcfa_file, 100))
    assert len(rc) == 1
    rc = rc[0]
    assert rc.het_matrix.shape == (1, 100)
    assert rc.het_matrix.sum() == 82
    assert rc.window_size == 100


@fixture
def vcz_path(tmp_path):
    ds = sgkit.create_genotype_call_dataset(
        variant_contig_names=["chr1", "chr2"],
        variant_contig=np.array([0, 0, 0, 1], dtype=np.int8),
        variant_position=np.array([1, 51, 101, 10], dtype=np.int32),
        variant_allele=np.array(
            [["A", "C"], ["G", "T"], ["C", "T"], ["A", "G"]], dtype=object
        ),
        sample_id=np.array(["sample1", "sample2"]),
        call_genotype=np.array(
            [
                [[0, 1], [0, 0]],
                [[1, 1], [0, 1]],
                [[0, 0], [0, 1]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        ),
    )
    path = tmp_path / "sample.vcz"
    save_dataset(ds, path)
    return str(path)


def test_vcz(vcz_path):
    vcz = VczContig(
        vcz_path,
        contig="chr1",
        interval=(1, 200),
        samples=["sample1", "sample2"],
    )
    d = vcz.get_data(100)
    assert d["het_matrix"].max() == 1
    assert d["het_matrix"].tolist() == [[1, 0], [1, 1]]
    assert np.all(d["afs"] == [2, 0, 1])


def test_vcz_empty_samples(vcz_path):
    # if samples is an empty list, it should raise an error
    with pytest.raises(ValueError):
        VczContig(
            vcz_path,
            contig="chr1",
            interval=(1, 200),
            samples=[],
        )


def test_vcz_missing_samples(vcz_path):
    with pytest.raises(ValueError, match="not found"):
        VczContig(
            vcz_path,
            contig="chr1",
            interval=(1, 200),
            samples=["missing"],
        )


def test_contig_vcz_factory(vcz_path):
    ds = contig(vcz_path, samples=["sample1", "sample2"], region="chr1:1-200")
    assert isinstance(ds, VczContig)


def test_contig_rejects_legacy_formats():
    with pytest.raises(ValueError, match="Only VCZ/Zarr input is supported"):
        contig("example.vcf.gz", samples=["sample1"], region="chr1:1-200")
    with pytest.raises(ValueError, match="Only VCZ/Zarr input is supported"):
        contig("example.trees", samples=["sample1"], region="chr1:1-200")


def test_ts(sim):
    tsc = MemoryContig.from_tree_sequence(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert d["het_matrix"].max() == 3
    assert d["het_matrix"].sum() == 570
    assert np.all(d["afs"] == [507, 172, 63])


def test_ts_mask_missing(sim):
    tsc = MemoryContig.from_tree_sequence(sim, [(0, 1), (2, 3)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"] != -1)
    assert np.all(
        d["afs"]
        == tsc.ts.allele_frequency_spectrum(span_normalise=False, polarised=True)[1:-1]
    )


def test_ts_mask(sim):
    tsc = MemoryContig.from_tree_sequence(sim, [(0, 1), (2, 3)], mask=[(250, 1000)])
    d = tsc.get_data(100)
    assert np.all(d["het_matrix"][:, 2:10] == -1)


def test_vcz_from_ts_variants(sim):
    tsc = MemoryContig.from_tree_sequence(sim, [(0, 1), (2, 3)])
    with tempfile.TemporaryDirectory() as d:
        positions = []
        genotypes = []
        for v in tsc.ts.variants(samples=[0, 1, 2, 3], copy=False):
            positions.append(int(1 + v.position))
            genotypes.append(np.asarray(v.genotypes).reshape(2, 2))
        ds = sgkit.create_genotype_call_dataset(
            variant_contig_names=["1"],
            variant_contig=np.zeros(len(positions), dtype=np.int8),
            variant_position=np.asarray(positions, dtype=np.int32),
            variant_allele=np.asarray([["A", "C"]] * len(positions), dtype=object),
            sample_id=np.asarray(["tsk_0", "tsk_1"]),
            call_genotype=np.asarray(genotypes, dtype=np.int8),
        )
        path = os.path.join(d, "tmp.vcz")
        save_dataset(ds, path)
        data_vcz = VczContig(
            path,
            samples=["tsk_0", "tsk_1"],
            contig="1",
            interval=(1, int(tsc.ts.sequence_length)),
        ).get_data(100)
    assert data_vcz["het_matrix"].shape[0] == 2
    assert data_vcz["afs"].sum() == len(positions)
