"Classes for importing data"

import itertools as it
import operator
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache, partial, reduce
from typing import NamedTuple

import jax
import jax.numpy as jnp
import msprime
import numpy as np
import pysam
import scipy
import tqdm.auto as tqdm
import tskit
import tszip
from intervaltree import Interval, IntervalTree
from jaxtyping import Array, Int, Int8
from loguru import logger
from scipy.interpolate import PPoly

from phlash.util import invert_cpwli


class Contig(NamedTuple):
    hets: Int8[Array, "N L 2"]
    afs: Int[Array, "n"]
    window_size: int

    @property
    def L(self):
        return self.hets[0, :, 0].sum()

    def chunk(self, overlap, chunk_size, window_size):
        return _chunk_het_matrix(
            het_matrix=self.hets, overlap=overlap, chunk_size=chunk_size
        )


def contig(
    src: str | tskit.TreeSequence,
    samples: list[str] | list[tuple[int, int]],
    region: str = None,
    genetic_map: msprime.RateMap = None,
) -> Contig:
    """
    Constructs and returns a Contig object based on the source file and specified
    parameters.

    This function supports VCF files (`.vcf`, `.vcf.gz`, `.bcf`), tree sequence files
    (`.trees`, `.ts`), and compressed tree sequence files (`.tsz`, `.tszip`). It
    requires different handling and parameters based on the file type.

    For VCF files, a bcftools region string must be passed in the `region` parameter.
    The function will parse this string to construct a VcfContig object.

    For tree sequence files and their compressed versions, the `region` parameter is
    not supported, and the function constructs a TreeSequenceContig object.

    Parameters:
    - src: Path to the source file, or a tskit.TreeSequence.
    - samples: A list of samples or a list of sample intervals.
    - region: A string specifying the genomic region, required for VCF files.
      Format should be "contig:start-end" (e.g., "chr1:1000-5000").
    - genetic_map: A msprime.RateMap object to use for genetic map information.

    Returns:
    - Contig: A Contig object which can be either VcfContig or TreeSequenceContig based
      on the input file type.

    Raises:
    - ValueError: If the region is not provided or incorrectly formatted for VCF files,
      or if region is provided for tree sequence files. Also raised if loading the file
      fails for any supported format.

    Examples:
    - contig("example.vcf.gz", samples=["sample1", "sample2"], region="chr1:1000-5000")
    - contig("example.trees", samples=[(1, 100), (101, 200)])

    Note:
    - See the documentation of VcfContig and TreeSequenceContig for more details on
    these classes.
    """
    if isinstance(src, str) and any(
        src.endswith(x) for x in [".vcf", ".vcf.gz", ".bcf"]
    ):
        if region is None or not re.match(r"\w+:\d+-\d+", region):
            raise ValueError(
                "VCF files require passing in a bcftools region string. "
                "See docstring for examples."
            )
        c, intv = region.split(":")
        a, b = map(int, intv.split("-"))
        try:
            return VcfContig(
                src, samples=samples, contig=c, interval=(a, b), genetic_map=genetic_map
            )
        except Exception as e:
            raise ValueError(f"Trying to load {src} as a VCF failed") from e

    if isinstance(src, tskit.TreeSequence):
        ts = src
    elif src.endswith(".trees") or src.endswith(".ts"):
        try:
            ts = tskit.load(src)
        except Exception as e:
            raise ValueError(f"Trying to load {src} as a tree sequence failed") from e
    elif src.endswith(".tsz") or src.endswith(".tszip"):
        try:
            ts = tszip.decompress(src)
        except Exception as e:
            raise ValueError(
                f"Trying to load {src} as a compressed tree sequence failed"
            ) from e
    if region is not None:
        raise ValueError(
            "Region string is not supported for tree sequence files. "
            "Use TreeSequence.keep_intervals() instead."
        )
    return TreeSequenceContig(ts, nodes=samples)


def _from_iter(
    rec_iter: Iterable[dict],
    start: int,
    end: int,
    N: int,
    window_size: int = 100,
    genetic_map: msprime.RateMap = None,
    mask: list[tuple[int, int]] = None,
) -> Contig:
    """Construct a contig from an iterator of records.

    Args:
        rec_iter: an iterator of records
        start: start position
        end: end position
        N: number of diploids
        window_size: window size
        genetic_map: map of recombination rates
        mask: list of intervals (a, b) to mask out.
    """
    w = window_size
    tr = IntervalTree.from_tuples(mask or [])
    L = int((end - start) // w)
    hets = np.empty((N, L + 1, 2), dtype=np.int16)
    n = hets[..., 0]
    d = hets[..., 1]
    n[:] = window_size  # num obs
    n[:, -1] = L % w
    d[:] = 0  # num derived

    afs = {}

    # mask out regions of missing values
    if mask is not None:
        for a, b in mask:
            assert start <= a < b <= end
            i = (np.arange(a - start, b - start) // w).astype(int)
            for nn in n:
                np.add.at(nn, i, -1)

    if genetic_map is not None:
        # function representing cumulative recombination rate
        if isinstance(genetic_map, float):
            R = lambda x: x * genetic_map
        else:
            R = PPoly(
                x=np.copy(genetic_map.position),
                c=np.nan_to_num(genetic_map.rate)[None],
            ).antiderivative()
        genetic_pos = []
        genotypes = []

    def is_seg(rec):
        gts = rec["gts"]
        miss = gts < 0
        if miss.all():
            return False
        return gts[~miss].var() > 0  # any variation at all in sample

    for rec in filter(is_seg, rec_iter):
        i = int((rec["pos"] - start) / w)
        if tr.overlaps(i):
            # position is masked, skip
            continue
        gts = rec["gts"]  # [N, 2]
        miss = (gts == -1).any(1)
        n[:, i] -= miss  # missing data
        d[~miss, i] += (gts[:, 0] != gts[:, 1])[~miss]  # number of derived alleles

        # afs calculations: total number of observations and total number of derived
        if (
            gts.max() == 1 and (gts == 0).any()
        ):  # skip tri/tetra allelic sites, and non-seg too
            di = np.sum(gts == 1)
            ni = np.sum(gts >= 0)
            assert di > 0
            afs.setdefault(ni, np.zeros(ni - 1, dtype=int))
            afs[ni][di.sum() - 1] += (
                1  # number of derived alleles, shifted to start at 1
            )

    return Contig(hets=hets, afs=afs, window_size=w)


def _from_ts(
    *,
    ts: tskit.TreeSequence,
    nodes: list[tuple[int, int]],
    left: int = None,
    right: int = None,
    window_size: int = 100,
    genetic_map: msprime.RateMap | float = None,
    mask: list[tuple[int, int]] = None,
    progress: bool = True,
) -> Contig:
    if left is None:
        left = 0

    if right is None:
        right = int(ts.get_sequence_length())

    nodes_flat = [x for t in nodes for x in t]

    def rec_iter():
        tri = []
        for v in ts.variants(samples=nodes_flat, left=left, right=right, copy=True):
            gts = v.genotypes.reshape(-1, 2)
            yield dict(pos=v.position, gts=gts)

    viter = rec_iter()

    if progress:
        pos = ts.tables.sites.position
        ns = np.sum((pos >= left) & (pos < right))
        viter = tqdm.tqdm(
            viter, total=ts.num_sites, desc="Reading tree sequence", unit="sites"
        )

    return _from_iter(
        viter,
        start=left,
        end=right,
        N=len(nodes),
        window_size=window_size,
        genetic_map=genetic_map,
        mask=mask,
    )


Contig.from_ts = _from_ts


def _from_vcf(
    *,
    vcf_path: str,
    contig: str,
    interval: tuple[int, int],
    sample_ids: list[str],
    window_size: int = 100,
    genetic_map: msprime.RateMap | float = None,
    mask: list[tuple[int, int]] = None,
    progress: bool = True,
) -> Contig:
    if len(sample_ids) == 0:
        raise ValueError("No sample ids provided")
    vcf = pysam.VariantFile(vcf_path)  # opens the VCF or BCF file
    vcf.subset_samples(sample_ids)

    if contig not in vcf.header.contigs:
        raise ValueError(f"Contig {contig} not found in VCF header")

    start, end = interval

    def rec_iter():
        # Check if the samples are in the VCF header
        tri = []
        for record in vcf.fetch(contig=contig, start=start, end=end):
            gts = np.zeros((len(sample_ids), 2), dtype=np.int8)
            for i, sample in enumerate(sample_ids):
                gt = record.samples[sample]["GT"]
                gts[i] = [-1 if x is None else x for x in gt]  # missing data
            yield dict(pos=record.pos, gts=gts)

    ri = rec_iter()

    if progress:
        try:
            res = subprocess.run(
                ["bcftools", "index", "--stats", vcf_path],
                check=True,
                capture_output=True,
            )
            for line in res.stdout.decode().split("\n"):
                c, length, ns = line.strip().split("\t")
                if c == contig:
                    ns = int(ns)
                    ri = tqdm.tqdm(ri, total=ns, desc="Reading VCF", unit="sites")
                    break
        except Exception as e:
            logger.warning("bcftools not found, progress bar disabled")
            logger.debug("Exception was: {}", e)

    return _from_iter(
        ri,
        start=start,
        end=end,
        N=len(sample_ids),
        window_size=window_size,
        genetic_map=genetic_map,
        mask=mask,
    )


Contig.from_vcf = _from_vcf


def _from_psmcfa(psmcfa_path: str, contig_name: str, window_size: int = 100) -> Contig:
    """Construct a list of contigs from a PSMC FASTA (.psmcfa) file.

    Args:
        psmcfa_path: The path to the .psmcfa file.
        contig_name: The name of the contig to read from the file.
        window_size: The size of the window that was used when binning entries
            to construct the FASTA file.

    Notes:
        The `window_size` parameter corresponds to the `-s` option that was passed
        to the `fq2psmcfa` utility when creating the .psmcfa file, and is usually
        set to 100bp.
    """
    # parse psmcfa file
    with pysam.FastxFile(psmcfa_path) as fx:
        for record in fx:
            if contig_name != record.name:
                continue
            logger.debug(f"Reading '{contig_name}' from {psmcfa_path}")
            seq = np.array(record.sequence, dtype="c")
            d = (seq == b"K").astype(np.int16)
            n = np.full_like(d, window_size)
            n[seq == b"N"] = 0  # account for missing data
            data = np.stack([n, d], 1)[None]
            afs = np.ones(1)
            return Contig(hets=data, afs=afs, window_size=window_size)
    raise ValueError(
        f"No contig named '{contig_name}' was encountered in {psmcfa_path}"
    )


Contig.from_psmcfa = _from_psmcfa


def _chunk_het_matrix(
    het_matrix: np.ndarray,
    overlap: int,
    chunk_size: int,
) -> np.ndarray:
    data = het_matrix
    assert data.ndim == 3
    data = np.ascontiguousarray(data)
    assert data.data.c_contiguous
    N, L, _ = data.shape
    assert data.shape == (N, L, 2)
    S = chunk_size + overlap
    L_pad = int(np.ceil(L / S) * S)
    padded_data = np.pad(data, [[0, 0], [0, L_pad - L], [0, 0]])
    assert L_pad % S == 0
    num_chunks = L_pad // S
    new_shape = (N, num_chunks, S, 2)
    new_strides = (
        padded_data.strides[0],
        padded_data.strides[1] * chunk_size,
        padded_data.strides[1],
        padded_data.strides[2],
    )
    chunked = np.lib.stride_tricks.as_strided(
        padded_data, shape=new_shape, strides=new_strides
    )
    return np.copy(chunked.reshape(-1, S, 2))


def init_mcmc_data(
    data: list[Contig],
    window_size: int,
    overlap: int,
    chunk_size: int = None,
    max_samples: int = 20,
    num_workers: int = None,
):
    """Chunk up the data. If chunk_size is missing, set it to ~1/5th of the shortest
    contig. (This may not be optimal)."""
    # this has to succeed, we can't have all the het matrices empty
    if all(ds.L is None for ds in data):
        raise ValueError("None of the contigs have a length")
    chunk_size = int(min(0.2 * ds.L / window_size for ds in data if ds.L))
    if chunk_size < 10 * overlap:
        logger.warning(
            "The chunk size is {}, which is less than 10 times the overlap ({}).",
            chunk_size,
            overlap,
        )
    # merge afs
    afss = {}
    for ds in data:
        for n in ds.afs:
            afss.setdefault(n, np.zeros(n - 1, dtype=int))
            afss[n] += ds.afs[n]
    # collect chunks
    chunks = [
        ds.chunk(overlap=overlap, chunk_size=chunk_size, window_size=window_size)
        for ds in data
    ]
    assert len({ch.shape[-1] for ch in chunks}) == 1
    assert all(ch.ndim == 3 for ch in chunks)
    assert all(ch.shape[2] == 2 for ch in chunks)
    return np.sum(afss, 0), np.concatenate(chunks, 0)
