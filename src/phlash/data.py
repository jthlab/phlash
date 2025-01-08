"Classes for importing data"

import operator
import subprocess
from collections.abc import Iterable
from typing import NamedTuple

import jax.tree
import msprime
import numpy as np
import pysam
import tqdm.auto as tqdm
import tskit
from intervaltree import IntervalTree
from jaxtyping import Array, Int, Int8
from loguru import logger
from scipy.interpolate import PPoly

from phlash.ld import calc_ld
from phlash.ld.data import LdStats


class Contig(NamedTuple):
    """Container for genetic variation data.

    This class supports VCF files (`.vcf`, `.vcf.gz`, `.bcf`), tree sequence files
    (`.trees`, `.ts`), and compressed tree sequence files (`.tsz`, `.tszip`). It
    requires different handling and parameters based on the file type.

    Examples:
    >>> Contig.from_vcf(vcf_path="example.vcf.gz",
                        samples=["sample1", "sample2"],
                        region="chr1:1000-5000")
    >>> ts = tskit.load("example.trees")
    >>> Contig.from_ts(ts=ts, nodes=[(0, 1), (2, 3)])
    """

    hets: Int8[Array, "N L 2"]  # noqa: F722
    afs: Int[Array, "n"]  # noqa: F821
    ld: dict[tuple[float, float], float]
    window_size: int
    pop_indices: Int8[Array, "N 2"]
    populations: tuple[str]

    @property
    def L(self):
        return self.hets[0, :, 0].sum()

    def chunk(self, overlap, chunk_size, window_size):
        return _chunk_het_matrix(
            het_matrix=self.hets, overlap=overlap, chunk_size=chunk_size
        )

    @classmethod
    def from_raw(cls, hets):
        N = hets.shape[0]
        w = hets[0, 0, 0]
        assert np.all(hets[..., 0] == w)
        return cls(
            hets,
            afs=None,
            ld=None,
            window_size=w,
            pop_indices=np.zeros(N, dtype=np.int8),
            populations=(0,),
        )


def _from_iter(
    rec_iter: Iterable[dict],
    start: int,
    end: int,
    N: int,
    window_size: int = 100,
    genetic_map: msprime.RateMap = None,
    ld_buckets: np.ndarray = None,
    ld_region_size: int = 1_000_000,
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
    L = end - start
    Lw = int(L // w)
    hets = np.empty((N, Lw + 1, 2), dtype=np.int16)
    n = hets[..., 0]
    d = hets[..., 1]
    n[:] = window_size  # num obs
    n[:, -1] = L % w
    d[:] = 0  # num derived

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

            def R(x):
                return x * genetic_map
        else:
            R = PPoly(
                x=np.copy(genetic_map.position),
                c=np.nan_to_num(genetic_map.rate)[None],
            ).antiderivative()
        physical_pos = []
        genetic_pos = []
        genotypes = []

    def is_seg(rec):
        gts = rec["gts"]
        miss = gts < 0
        if miss.all():
            return False
        return gts[~miss].var() > 0  # any variation at all in sample

    afs = {}

    for rec in filter(is_seg, rec_iter):
        i = int((rec["pos"] - start) / w)
        if tr.overlaps(i):
            # position is masked, skip
            continue

        gts = rec["gts"]  # [N, 2]
        miss = (gts == -1).any(1)
        n[:, i] -= miss  # missing data
        d[~miss, i] += (gts[:, 0] != gts[:, 1])[~miss]  # number of derived alleles

        if rec.get("afs"):
            k, a = rec["afs"]
            sh = np.array(k) + 1
            v = tuple(a.values())
            afs.setdefault(k, np.zeros(sh, dtype=int))
            afs[k][v] += 1

        # ld calculations
        if genetic_map is not None:
            physical_pos.append(rec["pos"])
            genetic_pos.append(R(rec["pos"]))
            genotypes.append(gts)

    ret = Contig(
        hets=hets, afs=afs, ld=None, window_size=w, populations=None, pop_indices=None
    )

    if genetic_map is not None:
        if ld_buckets is None:
            ld_buckets = np.geomspace(1e-6, 1e-2, 12)
        ld = calc_ld(
            np.array(physical_pos),
            np.array(genetic_pos),
            np.array(genotypes),
            ld_buckets,
            ld_region_size,
        )
        ret = ret._replace(ld=ld)

    return ret


def _from_ts(
    *,
    ts: tskit.TreeSequence,
    nodes: list[tuple[int, int]],
    left: int = None,
    right: int = None,
    window_size: int = 100,
    genetic_map: msprime.RateMap | float = None,
    ld_buckets: np.ndarray = None,
    mask: list[tuple[int, int]] = None,
    progress: bool = True,
) -> Contig:
    if left is None:
        left = 0

    if right is None:
        right = int(ts.get_sequence_length())

    nodes_flat = list({x for tup in nodes for x in tup})
    nodes_map = np.array([list(map(nodes_flat.index, tup)) for tup in nodes])

    def rec_iter():
        for v in ts.variants(samples=nodes_flat, left=left, right=right, copy=True):
            gts = v.genotypes[nodes_map]
            yield dict(pos=v.position, gts=gts)

    viter = rec_iter()

    if progress:
        pos = ts.tables.sites.position
        ns = np.sum((pos >= left) & (pos < right))
        viter = tqdm.tqdm(viter, total=ns, desc="Reading tree sequence", unit="sites")

    ret = _from_iter(
        viter,
        start=left,
        end=right,
        N=len(nodes),
        window_size=window_size,
        genetic_map=genetic_map,
        ld_buckets=ld_buckets,
        mask=mask,
    )

    def pop_lookup(n):
        p = ts.node(n).population
        try:
            return ts.population(p).metadata["name"]
        except Exception:
            return p

    node_pops = jax.tree.map(pop_lookup, nodes)
    all_pops = tuple(
        sorted(jax.tree.reduce(operator.or_, jax.tree.map(lambda x: {x}, node_pops)))
    )
    if -1 in all_pops and all_pops != (-1,):
        raise ValueError(
            "Some sample nodes have missing populations while others have "
            "non-missing populations."
        )

    # convert to numerical indices
    node_pops = jax.tree.map(all_pops.index, node_pops)

    sample_sets = {}
    for p in all_pops:
        sample_sets[p] = [n for n in nodes_flat if pop_lookup(n) == p]
    sample_sets = list(map(list, sample_sets.values()))
    afs = ts.allele_frequency_spectrum(
        sample_sets, span_normalise=False, polarised=True
    )
    # tskit has no missing sites, so the number of "non-missing" observations is just
    # the shape minus one
    afsf = afs.reshape(-1)
    afsf[0] = afsf[-1] = 0  # mask out non-seg entries
    k = tuple([n - 1 for n in afs.shape])
    pop_indices = np.array(node_pops, dtype=np.int8)
    (N, L, _) = ret.hets.shape
    assert ret.hets.shape == (N, L, 2)
    assert pop_indices.shape == (N, 2)
    return ret._replace(afs={k: afs}, populations=all_pops, pop_indices=pop_indices)


Contig.from_ts = _from_ts


def _from_vcf(
    *,
    vcf_path: str,
    contig: str,
    interval: tuple[int, int],
    sample_ids: list[str | tuple[str, str]],
    sample_pops: dict[str, str] = None,
    window_size: int = 100,
    genetic_map: msprime.RateMap | float = None,
    ld_buckets: np.ndarray = None,
    mask: list[tuple[int, int]] = None,
    progress: bool = True,
) -> Contig:
    vcf = pysam.VariantFile(vcf_path)  # opens the VCF or BCF file
    if contig not in vcf.header.contigs:
        raise ValueError(f"Contig {contig} not found in VCF header")
    if not sample_ids:
        raise ValueError("Empty samples")

    start, end = interval

    # convert diploid to phased haploid
    sample_ids = jax.tree.map(lambda x: (x, x) if isinstance(x, str) else x, sample_ids)

    # find unique ids to speed up iteration
    all_ids = jax.tree.reduce(operator.or_, jax.tree.map(lambda a: {a}, sample_ids))
    vcf.subset_samples(all_ids)

    if sample_pops is None:
        sample_pops = {sid: 0 for sid in all_ids}

    all_pops = list(set(sample_pops.values()))
    sample_pop_ids = jax.tree.map(
        all_pops.index, jax.tree.map(sample_pops.__getitem__, sample_ids)
    )

    def rec_iter():
        # Check if the samples are in the VCF header
        gts = np.zeros((len(sample_ids), 2), dtype=np.int8)
        esi = list(enumerate(sample_ids))
        for record in vcf.fetch(contig=contig, start=start, end=end):
            d = {p: 0 for p in all_pops}
            n = {p: 0 for p in all_pops}
            for i, sids in esi:
                if (sids[0] != sids[1]) and not all(
                    record.samples[s].phased for s in sids
                ):
                    raise ValueError(
                        "Attempting to record genotype for {sids[0]}|{sids[1]}, but "
                        "the variant is not phased."
                    )
                gt = [record.samples[s]["GT"][j] for j, s in enumerate(sids)]
                gts[i] = [-1 if x is None else x for x in gt]  # missing data
                for j in range(2):
                    s = sids[j]
                    nmiss = gts[i][j] >= 0
                    d[sample_pops[s]] += gts[i][j] * nmiss
                    n[sample_pops[s]] += nmiss
            if (gts <= 1).all():
                # ignore multi-allelic sites
                v = tuple(n.values())
                yield dict(pos=record.pos, gts=gts, afs=(v, d))

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

    ret = _from_iter(
        ri,
        start=start,
        end=end,
        N=len(sample_pops),
        window_size=window_size,
        genetic_map=genetic_map,
        ld_buckets=ld_buckets,
        mask=mask,
    )
    pop_indices = np.array(sample_pop_ids)
    return ret._replace(populations=tuple(all_pops), pop_indices=pop_indices)


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
            N, L, _ = data.shape
            assert data.shape == (N, L, 2)
            pop = np.full([N, 2], -1, dtype=np.int8)
            return Contig(
                hets=data,
                afs=afs,
                ld=None,
                window_size=window_size,
                populations=(-1,),
                pop_indices=pop,
            )
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
    return np.copy(chunked)


def init_chunks(
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
    if not all(c.populations == data[0].populations for c in data):
        raise ValueError("All contigs must be defined on the same populations")
    pops = data[0].populations
    if all(ds.L is None for ds in data):
        raise ValueError("None of the contigs have a length")
    chunk_size = int(min(0.2 * ds.L / ds.window_size for ds in data if ds.L))
    if chunk_size < 10 * overlap:
        logger.warning(
            "The chunk size is {}, which is less than 10 times the overlap ({}).",
            chunk_size,
            overlap,
        )
    # collect chunks
    chunks = [
        ds.chunk(overlap=overlap, chunk_size=chunk_size, window_size=window_size)
        for ds in data
    ]
    assert len({ch.shape[-1] for ch in chunks}) == 1
    assert all(ch.ndim == 4 for ch in chunks)
    assert all(ch.shape[3] == 2 for ch in chunks)
    combined_chunks = np.concatenate(chunks, 0)
    combined_pop_indices = np.concatenate([ds.pop_indices for ds in data], 0)
    assert len(combined_chunks) == len(combined_pop_indices)
    return combined_chunks, pops, combined_pop_indices


def init_afs(data: list[Contig]) -> dict[int, np.ndarray]:
    # merge afs
    afss = {}
    for ds in data:
        if ds.afs is None:
            continue
        for n, a in ds.afs.items():
            afss.setdefault(n, np.zeros_like(a))
            afss[n] += a
    return afss


def init_ld(data: list[Contig]) -> dict[tuple[float, float], np.ndarray]:
    # merge ld
    lds = {}
    for d in data:
        if d.ld is not None:
            for k, v in d.ld.items():
                lds.setdefault(k, []).extend(v)
    if lds:
        # convert list-of-pytrees to pytree of arrays
        lds = {k: LdStats.summarize(v) for k, v in lds.items() if v}  # v could be empty
    return lds
