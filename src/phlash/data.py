"Classes for importing data"

import gzip
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import sgkit
import tqdm.auto as tqdm
import tskit
import tszip
from intervaltree import IntervalTree
from jaxtyping import Array, Int, Int8
from loguru import logger
from pyfaidx import Fasta

from phlash.mp import JaxCpuProcessPoolExecutor


class ChunkedContig(NamedTuple):
    chunks: Int8[Array, "N L"]
    afs: Int[Array, "n"]


def _iter_fasta_records(path: str) -> Iterable[tuple[str, str]]:
    """Yield FASTA records without leaving a sidecar index behind."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = Fasta(
            path,
            indexname=f"{tmpdir}/psmcfa.fai",
            sequence_always_upper=True,
        )
        try:
            for name in fasta.keys():
                yield name, str(fasta[name])
        finally:
            fasta.close()


def _trim_het_matrix(het_matrix: np.ndarray):
    "trim off leading and trailing missing alleles"
    miss = np.all(het_matrix == -1, axis=0)
    a = miss.argmin()
    b = miss[:, a:].argmax()
    ret = het_matrix[:, a : a + b]
    logger.debug("trimmed het matrix from {} to {}", het_matrix.shape, ret.shape)
    return ret


def _chunk_het_matrix(
    het_matrix: np.ndarray,
    overlap: int,
    chunk_size: int,
) -> np.ndarray:
    data = het_matrix.clip(-1, 1).astype(np.int8)
    assert data.ndim == 2
    data = np.ascontiguousarray(data)
    assert data.data.c_contiguous
    N, L = data.shape
    S = chunk_size + overlap
    L_pad = int(np.ceil(L / S) * S)
    padded_data = np.pad(data, [[0, 0], [0, L_pad - L]], constant_values=-1)
    assert L_pad % S == 0
    num_chunks = L_pad // S
    new_shape = (N, num_chunks, S)
    new_strides = (
        padded_data.strides[0],
        padded_data.strides[1] * chunk_size,
        padded_data.strides[1],
    )
    chunked = np.lib.stride_tricks.as_strided(
        padded_data, shape=new_shape, strides=new_strides
    )
    return np.copy(chunked.reshape(-1, S))


class Contig(ABC):
    @abstractmethod
    def get_data(self, window_size: int) -> dict[str, np.ndarray]:
        """Compute the heterozygote matrix and AFS for this contig.

        Returns:
            dict with entries 'het_matrix' and 'afs'. these entries can be None,
            indicating that the contig has no data for that component.
        """
        ...

    @property
    @abstractmethod
    def N(self):
        "Number of ploids in this dataset."
        ...

    @property
    @abstractmethod
    def L(self):
        "Length of sequence"
        ...

    @property
    def size(self):
        if self.L is None or self.N is None:
            return None
        return self.L * self.N

    def to_memory(self, window_size: int) -> "MemoryContig":
        """Materialize this contig into an in-memory representation.

        Note:
            This method is useful for pickling a Contig where the get_data()
            step takes a long time to run.
        """
        d = self.get_data(window_size)
        return MemoryContig.from_data(
            het_matrix=d["het_matrix"],
            afs=d["afs"],
            window_size=window_size,
        )

    def to_chunked(
        self, overlap: int, chunk_size: int, window_size: int = 100
    ) -> ChunkedContig:
        d = self.get_data(window_size)
        if d["het_matrix"] is None:
            ch = None
        else:
            ch = _chunk_het_matrix(
                het_matrix=d["het_matrix"], overlap=overlap, chunk_size=chunk_size
            )
        return ChunkedContig(chunks=ch, afs=d["afs"])


@dataclass(frozen=True)
class MemoryContig(Contig):
    "An in-memory contig backed either by arrays or by a tree sequence."

    het_matrix: Int8[Array, "N L"] | None = None
    afs: Int[Array, "n"] | None = None
    window_size: int | None = None
    ts: tskit.TreeSequence | None = None
    nodes: list[tuple[int, int]] | None = None
    mask: list[tuple[int, int]] | None = None

    @classmethod
    def from_data(
        cls,
        het_matrix: Int8[Array, "N L"],
        afs: Int[Array, "n"],
        window_size: int,
    ) -> "MemoryContig":
        return cls(
            het_matrix=np.asarray(het_matrix, dtype=np.int8),
            afs=np.asarray(afs),
            window_size=int(window_size),
        )

    @classmethod
    def from_psmcfa_iter(
        cls, psmcfa_path: str, window_size: int
    ) -> Iterable["MemoryContig"]:
        """Construct a list of contigs from a PSMC FASTA (.psmcfa) file.

        Args:
            psmcfa_path: The path to the .psmcfa file.
            window_size: The size of the window that was used when binning entries
                to construct the FASTA file.

        Notes:
            The `window_size` parameter corresponds to the `-s` option that was passed
            to the `fq2psmcfa` utility when creating the .psmcfa file, and is usually
            set to 100bp.
        """
        for contig_name, sequence in _iter_fasta_records(psmcfa_path):
            logger.debug(f"Reading contig {contig_name} from {psmcfa_path}")
            seq = np.array(sequence, dtype="c")
            data = (seq == b"K").astype(np.int8)
            data[seq == b"N"] = -1  # account for missing data
            afs = np.ones(1)
            yield cls.from_data(
                het_matrix=data[None],
                afs=afs,
                window_size=window_size,
            )

    @classmethod
    def from_tree_sequence(
        cls,
        ts: tskit.TreeSequence,
        nodes: list[tuple[int, int]] = None,
        mask: list[tuple[int, int]] = None,
    ) -> "MemoryContig":
        return cls(ts=ts, nodes=nodes, mask=mask)

    @property
    def _nodes(self):
        if self.ts is None:
            raise ValueError("Precomputed contigs do not have tree-sequence nodes")
        if self.nodes is not None:
            return self.nodes
        return [tuple(i.nodes) for i in self.ts.individuals()]

    def __post_init__(self):
        precomputed = self.ts is None
        if precomputed:
            if any(x is None for x in (self.het_matrix, self.afs, self.window_size)):
                raise ValueError(
                    "Precomputed MemoryContig requires het_matrix, afs, and window_size"
                )
            if self.nodes is not None or self.mask is not None:
                raise ValueError(
                    "nodes/mask are only valid for tree-sequence-backed MemoryContig"
                )
            return
        if any(x is not None for x in (self.het_matrix, self.afs, self.window_size)):
            raise ValueError(
                "Tree-sequence-backed MemoryContig should not also store precomputed arrays"
            )
        try:
            assert isinstance(self._nodes, list)
            for x in self._nodes:
                assert isinstance(x, tuple)
                assert len(x) == 2
                for y in x:
                    assert isinstance(int(y), int)
        except AssertionError as exc:
            raise ValueError(
                "Nodes should be a list of tuples (node1, node2) leaf node ids in "
                "the tree sequence denoting the haplotype pairs to analyze."
            ) from exc

    @property
    def N(self):
        if self.ts is not None:
            return 2 * len(self._nodes)
        return 2 * self.het_matrix.shape[0]

    @property
    def L(self):
        if self.ts is not None:
            return int(self.ts.get_sequence_length())
        return self.het_matrix.shape[1] * self.window_size

    def get_data(self, window_size: int):
        if self.ts is None:
            if window_size != self.window_size:
                raise ValueError(
                    f"This contig was created with a window size of {self.window_size} "
                    f"but you requested {window_size}"
                )
            return dict(het_matrix=self.het_matrix, afs=self.afs)

        # form interval tree for masking
        mask = self.mask or []
        tr = IntervalTree.from_tuples([(0, self.L)])
        for a, b in mask:
            tr.chop(a, b)
        # compute breakpoints
        bp = np.array([x for i in tr for x in [i.begin, i.end]])
        assert len(set(bp)) == len(bp)
        assert (bp == np.sort(bp)).all()
        if bp[0] != 0.0:
            bp = np.insert(bp, 0, 0.0)
        if bp[-1] != self.L:
            bp = np.append(bp, self.L)
        mid = (bp[:-1] + bp[1:]) / 2.0
        unmasked = [bool(tr[m]) for m in mid]
        nodes_flat = list({x for t in self._nodes for x in t})
        afs = self.ts.allele_frequency_spectrum(
            sample_sets=[nodes_flat], windows=bp, polarised=True, span_normalise=False
        )[unmasked].sum(0)[1:-1]
        het_matrix = _read_ts(self.ts, self._nodes, window_size)
        # now mask out columns of the het matrix based on interval
        # overlap
        tr = IntervalTree.from_tuples(mask)
        column_mask = [
            bool(tr[a : a + window_size]) for a in range(0, self.L, window_size)
        ]
        assert len(column_mask) == het_matrix.shape[1]
        # set mask out these columns
        het_matrix[:, column_mask] = -1
        return dict(afs=afs, het_matrix=het_matrix)


def _read_ts(
    ts: tskit.TreeSequence,
    nodes: list[tuple[int, int]],
    window_size: int,
    progress: bool = False,
) -> np.ndarray:
    nodes_flat = list({x for t in nodes for x in t})
    node_inds = np.array([[nodes_flat.index(x) for x in t] for t in nodes])
    N = len(nodes)
    L = int(np.ceil(ts.get_sequence_length() / window_size))
    G = np.zeros([N, L], dtype=np.int8)
    with tqdm.tqdm(
        ts.variants(samples=nodes_flat, copy=False),
        total=ts.num_sites,
        disable=not progress,
    ) as pbar:
        pbar.set_description("Reading tree sequence")
        for v in pbar:
            g = v.genotypes[node_inds]
            ell = int(v.position / window_size)
            G[:, ell] += g[:, 0] != g[:, 1]
    return G


def _parse_region(region: str) -> tuple[str, tuple[int, int]]:
    if region is None or not re.fullmatch(r"[^:]+:\d+-\d+", region):
        raise ValueError(
            "VCZ inputs require a region string of the form 'contig:start-end'."
        )
    contig, interval = region.split(":")
    start, end = map(int, interval.split("-"))
    if start >= end:
        raise ValueError(
            "region must be an interval 'contig:start-end' with start < end"
        )
    return contig, (start, end)


def _vcz_contigs(ds) -> list[str]:
    if "contigs" in ds.attrs:
        return list(map(str, ds.attrs["contigs"]))
    return list(map(str, ds.contig_id.values))


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ret = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if not ret or start > ret[-1][1]:
            ret.append((start, end))
        else:
            ret[-1] = (ret[-1][0], max(ret[-1][1], end))
    return ret


def _region_to_half_open(interval: tuple[int, int]) -> tuple[int, int]:
    start, end = interval
    return start - 1, end


def _read_bed_intervals(
    bed_file: str,
    contig: str,
) -> list[tuple[int, int]]:
    opener = gzip.open if bed_file.endswith(".gz") else open
    intervals = []
    with opener(bed_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Invalid BED line in {bed_file!r}: {line!r}")
            chrom, start, end = fields[:3]
            if chrom != contig:
                continue
            intervals.append((int(start), int(end)))
    return _merge_intervals(intervals)


def _clip_intervals(
    intervals: Iterable[tuple[int, int]],
    region_start: int,
    region_end: int,
) -> list[tuple[int, int]]:
    clipped = []
    for start, end in intervals:
        start = max(start, region_start)
        end = min(end, region_end)
        if start < end:
            clipped.append((start, end))
    return _merge_intervals(clipped)


def _positions_in_intervals(
    positions: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    if len(intervals) == 0 or len(positions) == 0:
        return np.zeros(len(positions), dtype=bool)
    starts = np.array([a for a, _ in intervals], dtype=int)
    ends = np.array([b for _, b in intervals], dtype=int)
    idx = np.searchsorted(starts, positions, side="right") - 1
    ret = np.zeros(len(positions), dtype=bool)
    valid = idx >= 0
    ret[valid] = positions[valid] < ends[idx[valid]]
    return ret


def _masked_sites_per_window(
    intervals: list[tuple[int, int]],
    region_start: int,
    region_end: int,
    window_size: int,
) -> np.ndarray:
    num_windows = max(1, int(np.ceil((region_end - region_start) / window_size)))
    ret = np.zeros(num_windows, dtype=int)
    for a, b in intervals:
        first = max(0, (a - region_start) // window_size)
        last = min(num_windows - 1, (b - 1 - region_start) // window_size)
        for i in range(first, last + 1):
            w_start = region_start + i * window_size
            w_end = min(region_end, w_start + window_size)
            ret[i] += max(0, min(b, w_end) - max(a, w_start))
    return ret


@dataclass(frozen=True)
class VczContig(Contig):
    """Read data from a VCF Zarr (VCZ) store.

    Args:
        vcz_path: path to a VCZ/Zarr store
        contig: contig name
        interval: genomic interval (start, end)
        samples: list of sample ids to include
    """

    vcz_path: str
    samples: list[str]
    contig: str
    interval: tuple[int, int]
    mask: list[tuple[int, int]] | None = None
    bed_file: str | None = None
    max_missing_sites: int = 0
    _bed_intervals: list[tuple[int, int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.samples)

    @property
    def L(self):
        "Length of sequence."
        start, end = self.interval
        return end - start + 1

    def __post_init__(self):
        if not self.contig:
            raise ValueError("contig must be specified for VCZ inputs")
        if self.interval[0] >= self.interval[1]:
            raise ValueError("interval must satisfy start < end")
        if self.max_missing_sites < 0:
            raise ValueError("max_missing_sites must be nonnegative")
        if not all(isinstance(s, str) for s in self.samples):
            raise ValueError(
                "samples should be a list of sample identifiers in the VCZ store"
            )
        if len(self.samples) == 0:
            raise ValueError("no samples were provided")
        ds = sgkit.load_dataset(self.vcz_path)
        if ds.call_genotype.shape[-1] != 2:
            raise ValueError("phlash currently requires diploid genotypes in VCZ input")
        contigs = set(_vcz_contigs(ds))
        if self.contig not in contigs:
            raise ValueError(f"contig '{self.contig}' was not found in the VCZ store")
        diff = set(self.samples) - set(map(str, ds.sample_id.values))
        if diff:
            raise ValueError(
                f"the following samples were not found in the VCZ store: {diff}"
            )
        bed_intervals = None
        if self.bed_file is not None:
            if not isinstance(self.bed_file, str) or not self.bed_file:
                raise ValueError("bed_file must be a non-empty path string")
            bed_intervals = _read_bed_intervals(self.bed_file, self.contig)
        object.__setattr__(self, "_bed_intervals", bed_intervals)

    def _mask_intervals(self) -> list[tuple[int, int]]:
        intervals = []
        if self.mask:
            intervals.extend(self.mask)
        if self._bed_intervals is not None:
            intervals.extend(self._bed_intervals)
        region_start, region_end = _region_to_half_open(self.interval)
        return _clip_intervals(intervals, region_start, region_end)

    def get_data(self, window_size: int = 100) -> dict[str, np.ndarray]:
        ds = sgkit.load_dataset(self.vcz_path)
        contigs = _vcz_contigs(ds)
        contig_index = contigs.index(self.contig)
        start, end = self.interval
        region_start, region_end = _region_to_half_open(self.interval)
        L = end - start + 1
        N = len(self.samples)
        afs = np.zeros(2 * N + 1, dtype=np.int64)
        num_windows = max(1, int(np.ceil(L / window_size)))
        H = np.zeros([N, num_windows], dtype=np.int8)
        sample_lookup = {str(sample): i for i, sample in enumerate(ds.sample_id.values)}
        sample_index = np.array([sample_lookup[s] for s in self.samples], dtype=int)
        mask_intervals = self._mask_intervals()
        masked_sites = _masked_sites_per_window(
            mask_intervals, region_start, region_end, window_size
        )
        masked_windows = masked_sites > self.max_missing_sites
        variant_contig = np.asarray(ds.variant_contig.values)
        variant_position = np.asarray(ds.variant_position.values)
        variant_index = np.flatnonzero(
            (variant_contig == contig_index)
            & (variant_position >= start)
            & (variant_position <= end)
        )
        if variant_index.size == 0:
            H[:, masked_windows] = -1
            return dict(het_matrix=H, afs=afs[1:-1])

        variant_position0 = variant_position[variant_index] - 1
        variant_masked = _positions_in_intervals(variant_position0, mask_intervals)
        if variant_masked.any():
            variant_index = variant_index[~variant_masked]
            variant_position0 = variant_position0[~variant_masked]
        if variant_index.size == 0:
            H[:, masked_windows] = -1
            return dict(het_matrix=H, afs=afs[1:-1])

        gt = np.asarray(
            ds.call_genotype.isel(variants=variant_index, samples=sample_index).values
        )
        if "call_genotype_mask" in ds:
            gt_mask = np.asarray(
                ds.call_genotype_mask.isel(
                    variants=variant_index, samples=sample_index
                ).values
            )
        else:
            gt_mask = np.zeros_like(gt, dtype=bool)

        missing = (gt < 0) | gt_mask
        het = (gt[..., 0] != gt[..., 1]) & ~missing.any(axis=-1)
        nd = ((gt > 0) & ~missing).sum(axis=(1, 2))

        for pos0, het_row, nd_row in zip(variant_position0, het, nd):
            i = min(num_windows - 1, int((pos0 - region_start) / window_size))
            H[:, i] |= het_row.astype(np.int8)
            afs[int(nd_row)] += 1
        H[:, masked_windows] = -1
        return dict(het_matrix=H, afs=afs[1:-1])


def contig(
    src: str,
    samples: list[str],
    region: str = None,
    bed_file: str | None = None,
    max_missing_sites: int = 0,
) -> Contig:
    """
    Construct a file-backed Contig from a VCZ store.

    Parameters:
    - src: Path to a VCZ/Zarr store.
    - samples: A list of sample identifiers.
    - region: A string specifying the genomic region.
      Format should be "contig:start-end" (e.g., "chr1:1000-5000").
    - bed_file: Optional BED file containing masked intervals for this contig.
      BED intervals are interpreted in the standard 0-based, half-open convention.
    - max_missing_sites: Maximum number of masked bases allowed per window before
      the whole window is marked missing.

    Returns:
    - Contig: A VczContig object.

    Raises:
    - ValueError: If the path is not a VCZ store or the region is invalid.

    Examples:
    - contig("example.vcz", samples=["sample1", "sample2"], region="chr1:1000-5000")
    """
    if not isinstance(src, str):
        raise ValueError("contig() only supports VCZ/Zarr paths.")
    if src.endswith((".vcf", ".vcf.gz", ".bcf", ".trees", ".ts", ".tsz", ".tszip")):
        raise ValueError(
            "Only VCZ/Zarr input is supported. Convert VCF/BCF or tree sequences to "
            "VCZ first, e.g. with bio2zarr."
        )
    if not src.endswith((".vcz", ".zarr")):
        raise ValueError("Only VCZ/Zarr input is supported")
    contig_name, interval = _parse_region(region)
    return VczContig(
        src,
        samples=samples,
        contig=contig_name,
        interval=interval,
        bed_file=bed_file,
        max_missing_sites=max_missing_sites,
    )


def subsample_chrom(chrom_path, populations: tuple[int]):
    # convenience method for the paper analyses
    ts = tszip.decompress(chrom_path)

    nodes = []
    nodes = [
        tuple(ind.nodes)
        for ind, pop_id in zip(ts.individuals(), ts.individual_populations)
        if pop_id in populations
    ]
    nodes_flat = [x for n in nodes for x in n]
    assert nodes_flat
    # not necessary, but cuts down on memory usage and makes the trim step faster
    ts, m = ts.simplify(samples=nodes_flat, map_nodes=True)
    new_nodes = [(m[a], m[b]) for a, b in nodes]
    # the chromosomes are organized into different arms, however the tree sequence spans
    # the entire chromosome. so there is a big "missing" chunk which will appear as
    # nonsegregating if we just ignore it.
    # as a crude hack, just restrict to the interval containing all the sites. this will
    # throw away a few hundred flanking bps on either side, but in such a large dataset,
    # the effect is minimal.
    pos = ts.tables.sites.position
    ts = ts.keep_intervals([[pos.min(), pos.max()]]).trim()
    return MemoryContig.from_tree_sequence(ts, nodes=new_nodes)


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
    afss = []
    # this has to succeed, we can't have all the het matrices empty
    if all(ds.L is None for ds in data):
        raise ValueError("None of the contigs have a length")
    if chunk_size is None:
        chunk_size = int(min(0.2 * ds.L / window_size for ds in data if ds.L))
    if chunk_size < 10 * overlap:
        logger.warning(
            "The chunk size is {}, which is less than 10 times the overlap ({}).",
            chunk_size,
            overlap,
        )
    chunks = []
    total_size = sum(ds.size for ds in data if ds.size)
    with JaxCpuProcessPoolExecutor(num_workers) as pool:
        futs = {}
        for i, ds in enumerate(data):
            fut = pool.submit(
                ds.to_chunked,
                overlap=overlap,
                chunk_size=chunk_size,
                window_size=window_size,
            )
            futs[fut] = i
        with tqdm.tqdm(total=total_size, unit="bp", unit_scale=True) as pbar:
            for f in as_completed(futs):
                i = futs[f]
                size = data[i].size
                # data[i] = None  # free memory associated with dataset
                if size:
                    pbar.update(size)
                d = f.result()
                if d.afs is not None:
                    afss.append(d.afs)
                if d.chunks is not None:
                    chunks.append(d.chunks)

    assert all(a.ndim == 1 for a in afss)
    assert len({a.shape for a in afss}) == 1
    # all afs have same dimension
    assert len({ch.shape[-1] for ch in chunks}) == 1
    assert all(ch.ndim == 2 for ch in chunks)
    return np.sum(afss, 0), np.concatenate(chunks, 0)
