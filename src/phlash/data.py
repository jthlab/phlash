"Classes for importing data"

import re
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import cyvcf2
import dinopy
import numpy as np
import tqdm.auto as tqdm
import tskit
import tszip
from intervaltree import IntervalTree
from jaxtyping import Array, Int, Int8
from loguru import logger

from phlash.mp import JaxCpuProcessPoolExecutor


class ChunkedContig(NamedTuple):
    chunks: Int8[Array, "N L"]
    afs: Int[Array, "n"]


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

    def to_raw(self, window_size: int) -> "RawContig":
        """Convert to a RawContig.

        Note:
            This method is useful for pickling a Contig where the get_data()
            step takes a long time to run.
        """
        return RawContig(**self.get_data(window_size), window_size=window_size)

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
class RawContig(Contig):
    "A contig with pre-computed het matrix and afs."
    het_matrix: Int8[Array, "N L"]
    afs: Int[Array, "n"]
    window_size: int

    @classmethod
    def from_psmcfa(
        cls, psmcfa_path: str, contig: str, window_size: int
    ) -> "RawContig":
        """Construct a contig from a PSMC FASTA (.psmcfa) file.

        Args:
            psmcfa_path: The path to the .psmcfa file.
            contig: The name of the contig to read in.
            window_size: The size of the window that was used when binning entries
                to construct the FASTA file.

        Notes:
            The `window_size` parameter corresponds to the `-s` option that was passed
            to the `fq2psmcfa` utility when creating the .psmcfa file, and is usually
            set to 100bp.
        """
        # parse psmcfa file
        far = dinopy.FastaReader(psmcfa_path)
        try:
            c = next(far.chromosomes(contig))
        except StopIteration:
            raise ValueError(f"A contig named {contig} was not found in {psmcfa_path}")
        seq = np.frombuffer(c.sequence, dtype="c")
        data = (seq == b"K").astype(np.uint8)
        (L,) = data.shape
        afs = np.ones(1)
        return cls(het_matrix=data[None], afs=afs, window_size=window_size)

    @property
    def N(self):
        # the het matrix has one row per diploid pair, so the number of ploids
        # is twice its first dimension.
        if self.het_matrix is None:
            return None
        return 2 * self.het_matrix.shape[0]

    @property
    def L(self):
        if self.het_matrix is None:
            return None
        return self.het_matrix.shape[1] * self.window_size

    def get_data(self, window_size: int):
        if window_size != self.window_size:
            raise ValueError(
                f"This contig was created with a window size of {self.window_size} "
                "but you requested {window_size}"
            )
        return asdict(self)


@dataclass(frozen=True)
class TreeSequenceContig(Contig):
    """Read data from a tree sequence.

    Args:
        ts: tree sequence
        nodes: list of (node1, node2) pairs to include. Each pair corresponds to a
            diploid genome. If None, include all individuals in the tree sequence.
        mask: list of intervals (a, b). All positions within these intervals are
            ignored.
    """

    ts: tskit.TreeSequence
    nodes: list[tuple[int, int]] = None
    mask: list[tuple[int, int]] = None

    @property
    def _nodes(self):
        if self.nodes is not None:
            return self.nodes
        return [tuple(i.nodes) for i in self.ts.individuals()]

    def __post_init__(self):
        try:
            assert isinstance(self._nodes, list)
            for x in self._nodes:
                assert isinstance(x, tuple)
                assert len(x) == 2
                for y in x:
                    assert isinstance(int(y), int)
        except AssertionError:
            raise ValueError(
                "Nodes should be a list of tuples (node1, node2) "
                "leaf node ids in the tree sequence denoting the pairs "
                "of haplotypes that are to be analyzed."
            )

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self._nodes)

    @property
    def L(self):
        return int(self.ts.get_sequence_length())

    def get_data(self, window_size: int):
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
        nodes_flat = [x for t in self._nodes for x in t]
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
    nodes_flat = [x for t in nodes for x in t]
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
            g = v.genotypes.reshape(-1, 2)
            ell = int(v.position / window_size)
            G[:, ell] += g[:, 0] != g[:, 1]
    return G


@dataclass(frozen=True)
class VcfContig(Contig):
    """Read data from a VCF file.

    Args:
        vcf_file: path to VCF file
        contig: contig name
        interval: genomic interval (start, end)
        samples: list of sample ids to include
    """

    vcf_file: str
    samples: list[str]
    contig: str
    interval: tuple[int, int]
    mask: list[tuple[int, int]] = None
    _allow_empty_region: bool = field(
        repr=False, default=False, metadata=dict(docs=False)
    )

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.samples)

    @property
    def L(self):
        "Length of sequence"
        if self.interval is None:
            v = self._vcf
            if self.contig is None:
                assert len(v.seqnames) == 1
                return v.seqlens[0]
            else:
                return v.seqlens[v.seqnames.index(self.contig)]
        return self.interval[1] - self.interval[0]

    def __post_init__(self):
        if self.mask is not None:
            raise NotImplementedError(
                "masking is not yet implemented for VCF files, please use vcftools or "
                "a similar method."
            )
        if not self._allow_empty_region:
            if not self.contig:
                raise ValueError(
                    "contig must be specified. reading in the entire vcf file "
                    "without specifying a contig and region is unsupported."
                )
            if self.interval[0] >= self.interval[1]:
                raise ValueError("region must be an interval (a,b) with a < b")
        if not all(isinstance(s, str) for s in self.samples):
            raise ValueError(
                "samples should be a list of (string) sample identifiers in the vcf"
            )
        diff = set(self.samples) - set(self._vcf.samples)
        if diff:
            raise ValueError(f"the following samples were not found in the vcf: {diff}")

    @property
    def _vcf(self):
        return cyvcf2.VCF(self.vcf_file)

    def get_data(self, window_size: int = 100) -> dict[str, np.ndarray]:
        args = (self.vcf_file, self.samples, window_size)
        if not self._allow_empty_region:
            args += (self.contig, *self.interval)
        return _read_vcf(*args)


def _read_vcf(
    vcf_file: str,
    samples: list[str],
    window_size: int,
    contig: str = None,
    start: int = None,
    end: int = None,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    vcf = cyvcf2.VCF(vcf_file, samples=samples, gts012=True)
    if contig and start and end:
        vcf_iter_args = (f"{contig}:{start}-{end}",)
    else:
        vcf_iter_args = ()
        start = 1
        assert len(vcf.seqnames) == 1
        end = vcf.seqlens[0]
    L = end - start + 1
    N = len(samples)
    afs = np.zeros(2 * N + 1, dtype=np.int64)
    H = np.zeros([N, int(L / window_size)], dtype=np.int8)
    with tqdm.tqdm(total=L, disable=not progress, desc="Reading VCF") as pbar:
        for variant in vcf(*vcf_iter_args):
            x = variant.POS - start
            pbar.update(x - pbar.n)
            i = min(H.shape[1] - 1, int(x / window_size))
            ty = variant.gt_types
            H[:, i] += ty == 1
            # TODO this doesn't handle missing entries correctly
            afs[ty[ty < 3].sum()] += 1
    return dict(het_matrix=H, afs=afs[1:-1])


def contig(
    src: str | tskit.TreeSequence,
    samples: list[str] | list[tuple[int, int]],
    region: str = None,
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
            return VcfContig(src, samples=samples, contig=c, interval=(a, b))
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
    return contig(ts, samples=new_nodes)


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
