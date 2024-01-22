import re
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import NamedTuple

import cyvcf2
import numpy as np
import tqdm.auto as tqdm
import tskit
import tszip
from intervaltree import IntervalTree
from jaxtyping import Array, Int, Int8

from eastbay.log import getLogger
from eastbay.memory import memory
from eastbay.size_history import DemographicModel, SizeHistory

logger = getLogger(__name__)


class ChunkedContig(NamedTuple):
    chunks: Int8[Array, "N L"]
    afs: Int[Array, "n"]


def _trim_het_matrix(het_matrix: np.ndarray):
    "trim off leading and trailing missing alleles"
    miss = np.all(het_matrix == -1, axis=0)
    a = miss.argmin()
    b = miss[:, a:].argmax()
    ret = het_matrix[:, a : a + b]
    logger.debug("trimmed het matrix from %s to %s", het_matrix.shape, ret.shape)
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

    def to_basic(self) -> "RawContig":
        """Convert to a RawContig.

        Note:
            This method is useful for pickling a Contig where the get_data()
            step takes a long time to run.
        """

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
            diploid genome.
        mask: list of intervals (a, b). All positions within these intervals are
            ignored.
    """

    ts: tskit.TreeSequence
    nodes: list[tuple[int, int]]
    mask: list[tuple[int, int]] = None

    def __post_init__(self):
        try:
            assert isinstance(self.nodes, list)
            for x in self.nodes:
                assert isinstance(x, tuple)
                assert len(x) == 2
                for y in x:
                    assert isinstance(y, (int, np.integer))
        except AssertionError:
            raise ValueError(
                "Nodes should be a list of tuples (node1, node2) "
                "leaf node ids in the tree sequence denoting the pairs "
                "of haplotypes that are to be analyzed."
            )

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.nodes)

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
        nodes_flat = [x for t in self.nodes for x in t]
        afs = self.ts.allele_frequency_spectrum(
            sample_sets=[nodes_flat], windows=bp, polarised=True, span_normalise=False
        )[unmasked].sum(0)[1:-1]
        het_matrix = _read_ts(self.ts, self.nodes, window_size)
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


@memory.cache
def _read_ts(
    ts: tskit.TreeSequence,
    nodes: list[tuple[int, int]],
    window_size: int,
    progress: bool = True,
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
class VcfContig:
    """Read data from a VCF file.

    Args:
        vcf_file: path to VCF file
        contig: contig name
        interval: genomic interval (start, end)
        samples: list of sample ids to include
    """

    vcf_file: str
    contig: str
    interval: tuple[int, int]
    samples: list[str]
    mask: list[tuple[int, int]] = None

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.samples)

    @property
    def L(self):
        "Length of sequence"
        return self.interval[1] - self.interval[0]

    def __post_init__(self):
        if self.mask is not None:
            raise NotImplementedError(
                "masking is not yet implemented for VCF files, please use vcftools or "
                "a similar method."
            )
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
        vcf = cyvcf2.VCF(self.vcf_file)
        diff = set(self.samples) - set(vcf.samples)
        if diff:
            raise ValueError(f"the following samples were not found in the vcf: {diff}")

    def get_data(self, window_size: int = 100) -> dict[str, np.ndarray]:
        return _read_vcf(
            self.vcf_file, self.contig, *self.interval, self.samples, window_size
        )


@memory.cache
def _read_vcf(
    vcf_file: str,
    contig: str,
    start: int,
    end: int,
    samples: list[str],
    window_size: int,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    vcf = cyvcf2.VCF(vcf_file, samples=samples, gts012=True)
    region = f"{contig}:{start}-{end}"
    L = end - start + 1
    N = len(samples)
    afs = np.zeros(2 * N + 1, dtype=np.int64)
    H = np.zeros([N, int(L / window_size + 1)], dtype=np.int8)
    with tqdm.tqdm(total=L, disable=not progress, desc="Reading VCF") as pbar:
        for variant in vcf(region):
            x = variant.POS - start
            pbar.update(x - pbar.n)
            i = int(x / window_size)
            ty = variant.gt_types
            H[:, i] += ty == 1
            # TODO this doesn't handle missing entries correctly
            afs[ty[ty < 3].sum()] += 1
    return dict(het_matrix=H, afs=afs[1:-1])


def _find_stdpopsim_model(
    species_id: str,
    model_id: str,
) -> tuple["stdpopsim.Species", "stdpopsim.DemographicModel"]:
    import stdpopsim

    species = stdpopsim.get_species(species_id)
    if isinstance(model_id, stdpopsim.DemographicModel):
        return (species, model_id)
    for model in species.demographic_models:
        if model.id == model_id:
            model = species.get_demographic_model(model.id)
            return (species, model)
    if model_id == "Constant":
        N0 = species.population_size
        model = stdpopsim.PiecewiseConstantSize(N0)
        return (species, model)
    raise ValueError


@memory.cache
def stdpopsim_dataset(
    species_id: str,
    model_id: str,
    population: str | tuple[str, str] = None,
    n_samples: int = 1,
    included_contigs: list[str] = None,
    excluded_contigs: list[str] = ["X", "Mt"],
    options: dict = {},
) -> tuple[DemographicModel, list[tskit.TreeSequence]]:
    r"""Convenience method for simulating data from the stdpopsim catalog.

    Args:
        model_id: the unique model identifier (e.g., 'Zigzag_1S14')
        population: the population from which to draw samples
        n_samples: number of diploid samples to simulate for each contig.
        excluded_contigs: ids of contigs that should be excluded from simulations
            (sex/non-recombining chromosomes, mtDNA, etc.)

    Returns:
        Tuple with two elements. The first element is the "true" demographic model
        under which the data were simulated. The second element is a chunked data set
        consisting of all non-excluded diploid chromosomes for the corresponding
        species.
    """
    # keep this import local since most users won't require it. also it generates
    # annoying warnings on load.
    import stdpopsim

    try:
        species, model = _find_stdpopsim_model(species_id, model_id)
    except ValueError:
        raise ValueError("could not find a model with id %s" % model_id)
    if population is None:
        assert len(model.populations) == 1
        population = model.populations[0].name
    if isinstance(population, tuple):
        assert n_samples % 2 == 0
        assert len(population) == 2
        pop_dict = {p: n_samples // 2 for p in population}
    else:
        pop_dict = {population: n_samples}
    pop_dict.update(
        {pop.name: 0 for pop in model.populations if pop.name not in pop_dict}
    )
    mu = species.genome.chromosomes[0].mutation_rate
    if included_contigs is not None:

        def filt(c):
            return c.id in included_contigs

    else:

        def filt(c):
            return c.id not in excluded_contigs

    chroms = [
        species.get_contig(
            chrom.id,
            mutation_rate=mu,
            length_multiplier=options.get("length_multiplier", 1.0),
        )
        for chrom in species.genome.chromosomes
        if chrom.ploidy == 2 and filt(chrom)
    ]
    engine = stdpopsim.get_engine("msprime")
    ds = []
    with ProcessPoolExecutor(len(chroms)) as pool:
        futs = [
            pool.submit(engine.simulate, model, chrom, pop_dict) for chrom in chroms
        ]
        for f in as_completed(futs):
            ts = f.result()
            ds.append(
                TreeSequenceContig(
                    ts,
                    # this covers the case where the nodes come one or two populations:
                    [(i, n_samples + i) for i in range(n_samples)],
                )
            )

    # representation of true(simulated) demography
    md = model.model.debug()
    t_min = 1e1
    t_max = max(1e5, md.epochs[-1].start_time + 1)
    assert np.isinf(md.epochs[-1].end_time)
    t = np.geomspace(t_min, t_max, 1000)
    if isinstance(population, str):
        d = {population: 2}
    else:
        d = {p: 1 for p in population}
    c, _ = md.coalescence_rate_trajectory(t, d)
    eta = SizeHistory(t=t, c=c)
    true_dm = DemographicModel(eta=eta, theta=mu, rho=None)
    return true_dm, ds


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

    if region is not None:
        raise ValueError(
            "Region string is not supported for tree sequence files. "
            "Use TreeSequence.keep_intervals() instead."
        )
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
    return TreeSequenceContig(ts, nodes=samples)
