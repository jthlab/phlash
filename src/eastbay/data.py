from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cyvcf2
import numpy as np
import stdpopsim
import tqdm.auto as tqdm
import tskit

import eastbay.size_history
from eastbay.log import getLogger
from eastbay.memory import memory

logger = getLogger(__name__)


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


@dataclass
class Contig:
    def get_data(self, window_size: int = 100) -> dict[str, np.ndarray]:
        """Return a dict with keys 'het_matrix' and 'afs'."""
        raise NotImplementedError

    def chunk(
        self, window_size: int, overlap: int, chunk_size: int
    ) -> dict[str, np.ndarray]:
        """Chunk the data into overlapping windows of size chunk_size. The overlap
        between windows is overlap.

        Args:
            window_size: number of base pairs in each window
            overlap: number of base pairs to overlap between windows
            chunk_size: number of base pairs in each chunk

        Returns:
            dict with keys 'chunks' and 'afs'. The value of 'chunks' is a 3d array
            with shape (N, num_chunks, chunk_size).
        """
        d = self.get_data(window_size)
        ch = _chunk_het_matrix(d["het_matrix"], overlap, chunk_size)
        return {"chunks": ch, "afs": d["afs"]}


@dataclass
class TreeSequenceContig(Contig):
    """Read data from a tree sequence.

    Args:
        ts: tree sequence
        nodes: list of (node1, node2) pairs to include. Each pair corresponds to a
            diploid genome.
    """

    ts: tskit.TreeSequence
    nodes: list[tuple[int, int]]

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.nodes)

    @property
    def L(self):
        return self.ts.get_sequence_length()

    def get_data(self, window_size: int = 100) -> dict[str, np.ndarray]:
        ret = {}
        ret["het_matrix"] = _read_ts(self.ts, self.nodes, window_size)
        ret["afs"] = self._afs()
        return ret

    def _afs(self):
        nodes_flat = [x for t in self.nodes for x in t]
        return self.ts.allele_frequency_spectrum(
            sample_sets=[nodes_flat], polarised=True, span_normalise=False
        )[1:-1]


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


@dataclass
class VcfContig(Contig):
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

    @property
    def N(self):
        "Number of ploids in this dataset."
        return 2 * len(self.samples)

    @property
    def L(self):
        "Length of sequence"
        return self.interval[1] - self.interval[0]

    def __post_init__(self):
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
) -> dict[str, np.ndarray]:
    vcf = cyvcf2.VCF(vcf_file, samples=samples, gts012=True)
    region = f"{contig}:{start}-{end}"
    L = end - start + 1
    N = len(samples)
    afs = np.zeros(2 * N + 1, dtype=np.int64)
    H = np.zeros([N, int(L / window_size + 1)], dtype=np.int8)
    for variant in vcf(region):
        i = int((variant.POS - start) / window_size)
        ty = variant.gt_types
        H[:, i] += ty == 1
        afs[ty[ty < 3].sum()] += 1
    return dict(het_matrix=H, afs=afs[1:-1])


def _find_stdpopsim_model(
    species_id: str,
    model_id: str,
) -> tuple[stdpopsim.Species, stdpopsim.DemographicModel]:
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
    excluded_contigs: list[str] = ["X", "Mt"],
    options: dict = {},
) -> tuple[eastbay.size_history.DemographicModel, list[tskit.TreeSequence]]:
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
    chroms = [
        species.get_contig(
            chrom.id,
            mutation_rate=mu,
            length_multiplier=options.get("length_multiplier", 1.0),
        )
        for chrom in species.genome.chromosomes
        if chrom.ploidy == 2 and chrom.id not in excluded_contigs
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
    t_min = 1e0
    t_max = max(1e5, md.epochs[-1].start_time + 1)
    assert np.isinf(md.epochs[-1].end_time)
    t = np.geomspace(t_min, t_max, 1000)
    if isinstance(population, str):
        d = {population: 2}
    else:
        d = {p: 1 for p in population}
    c, _ = md.coalescence_rate_trajectory(t, d)
    eta = eastbay.size_history.SizeHistory(t=t, c=c)
    true_dm = eastbay.size_history.DemographicModel(eta=eta, theta=mu, rho=None)
    return true_dm, ds
