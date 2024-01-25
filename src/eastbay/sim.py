"Tools to check performance on simulated data"

import os.path
import re
import shlex
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import TemporaryDirectory
from typing import TypedDict

import demes
import numpy as np
import sh
import stdpopsim
from loguru import logger

from eastbay.data import Contig, TreeSequenceContig, VcfContig
from eastbay.memory import memory
from eastbay.size_history import DemographicModel, SizeHistory


class SimResult(TypedDict):
    data: list[Contig]
    truth: DemographicModel


@memory.cache
def stdpopsim_dataset(
    species_id: str,
    model_id: str,
    population: str | tuple[str, str] = None,
    n_samples: int = 1,
    contigs: list[str] = None,
    use_scrm: bool = None,
    seed: int = 1,
    options: dict = {},
) -> SimResult:
    r"""Convenience method for simulating data from the stdpopsim catalog.

    Args:
        species_id: the stdpopsim species identifier (e.g., "HomSap")
        model_id: the unique model identifier (e.g., 'Zigzag_1S14')
        population: the population(s) from which to draw samples. if a tuple of two
            populations are specified, the returned dataset will contain diploid
            genomes where one ploid is drawn from each population.
        n_samples: number of diploid samples to simulate for each contig.
        contigs: ids of contigs that should be simulated. if None, all contigs
            that are diploid; numbered; and have recombination rate >0 are simulated.
        use_scrm: if True, use scrm instead of msprime to simulate data. If None (the
            default), scrm will be used if the scaled recombination rate r=4NrL>1e5.

    Returns:
        Tuple with two elements. The first element is the "true" pairwise coalescent
        rate function for the model under which the data were simulated. The second
        element is a chunked data set consisting of all simulated chromosomes for the
        corresponding species.
    """
    try:
        species, model = _find_stdpopsim_model(species_id, model_id)
    except ValueError:
        raise ValueError("could not find a model with id %s" % model_id)
    if population is None:
        assert len(model.populations) == 1
        population = model.populations[0].name
    if isinstance(population, tuple):
        assert len(population) == 2
        pop_dict = {p: n_samples for p in population}
    else:
        pop_dict = {population: n_samples}
    pop_dict.update(
        {pop.name: 0 for pop in model.populations if pop.name not in pop_dict}
    )
    mu = species.genome.chromosomes[0].mutation_rate
    if contigs is not None:

        def filt(c):
            return c.id in contigs

    else:

        def filt(c):
            return (
                (c.ploidy == 2)
                and (c.recombination_rate > 0)
                and re.match(r"\d+", c.id)
            )

    chroms = {
        chrom.id: species.get_contig(
            chrom.id,
            mutation_rate=mu,
            length_multiplier=options.get("length_multiplier", 1.0),
        )
        for chrom in filter(filt, species.genome.chromosomes)
    }
    for chrom_id, chrom in chroms.items():
        chrom.id = chrom_id
    ds = {}
    with ProcessPoolExecutor(8) as pool:
        futs = {
            pool.submit(_simulate, model, chrom, pop_dict, seed, use_scrm): chrom_id
            for chrom_id, chrom in chroms.items()
        }
        for f in as_completed(futs):
            chrom_id = futs[f]
            ds[chrom_id] = f.result()

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
    return {"data": ds, "truth": true_dm}


@memory.cache
def _get_N0(dm: stdpopsim.DemographicModel, pop_dict: dict) -> float:
    "Compute N0 = ETMRCA / 2. for this demographic model."
    # this involves numerical integration and can be really slow, so it's cached.
    # pop_dict = {pop: # of diploids}, but this function wants {pop: # of ploids}.
    logger.debug("Computing N0 for dm={} pops={}", dm.id, pop_dict)
    return dm.model.debug().mean_coalescence_time(pop_dict, max_iter=20, rtol=0.01) / 2


def _simulate(
    model: stdpopsim.DemographicModel,
    chrom: stdpopsim.Contig,
    pop_dict: dict,
    seed: int,
    use_scrm: bool,
) -> Contig:
    active_pops = [p for p, n in pop_dict.items() if n > 0]
    if len(active_pops) == 1:
        pd = {active_pops[0]: 2}
    else:
        assert len(active_pops) == 2
        pd = {p: 1 for p in active_pops}
    N0 = _get_N0(model, pd)
    r = chrom.recombination_map.rate
    assert len(r) == 1
    r = r.item()
    L = chrom.length
    rho = 4 * N0 * r * L
    if use_scrm or (use_scrm is None and rho > 1e5):
        logger.debug(
            "Using scrm for model={}, chrom={}, pops={}", model.id, chrom.id, pop_dict
        )
        return _simulate_scrm(model, chrom, pop_dict, N0, seed)
    else:
        return _simulate_msp(model, chrom, pop_dict, seed)


def _simulate_msp(model, chrom, pop_dict, seed) -> Contig:
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, chrom, pop_dict, seed=seed)
    ips = ts.individual_populations
    n = len(ips)
    if len(pop_dict) == 2:
        assert len(set(ips[: n // 2])) == len(set(ips[n // 2 :])) == 1
    else:
        assert len(set(ips)) == 1
    # either return all nodes from single population, or zip together
    # nodes from two populations
    nodes = [(i, n + i) for i in range(n)]
    return TreeSequenceContig(ts, nodes)


def _simulate_scrm(model, chrom, pop_dict, N0, seed) -> Contig:
    scrm = sh.Command(os.environ.get("SCRM_PATH", "scrm"))
    assert chrom.interval_list[0].shape == (1, 2)
    assert chrom.interval_list[0][0, 0] == 0.0
    L = chrom.interval_list[0][0, 1]
    theta = 4 * N0 * chrom.mutation_rate * L
    assert chrom.recombination_map.rate.shape == (1,)
    rho = 4 * N0 * chrom.recombination_map.rate[0] * L
    g = model.model.to_demes()
    samples = [0] * len(g.demes)
    for pop, n in pop_dict.items():
        i = [d.name for d in g.demes].index(pop)
        samples[i] += 2 * n
    cmd = demes.to_ms(g, N0=N0, samples=samples)
    args = shlex.split(cmd)
    args.extend(
        [
            "-t",
            theta,
            "-r",
            rho,
            L,
            "--transpose-segsites",
            "-SC",
            "abs",
            "-p",
            14,
            "-oSFS",
            "-seed",
            seed,
        ]
    )
    scrm_out = scrm(sum(samples), 1, *args, _iter=True)
    return _parse_scrm(scrm_out, chrom.id)


def _parse_scrm(scrm_out, chrom_name) -> Contig:
    "Create a VCF from the scrm output, parse it, and return a raw contig"
    cmd_line = next(scrm_out).strip()
    L = int(re.search(r"-r [\d.]+ (\d+)", cmd_line)[1])
    scrm_cmds = cmd_line.strip().split(" ")
    assert scrm_cmds[0] == "scrm"
    assert scrm_cmds[2] == "1"  # num reps
    ploids = int(scrm_cmds[1])
    assert ploids % 2 == 0
    n = ploids // 2

    tmpdir = TemporaryDirectory()
    vcf_path = os.path.join(tmpdir.name, "tmp.vcf")
    header = [
        "##fileformat=VCFv4.0",
        """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">""",
    ]
    header.append(f"##contig=<ID={chrom_name},length={L}>")
    h = "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT".split()
    samples = ["sample%d" % i for i in range(n)]
    h.extend(samples)
    header.append("\t".join(h))
    while not next(scrm_out).startswith("position"):
        continue
    # if there are two populations then we will combine haplotypes from each
    with open(vcf_path, "w") as vcf:
        print("\n".join(header), file=vcf)
        for line in scrm_out:
            if line.startswith("SFS: "):
                #     with open(snakemake.output[1], 'wt') as sfs:
                #         sfs.write(line[5:])
                continue
            pos, _, *gts = line.strip().split(" ")
            # vcf is 1-based; if a variant has pos=0 it messes up bcftools
            pos = int(1 + float(pos))
            cols = [chrom_name, str(pos), ".", "A", "C", ".", "PASS", ".", "GT"]
            # zip together one ploid from each population. this covers both cases:
            # - single population, in which case everything is exchangeable;
            # - two populations, in which case we take a ploid from each population
            n = len(gts)
            assert n % 2 == 0
            gtz = zip(gts[: n // 2], gts[n // 2 :])
            cols += ["|".join(gt) for gt in gtz]
            print("\t".join(cols), file=vcf)

    return VcfContig(
        vcf_path, samples, contig=None, interval=None, _allow_empty_region=True
    ).to_raw(100)


def _find_stdpopsim_model(
    species_id: str,
    model_or_id: str,
) -> tuple[stdpopsim.Species, stdpopsim.DemographicModel]:
    species = stdpopsim.get_species(species_id)
    if isinstance(model_or_id, stdpopsim.DemographicModel):
        return (species, model_or_id)
    model_id = model_or_id
    for model in species.demographic_models:
        if model.id == model_id:
            model = species.get_demographic_model(model.id)
            return (species, model)
    raise ValueError(f"Couldn't find a demographic model with id '{model_or_id}'")
