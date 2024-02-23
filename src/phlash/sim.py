"Tools to check performance on simulated data"

import io
import os.path
import re
import shlex
import subprocess
import tempfile
from concurrent.futures import as_completed
from typing import TypedDict

import demes
import numpy as np
import stdpopsim
from loguru import logger

from phlash.data import Contig, TreeSequenceContig, VcfContig
from phlash.mp import JaxCpuProcessPoolExecutor
from phlash.size_history import DemographicModel, SizeHistory


class SimResult(TypedDict):
    data: dict[str, Contig] | dict[str, str]
    truth: DemographicModel


def stdpopsim_dataset(
    species_id: str,
    model_id: str,
    populations: dict[str, int],
    contigs: list[str] = None,
    use_scrm: bool = None,
    seed: int = 1,
    options: dict = {},
) -> SimResult:
    r"""Convenience method for simulating data from the stdpopsim catalog.

    Args:
        species_id: the stdpopsim species identifier (e.g., "HomSap")
        model_id: the unique model identifier (e.g., 'Zigzag_1S14')
        populations: the population(s) and sample sizes.
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
    # not necessary but simplifies N0 computation
    assert len(populations) in [1, 2]
    pop_dict = {pop.name: 0 for pop in model.populations}
    pop_dict.update(populations)
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
    return_vcf = options.get("return_vcf")
    with JaxCpuProcessPoolExecutor(max_workers=options.get("num_threads")) as pool:
        futs = {
            pool.submit(
                _simulate, model, chrom, pop_dict, seed, use_scrm, return_vcf
            ): chrom_id
            for chrom_id, chrom in chroms.items()
        }
        for f in as_completed(futs):
            chrom_id = futs[f]
            ds[chrom_id] = f.result()
    true_eta = compute_truth(model, list(populations))
    true_dm = DemographicModel(eta=true_eta, theta=mu, rho=None)
    return {"data": ds, "truth": true_dm}


def compute_truth(
    model: stdpopsim.DemographicModel, populations: list[str], **kwargs
) -> SizeHistory:
    "Compute pairwise coalescent rate function for model and populations."
    md = model.model.debug()
    t_min = 1e1
    t_max = max(1e5, md.epochs[-1].start_time + 1)
    t_min = kwargs.get("t_min", t_min)
    t_max = kwargs.get("t_max", t_max)
    assert np.isinf(md.epochs[-1].end_time)
    t = np.geomspace(t_min, t_max, 1000)
    if len(populations) == 1:
        d = {p: 2 for p in populations}
    else:
        assert len(populations) == 2
        d = {p: 1 for p in populations}
    c, _ = md.coalescence_rate_trajectory(t, d)
    return SizeHistory(t=t, c=c)


def _get_N0(dm: stdpopsim.DemographicModel, pop_dict: dict) -> float:
    "Compute N0 = ETMRCA / 2. for this demographic model."
    # this involves numerical integration and can be really slow, so it's cached.
    # pop_dict = {pop: # of diploids}, but this function wants {pop: # of ploids}.
    logger.debug("Computing N0 for dm={} pops={}", dm.id, pop_dict)
    return dm.model.debug().mean_coalescence_time(pop_dict, max_iter=20, rtol=0.01) / 2


def _params_for_sim(
    model: stdpopsim.DemographicModel,
    chrom: stdpopsim.Contig,
    pop_dict: dict,
):
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
    return dict(rho=rho, L=L, N0=N0)


def _simulate(
    model: stdpopsim.DemographicModel,
    chrom: stdpopsim.Contig,
    pop_dict: dict,
    seed: int,
    use_scrm: bool,
    return_vcf: bool,
) -> Contig:
    pd = _params_for_sim(model, chrom, pop_dict)
    if use_scrm or (use_scrm is None and pd["rho"] > 1e5 and return_vcf is not False):
        logger.debug(
            "Using scrm for model={}, chrom={}, pops={}", model.id, chrom.id, pop_dict
        )
        try:
            return _simulate_scrm(model, chrom, pop_dict, pd["N0"], seed, return_vcf)
        except Exception as e:
            logger.debug("Running scrm failed: {}", e)
    return _simulate_msp(model, chrom, pop_dict, seed, return_vcf)


def _simulate_msp(model, chrom, pop_dict, seed, return_vcf) -> Contig | str:
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, chrom, pop_dict, seed=seed)
    if return_vcf:

        def pt(x):
            return (1 + np.array(x)).astype(int)

        samples = [f"sample{i}" for i in range(ts.num_individuals)]
        return ts.as_vcf(
            individual_names=samples, position_transform=pt, contig_id=chrom.id
        )
    return TreeSequenceContig(ts)


def _simulate_scrm(model, chrom, pop_dict, N0, seed, return_vcf, out_file=None):
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
    if sum(samples) > 200:
        # for simulating very large samples, reduce the number of recombination windows
        args.extend(["-l", "100r"])
    scrm = os.environ.get("SCRM_PATH", "scrm")
    cmd = [scrm, sum(samples), 1] + args
    cmd = list(map(str, cmd))
    # if an output file is specified, save it and return
    # this functionality is here mainly for the paper pipeline
    if out_file is not None:
        with open(out_file, "w") as f:
            subprocess.run(cmd, stdout=f, text=True)
            return
    # otherwise, process the file and return a vcf
    with subprocess.Popen(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        vcf = _parse_scrm(proc.stdout, chrom.id)
    if return_vcf:
        return vcf
    fd, vcf_path = tempfile.mkstemp(suffix=".vcf")
    with os.fdopen(fd, "wt") as f:
        f.write(vcf)
    n = sum(samples) // 2
    samples = [f"sample{i}" for i in range(n)]
    return VcfContig(
        vcf_path, samples, contig=None, interval=None, _allow_empty_region=True
    ).to_raw(100)


def _parse_scrm(scrm_out, chrom_name) -> str:
    "Create a VCF from the scrm output, parse it, and return a raw contig"
    cmd_line = next(scrm_out).strip()
    L = int(re.search(r"-r [\d.]+ (\d+)", cmd_line)[1])
    scrm_cmds = cmd_line.strip().split(" ")
    assert scrm_cmds[0] == "scrm"
    assert scrm_cmds[2] == "1"  # num reps
    ploids = int(scrm_cmds[1])
    assert ploids % 2 == 0
    n = ploids // 2
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
    vcf = io.StringIO()
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
        n = len(gts)
        assert n % 2 == 0
        gtz = zip(gts[::2], gts[1::2])
        cols += ["|".join(gt) for gt in gtz]
        print("\t".join(cols), file=vcf)
    return vcf.getvalue()


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
