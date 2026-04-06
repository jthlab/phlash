---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

This document explains how to run <code>phlash</code>. Before doing so, please ensure that your system meets the [requirements](../README.md) and that you have installed the package.

## If you're coming from PSMC
If you are familiar with the PSMC software, you may find it helpful to know that `phlash` is conceptually similar to PSMC, but with a few key differences:

- `phlash` does not (yet) have a command-line interface. It is a Python package that is imported and used in a Python script or Jupyter notebook.
- `phlash` is a Bayesian method that estimates a posterior distribution over demographic models, rather than a point estimate. This means that the output of `phlash` is a list of demographic models, each of which is a valid sample from the posterior distribution.
- `phlash` uses a single file-backed input format: VCZ, a Zarr store in sgkit's genotype dataset layout.

If you already have .psmcfa files (generated using i.e. the `fq2psmcfa` utility), a convenience function is provided for reanalyzing them with `phlash`:

```python
# import phlash
# posterior_samples = phlash.psmc(['/path/to/file1.psmcfa', '/path/to/file2.psmcfa', ...])
```

## General usage guide
The following is a general guide to using `phlash` in a Jupyter notebook. For more detailed information, please refer to the [API documentation](../docs/build/html/index.html).

### Importing the package
Load the package by executing:

```python
import phlash
```

### Loading your data

The `phlash.contig()` function is used to specify the contig(s) you will use to perform your analysis.
<code>phlash</code> intentionally uses a single file-backed input format: VCZ/Zarr.
If your data currently live in another format, convert them first using the official
tools:

- VCF / BCF to VCZ: [`bio2zarr` `vcf2zarr`](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html)
- PLINK to VCZ: [`bio2zarr` `plink2zarr`](https://sgkit-dev.github.io/bio2zarr/plink2zarr/overview.html)
- tskit to VCZ: [`bio2zarr` `tskit2zarr`](https://sgkit-dev.github.io/bio2zarr/tskit2zarr/overview.html)

<!-- #region -->
#### Loading VCZ data


For example, to load data for sample `NA12878` from chromosome 22 in a preconverted
1000 Genomes VCZ store, execute the following:
<!-- #endregion -->

```python
import os.path

onekg_base = "/scratch/1kg"  # update with path on your local system
template = "ALL.{chrom}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcz"

chr22_path = os.path.join(onekg_base, template.format(chrom="chr22"))
chr22_c = phlash.contig(chr22_path, samples=["NA12878"], region="22:5000000-30000000")
chroms_1kg = [chr22_c]
```

To load data from all the autosomes, simply repeat this command for each of them:

```python
# chroms_1kg = []
# for chrom in range(1, 23):
#     path = os.path.join(onekg_base, template.format(chrom=f"chr{chrom}"))
#     chroms_1kg.append(
#         phlash.contig(path, samples=["NA12878"], region=f"{chrom}:1-10000000")
#     )
```

Notice that, to prevent inadvertent errors (such as the inclusion of telomeric regions into the analysis), the `samples=` and `region=` arguments are required when loading VCZ data.

You may also provide a BED file of masked regions. Masked bases are treated as missing,
and `max_missing_sites` controls how many masked bases are tolerated within a window
before the entire window is marked missing:

```python
chr22_masked = phlash.contig(
    chr22_path,
    samples=["NA12878"],
    region="22:5000000-30000000",
    bed_file="/path/to/mask.bed.gz",
    max_missing_sites=20,
)
```

Tree-sequence, VCF/BCF, and PLINK inputs should be converted to VCZ before calling
`phlash.contig()`.


### Fitting the model

Estimation is performed using `phlash.fit()`. In the most basic use-case, it takes a list of contigs and fits the model:

```python
results = phlash.fit(chroms_1kg, mutation_rate=1.29e-8)
```

The output of `fit()` is a list of `phlash.size_history.DemographicModel` classes. These are just [named tuples](https://docs.python.org/3/library/collections.html#collections.namedtuple) with fields `theta`, `rho`, and `eta`. The latter is itself an instance of `phlash.size_history.SizeHistory`, which represents a piecewise-constant size history function.

Since each `DemographicModel` is a valid posterior sample, posterior inference is easy: just examine the empirical distribution of whatever statistic you are interested in. For example, to plot the pointwise posterior median:

```python
import matplotlib.pyplot as plt
import numpy as np

times = np.array([dm.eta.t[1:] for dm in results])
# choose a grid of points at which to evaluate the size history functions
T = np.geomspace(times.min(), times.max(), 1000)
Nes = np.array([dm.eta(T, Ne=True) for dm in results])
plt.plot(T, np.median(Nes, axis=0))
plt.xscale('log')
plt.yscale('log')
```

#### Rescaling the output
By default, `phlash` works in the coalescent scaling -- it assumes that the mutation rate per unit of time is $\theta = 4 N_0 \mu$, and estimates $\theta$ by Watterson's formula. If the true rate of mutation is known, you may specify it at estimation time by passing the `mutation_rate=` parameter to `fit()`, for example:

```python
results = phlash.fit(chroms_1kg, mutation_rate=1.29e-8)
```

Alternatively, and equivalently, you can use the `DemographicModel.rescale()` function to rescale the output after fitting. So, to modify the above plot to be in units of generations, use:

```python
import matplotlib.pyplot as plt
import numpy as np

times = np.array([dm.rescale(1.29e-8).eta.t[1:] for dm in results])
# choose a grid of points at which to evaluate the size history functions
T = np.geomspace(times.min(), times.max(), 1000)
Nes = np.array([dm.eta(T, Ne=True) for dm in results])
plt.plot(T, np.median(Nes, axis=0))
plt.xscale('log')
plt.yscale('log')
```

#### Held-out data

If provided with a held-out chromosome, `phlash` will use this to assess out-of-sample predictive performance and prevent overfitting. I recommend using this option if there is enough data. To do so, simply divide your contigs into a list of "training" contigs, plus one "test" contig, and specify them accordingly:

```python
test_data = chroms_1kg[0]
train_data = chroms_1kg[1:]
results = phlash.fit(data=train_data, test_data=test_data)
```

#### Specifying additional options
A number of options can be passed to `phlash.fit()` that affect the behavior of the algorithm. Documentation on these is currently a work in progress, however, most of them are accompanied by (hopefully) self-explanatory comments in the [source code](../src/phlash/mcmc.py#L33).


### Analyzing simulated data
To explore how `phlash` performs under various settings, it can be useful to run it on
simulated data. A convenience method is available to simulate data from the
[`stdpopsim` catalog](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html).


```python
import phlash.sim

sim_contigs = phlash.sim.stdpopsim_dataset(
    "HomSap",
    "Zigzag_1S14",
    {"generic": 100},
    options=dict(length_multiplier=0.1),
)
```

`sim_contigs` is a `dict` containing a `data` entry (list of `phlash` contigs) and a `truth` entry, containing the true demographic model. It can now be analyzed as above:

```python
test_k = list(sim_contigs["data"])[0]
test_data = sim_contigs["data"][test_k]
train_data = [v for k, v in sim_contigs["data"].items() if k != test_k]
results = phlash.fit(train_data, test_data, truth=sim_contigs["truth"], fold_sfs=False)
```
