---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext nb_black
```

This document explains how to run <code>phlash</code>. Before doing so, please ensure that your system meets the [requirements](../README.md) and that you have installed the package.

## If you're coming from PSMC
If you are familiar with the PSMC software, you may find it helpful to know that `phlash` is conceptually similar to PSMC, but with a few key differences:

- `phlash` does not (yet) have a command-line interface. It is a Python package that is imported and used in a Python script or Jupyter notebook.
- `phlash` is a Bayesian method that estimates a posterior distribution over demographic models, rather than a point estimate. This means that the output of `phlash` is a list of demographic models, each of which is a valid sample from the posterior distribution.
- `phlash` does not have a proprietary data format. It can read data from VCF/BCF files or tree sequences, and can be extended to read other formats.

If you already have .psmcfa files (generated using i.e. the `fq2psmcfa` utility), a convenience function is provided for reanalyzing them with `phlash`:

```python
import phlash
posterior_samples = phlash.psmc(['/path/to/file1.psmcfa', '/path/to/file2.psmcfa', ...])
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
<code>phlash</code> intentionally does not have its own proprietary data format. Data can be loaded natively in using either VCF/BCF- or TreeSequence-formatted files.

<!-- #region -->
#### Loading VCF data


For example, to load data for sample `NA12878` from the first ten megabases of chromosome 22 in 1000 Genomes Phase 3 data release, execute the following:
<!-- #endregion -->

```python
import os.path

onekg_base = "/scratch/1kg"  # update with path on your local system
template = (
    "ALL.{chrom}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
)

chr22_path = os.path.join(onekg_base, template.format(chrom="chr22"))
chr22_c = phlash.contig(chr22_path, samples=["NA12878"], region="22:1-10000000")
```

To load data from all the autosomes, simply repeat this command for each of them:

```python
chroms_1kg = []
for chrom in range(1, 23):
    path = os.path.join(onekg_base, template.format(chrom=f"chr{chrom}"))
    chroms_1kg.append(
        phlash.contig(path, samples=["NA12878"], region=f"{chrom}:1-10000000")
    )
```

Notice that, to prevent inadvertent errors (such as the inclusion of telomeric regions into the analysis), the `samples=` and `region=` argument are required when loading VCF data. If you forget to provide them, the function will throw an error.


#### Reading tree sequence data

`phlash.contig()` also supports natively reading variants from tree sequences. For example, to load data for the first individual (represented as nodes `(0,1)`) in the [Wohns et al. inferred tree sequences](https://zenodo.org/records/5512994), execute the following commands:

```python
import glob

unified_base = "/scratch/unified"  # update with path on your local system
pattern = "hgdp_tgp_sgdp_high_cov_ancients_chr*_?.dated.trees.tsz"

chroms_ts = []

for chrom in glob.glob(os.path.join(unified_base, pattern)):
    chroms_ts.append(phlash.contig(chrom, samples=[(0, 1)]))
```

This cell takes longer to run, because `phlash` has to decompress and load each tree sequence, and also consumes more memory. (Note that `phlash.contig()` supports either tszipped or raw tree sequence files, or instantiated `TreeSequence` objects.)

If memory consumption is a limiting factor on your machine, you may consider using [`TreeSequence.simplify()`](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.simplify) to subset the data before loading.


#### Rolling your own data

For use cases that are not covered here, you may directly import your own data using lower-level classes that are built into `phlash`. See [contig.py](../src/phlash/contig.py) for more information.


### Fitting the model

Estimation is performed using `phlash.fit()`. In the most basic use-case, it takes a list of contigs and fits the model:

```python
results = phlash.fit(chroms_1kg)
```

The output of `fit()` is a list of `phlash.size_history.DemographicModel` classes. These are simple [named tuples](https://docs.python.org/3/library/collections.html#collections.namedtuple) with fields `theta`, `rho`, and `eta`. The latter is itself an instance of `phlash.size_history.SizeHistory`, which represents a piecewise-constant size history function.

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
To explore how `phlash` performs under various settings, it can be useful to run it on simulated data. `phlash` can already natively import the results of `msprime` simulations (since they are TreeSequences; see above). A convenience method is also available to simulate data from the [`stdpopsim` catalog](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html).


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
