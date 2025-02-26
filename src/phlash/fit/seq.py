import numpy as np
from jax import vmap
from loguru import logger

import phlash.model

from ..afs import default_afs_transform
from .base import BaseFitter


class SequenceFitter(BaseFitter):
    def __init__(self, data, test_data=None, **options):
        self.num_samples = data[0].hets.shape[0]
        return super().__init__(data, test_data, **options)

    def load_data(self):
        # in the one population case, all the chunks are exchangeable, so we can just
        # merge all the chunks
        super().load_data()
        # convert afs to standard vector representation in the 1-pop case
        daft = self.options.get("afs_transform", default_afs_transform)
        if self.afs:
            self.afs = {k: v.todense()[1:-1] for k, v in self.afs.items()}
            self.afs_transform = {n: daft(self.afs[n]) for n in self.afs}
            for n in self.afs:
                logger.debug(
                    "transformed afs[{}]:{}", n, self.afs_transform[n](self.afs[n])
                )
        else:
            self.afs_transform = None
        if self.test_afs:
            self.test_afs = {k: v.todense()[1:-1] for k, v in self.test_afs.items()}
            self.test_afs_transform = {n: daft(self.test_afs[n]) for n in self.test_afs}
        else:
            self.test_afs_transform = None

        # massage the chunks
        self.chunks = self.chunks.reshape(-1, *self.chunks.shape[2:])[:, None]
        # if too many chunks, downsample so as not to use up all the gpu memory
        if self.num_chunks > 5 * self.minibatch_size * self.niter:
            # important: use numpy to do this _not_ jax. (jax will put it on the gpu
            # which causes the very problem we are trying to solve.)
            old_size = self.chunks.size
            rng = np.random.default_rng(np.asarray(self.get_key()))
            self.chunks = rng.choice(
                self.chunks, size=(5 * self.minibatch_size * self.niter,), replace=False
            )
            gb = 1024**3
            logger.debug(
                "Downsampled chunks from {:.2f}Gb to {:.2f}Gb",
                old_size / gb,
                self.chunks.size / gb,
            )
        logger.debug("after merging: chunks.shape={}", self.chunks.shape)

    def optimization_step(self, data, **kwargs):
        """
        Perform a single optimization step.
        """
        # Placeholder for actual optimization logic.
        inds = data
        kwargs["inds"] = inds
        kwargs["kern"] = self.train_kern
        kwargs["warmup"] = self.warmup_chunks[inds]
        kwargs["afs_transform"] = self.afs_transform
        return super().optimization_step(data, **kwargs)

    def log_density(self, particle, **kwargs):
        """
        Compute the log density.
        """

        @vmap
        def f(mcp, inds, warmup):
            return phlash.model.log_density(
                mcp=mcp,
                weights=kwargs["weights"],
                inds=inds,
                warmup=warmup,
                ld=kwargs.get("ld"),
                afs_transform=self.afs_transform,
                afs=kwargs.get("afs"),
                kern=kwargs["kern"],
                alpha=self.options.get("alpha", 0.0),
                beta=self.options.get("beta", 0.0),
                _components=kwargs.get("_components"),
            )

        inds = kwargs["inds"]
        warmup = kwargs["warmup"]
        mcps = vmap(lambda _: particle)(inds)
        return f(mcps, inds, warmup).sum(0)


def fit(data, test_data=None, **options):
    global _fitter
    _fitter = SequenceFitter(data, test_data, **options)
    return _fitter.fit()
