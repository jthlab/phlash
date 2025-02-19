from collections.abc import Callable
from typing import NamedTuple

import demes
import jax.numpy as jnp
import msprime
import numpy as np
import scipy
from jax import jit, lax, vmap

from phlash.jax_ppoly import JaxPPoly
from phlash.memory import memory
from phlash.util import Pattern


def _expm1inv(x):
    # 1 / expm1(x)
    x_large = x > 10.0
    x_safe = jnp.where(x_large, 1.0, x)
    # x = id_print(x, what="x")
    return jnp.where(x_large, -jnp.exp(-x) / jnp.expm1(-x), 1.0 / jnp.expm1(x_safe))


class SizeHistory(NamedTuple):
    t: jnp.ndarray
    c: jnp.ndarray

    @property
    def M(self):
        assert len(self.t) == len(self.c)
        return len(self.t)

    @classmethod
    def random(cls, rng):
        log_dt, log_c = rng.normal(size=(2, 10))
        t = np.exp(log_dt).cumsum()
        t[0] = 0.0
        return cls(t=t, c=np.exp(log_c))

    def to_demes(self, deme_name: str = "pop") -> demes.Graph:
        b = demes.Builder()
        epochs = []
        for ti, Ne in zip(self.t, self.Ne):
            ti = float(ti)
            Ne = float(Ne)
            epochs.append(
                {
                    "end_time": ti,
                    "start_size": Ne,
                    "end_size": Ne,
                    "size_function": "constant",
                }
            )
        b.add_deme(deme_name, epochs=epochs[::-1])
        return b.resolve()

    def draw(self, ax=None, density: bool = False, c: float = 1.0, **kwargs) -> None:
        """Plot this size history onto provided/current axis.

        Args:
            ax: matplotlib axis on which to draw plot, or the current axis if None.
            density: If True, plot the coalescent density function. If False, plot
                     Ne(t).
        """
        if ax is None:
            import matplotlib.pyplot

            ax = matplotlib.pyplot.gca()
        if density:
            x = np.geomspace(self.t[1], 2.0 * self.t[-1], 1000)
            y = self.dens(x, c)
        else:
            x = self.t
            y = self.Ne
            # plot the last part as a point, but don't label it
            kw = dict(kwargs)
            kw["label"] = None
            kw["marker"] = "."
            ax.scatter(self.t[-1:], y[-1:], **kw)
            # some nice looking defaults
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlabel("Generations")
            ax.set_ylabel("$N_e$")
            kwargs.setdefault("drawstyle", "steps-post")
        ax.plot(x, y, **kwargs)

    @classmethod
    def default(cls, K: int) -> "SizeHistory":
        """Return a default model, with timepoints initialized to quantiles of the
        Exponential(1) distribution."""
        q = np.linspace(0, 1, K, endpoint=False)
        t = scipy.stats.expon.ppf(q)
        return cls(t=t, c=jnp.ones_like(t))

    @classmethod
    def from_pmf(cls, t, p):
        """Initialize a size history from a distribution function of coalescent times.

        Args:
            t: time points
            p: p[i] = probability of coalescing in [t[i], t[i + 1])
        """
        R = 0.0
        c = []
        eps = 1e-8
        for dt, p_i in zip(jnp.diff(t), p):
            denom = jnp.maximum(jnp.exp(-R) - p_i, eps)
            c_i = (-R - jnp.log(denom)) / dt
            c.append(c_i)
            R += c_i * dt
        # coalescent rate in last period is not identifiable from this data.
        c.append(c[-1])
        return cls(t=jnp.array(t), c=jnp.array(c))

    @property
    def Ne(self):
        return 1.0 / (2.0 * self.c)

    @property
    def K(self):
        return len(self.c)

    def to_pp(self) -> JaxPPoly:
        return JaxPPoly(c=jnp.array(self.c)[None], x=jnp.append(self.t, jnp.inf))

    @property
    def R(self):
        return self.to_pp().antiderivative()

    def surv(self) -> np.ndarray:
        "returns the survival function of the coalescence density"
        dt = jnp.diff(self.t)
        a = self.c[:-1] * dt
        H = a.cumsum()
        return jnp.append(jnp.exp(-H), 0.0)

    # TODO: eliminate calls to p_coal, just use .pi
    @property
    def pi(self):
        return self.p_coal()

    def p_coal(self) -> np.ndarray:
        "returns the coalescence density at each time"
        Ci = -jnp.diff(self.surv())
        return jnp.concatenate([1.0 - Ci.sum(keepdims=True), Ci])

    def __call__(self, x: float, Ne: bool = False) -> float:
        """Evaluate this function at the points x.

        Args:
            x: vector of points at which to evaluate eta.
            Ne: If False, return eta(x). If True, return 1 / (2 * eta(x)).

        Returns:
            The function value at each point x.
        """
        i = jnp.searchsorted(jnp.append(self.t, jnp.inf), x, "right") - 1
        if Ne:
            return 1.0 / 2.0 / self.c[i]
        return self.c[i]
        # return self.to_pp()(x)

    def density(self, c: float = 1.0) -> Callable[[float], float]:
        R = self.R
        return lambda x: c * self(x) * jnp.exp(-c * R(x))

    @property
    def sf(self) -> Callable[[float], float]:
        R = self.R
        return lambda x: jnp.exp(-R(x))

    @property
    def cdf(self) -> Callable[[float], float]:
        R = self.R
        return lambda x: -jnp.expm1(-R(x))

    def ect(self):
        "expected time to coalescence within each interval"
        c = self.c[:-1]
        c0 = jnp.isclose(c, 0)
        cinf = jnp.isinf(c) | (c > 100.0)
        c_safe = jnp.where(c0 | cinf, 1.0, c)
        t0 = self.t[:-1]
        t1 = self.t[1:]
        dt = t1 - t0
        # Always have to be careful with exp... NaNs in gradients
        e_coal_safe = 1 / c_safe + t0 - dt * _expm1inv(c_safe * dt)
        # e_coal_safe, *_ = id_print((e_coal_safe, c_safe, dt), what="ect_safe")
        e_coal = jnp.select(
            [c0, cinf],
            [
                (t0 + t1) / 2,
                t0,
            ],
            e_coal_safe,
        )
        e_coal = jnp.append(e_coal, self.t[-1] + 1.0 / self.c[-1])
        # expected coal time of zero messes things up
        e_coal = jnp.maximum(e_coal, 1e-20)
        return e_coal

    def quantile(self, q: float) -> float:
        "returns the time at which the coalescence density quantile is q"
        R = self.R

        def f(x):
            return -jnp.expm1(-R(x)) - q

        def g(_, tup):
            a, b = tup
            c = (a + b) / 2
            cond = f(c) > 0
            a = jnp.where(cond, a, c)
            b = jnp.where(cond, c, b)
            return (a, b)

        a = 0.0
        b = self.t[-1]
        # find b such that f(b) > 0
        b = lax.while_loop(lambda x: f(x) <= 0, lambda x: x * 1.1, b)
        # run a few bisection steps
        a, b = lax.fori_loop(0, 20, g, (a, b))
        return (a + b) / 2.0

    def balance(self) -> "SizeHistory":
        "returns the balance of the coalescence density at time t"
        t = [self.quantile(q) for q in np.linspace(0, 1, self.K, endpoint=True)]
        return SizeHistory(t=np.array(t), c=self(t))

    @property
    def mu(self):
        "Expected coalescent time according to this size history"
        return self.to_pp().exp_integral()

    def etjj(self, n: int):
        @vmap
        def f(k):
            return self.__class__(t=self.t, c=k * (k - 1) / 2 * self.c).mu

        return f(jnp.arange(2, n + 1))

    def etbl(self, n: int):
        W = _W_matrix(n)
        return W @ self.etjj(n)

    def tv(self, other: "SizeHistory", n: int = 1) -> float:
        "Total variation distance between coalescent densities"
        n *= 2  # diploid => haploid
        c = n * (n - 1) / 2
        t = jnp.array(sorted(set(self.t.tolist()) | set(other.t.tolist())))
        assert t[0] == 0.0
        taug = jnp.append((t[:-1] + t[1:]) / 2, t[-1] + 1.0)
        # align the two size histories
        eta1 = SizeHistory(t=t, c=c * self(taug))
        eta2 = SizeHistory(t=t, c=c * other(taug))
        R1 = eta1.R
        R2 = eta2.R
        return _tv(R1, R2)

    def hellinger(self, other: "SizeHistory") -> float:
        "Hellinger distance between coalescent densities"
        # place the two densities on the same set of breakpoints
        t = jnp.sort(jnp.concatenate([self.t, other.t]))
        eta1 = SizeHistory(t=t, c=self(t))
        eta2 = SizeHistory(t=t, c=other(t))
        R1 = eta1.R
        R2 = eta2.R
        return _hellinger(R1, R2)

    def squared_l2(self, other: "SizeHistory", t_max, log: bool = False) -> float:
        "L2 distance between N_self(t) and N_other(t) up to time t_max."
        t = jnp.concatenate([self.t, other.t])
        t = jnp.append(t, t_max)
        t = jnp.sort(t)
        if log:
            f = jnp.log
            g = jnp.exp
            t = f(t[2:])  # omit zero entries
        else:
            f = g = lambda x: x
        # both Ne functions will be constant on the unioned intervals, so we can call
        # them at the interval midpoint to get the correct value
        tmid = (t[:-1] + t[1:]) / 2.0
        Ne1, Ne2 = (h(g(tmid), Ne=True) for h in (self, other))
        s = (f(Ne1) - f(Ne2)) ** 2 * jnp.diff(t)
        s *= t[1:] <= f(t_max)
        return s.sum()

    def l2(self, other: "SizeHistory", t_max: float, log: bool = False) -> float:
        return jnp.sqrt(self.squared_l2(other, t_max, log))

    @classmethod
    def from_demography(cls, demo: msprime.Demography) -> "SizeHistory":
        """Instantiate size history from an msprime demography.

        Args:
            demo: The demography containing the size history.

        Notes:
            If the demography contains more than one population, an error is raised.
        """
        assert isinstance(demo, msprime.Demography)
        if demo.num_populations > 1:
            raise ValueError(
                "this method only works for demographies containing a single population"
            )
        dbg: msprime.DemographyDebugger = demo.debug()
        t_max = dbg.epoch_start_time.max()
        # this is pretty wasteful, but we mostly use this method for plotting...
        t = np.arange(1 + t_max)
        Ne = dbg.population_size_trajectory(steps=t).squeeze()
        i = np.insert(Ne[1:] != Ne[:-1], 0, True)
        return cls(t=t[i], c=1 / (2 * Ne[i]))


@jit
def _tv(R1, R2):
    v = vmap(_tv_helper, (1, 1, 0))(R1.c, R2.c, jnp.diff(R1.x))
    return 0.5 * v.sum()


def _tv_helper(ab1, ab2, T):
    "int_0^T |a1 exp(-(a1*t + b1)) - a2 exp(-(a2*t + b2))| dt"
    a1, b1 = ab1
    a2, b2 = ab2

    def I(a, b, U=jnp.inf):  # noqa: E743
        "int_0^T a exp(-(a*t + b)) dt"
        # works even if T=+oo assuming a>0 which it always is
        return jnp.exp(-b) * jnp.where(jnp.isinf(U), 1.0, -jnp.expm1(-a * U))

    a1_eq_a2 = jnp.isclose(a1, a2)
    a1_m_a2_safe = jnp.where(a1_eq_a2, 1.0, a1 - a2)

    t_star = jnp.clip((jnp.log(a1 / a2) + b2 - b1) / a1_m_a2_safe, 0.0, T)
    t_star = jnp.where(a1_eq_a2, 0.0, t_star)
    i1 = I(a1, b1, t_star)
    i2 = I(a2, b2, t_star)
    return abs(i1 - i2) + abs((I(a1, b1, T) - i1) - (I(a2, b2, T) - i2))


def _hellinger(R1, R2):
    return 1.0 - vmap(_hellinger_helper, (1, 1, 0))(R1.c, R2.c, jnp.diff(R1.x)).sum()


def _hellinger_helper(ab1, ab2, T):
    "int_0^T sqrt(a1 exp(-(a1*t + b1)) * a2 exp(-(a2*t + b2))) dt"
    a1, b1 = ab1
    a2, b2 = ab2
    # = sqrt(a1 * a2) * int_0^T exp(-0.5 * (a1 + a2) * t - 0.5 * (b1 + b2)) dt
    # = sqrt(a1 * a2) * exp(-0.5 * (b1 + b2)) * (1 - exp(-0.5 * (a1 + a2) * T))
    #   / (0.5 * (a1 + a2))
    T_safe = jnp.where(jnp.isinf(T), 1.0, T)
    r1 = jnp.sqrt(a1 * a2) * jnp.exp(-0.5 * (b1 + b2)) / (0.5 * (a1 + a2))
    r2 = jnp.where(jnp.isinf(T), 1.0, -jnp.expm1(-0.5 * (a1 + a2) * T_safe))
    return r1 * r2


def _psmc_size_history(pattern, alpha, t_max) -> SizeHistory:
    p = Pattern(pattern)
    N = p.M - 1
    beta = jnp.log1p(t_max / alpha) / N
    k = jnp.arange(N)
    t = jnp.append(alpha * jnp.expm1(beta * k), t_max)
    t = np.concatenate([[0.0], np.geomspace(1e-3, 15.0, p.M - 1)])
    return SizeHistory(t=t, c=jnp.ones(p.M))


class DemographicModel(NamedTuple):
    eta: SizeHistory
    theta: float
    rho: float

    @classmethod
    def default(
        cls, pattern: str, theta: float, rho: float = None, t_max: float = 15.0
    ):
        if rho is None:
            rho = theta
        # from PSMC. these defaults seem to work pretty well.
        eta = _psmc_size_history(pattern=pattern, alpha=0.1, t_max=t_max)
        return cls(eta=eta, theta=theta, rho=rho)

    def rescale(self, mu: float) -> "DemographicModel":
        """Rescale model so that the mutation rate per unit time is mu.

        Args:
            mu: The mutation rate per locus per generation.

        Returns:
            Rescaled demographic model.
        """
        # the rate of mutation per unit of time in our model is theta/2
        N1_N0 = (self.theta / 2) / mu
        t = N1_N0 * self.eta.t
        c = self.eta.c / N1_N0
        eta = SizeHistory(t=t, c=c)
        rho_sc = self.rho / N1_N0 if self.rho is not None else None
        return DemographicModel(theta=mu, rho=rho_sc, eta=eta)

    @property
    def M(self):
        return self.eta.M


@memory.cache
def _W_matrix(n: int) -> np.ndarray:
    from fractions import Fraction as mpq  # mimic gmpy2 api

    # IMPORTANT (DO NOT DELETE): this cast makes sure that n is a Python bignum, and not
    # a sneaky np.int64 in disguise. this matters because we need exact integer
    # arithmetic over an unbounded range in the code below.
    n = int(n)
    assert isinstance(n, int)

    # returns W matrix as calculated as eq 13:15 @ Polanski 2013
    # n: sample size
    if n == 1:
        return np.array([[]], dtype=np.float64)
    W = np.zeros(
        [n - 1, n - 1], dtype=object
    )  # indices are [b, j] offset by 1 and 2 respectively
    W[:, 2 - 2] = mpq(6, n + 1)
    if n == 2:
        return W.astype(np.float64)
    b = list(range(1, n))
    W[:, 3 - 2] = np.array([mpq(30 * (n - 2 * bb), (n + 1) * (n + 2)) for bb in b])
    for j in range(2, n - 1):
        A = mpq(-(1 + j) * (3 + 2 * j) * (n - j), j * (2 * j - 1) * (n + j + 1))
        B = np.array([mpq((3 + 2 * j) * (n - 2 * bb), j * (n + j + 1)) for bb in b])
        W[:, j + 2 - 2] = A * W[:, j - 2] + B * W[:, j + 1 - 2]
    return W.astype(np.float64)
