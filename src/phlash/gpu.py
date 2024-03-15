import ctypes
import os.path
import threading
import warnings
from functools import partial, singledispatchmethod

import jax
import jax.numpy as jnp
import numpy as np
import nvidia.cuda_nvrtc.lib
from cuda import cuda, cudart, nvrtc
from jax import custom_vjp, tree_map
from loguru import logger

import phlash.size_history
from phlash.params import PSMCParams


class CudaError(RuntimeError):
    def __init__(self, err):
        self.err = err
        RuntimeError.__init__(self, f"Cuda Error {self.err}: {self.name}")

    @property
    def name(self):
        _, name = cuda.cuGetErrorName(self.err)
        return name


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise CudaError(err)
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, name = nvrtc.nvrtcGetErrorString(err)
            raise RuntimeError(f"Nvrtc Error {err}: {name}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CudaRT Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def _compile(code: str, compute_capability: str) -> bytes:
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(code), b"kern.cu", 0, [], [])
    ASSERT_DRV(err)
    try:
        (err,) = nvrtc.nvrtcCompileProgram(
            prog, 1, [f"--gpu-architecture=sm_{compute_capability}".encode()]
        )
        ASSERT_DRV(err)
    except RuntimeError as e:
        # Get log from compilation
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        ASSERT_DRV(err)
        log = bytearray(logSize)
        ASSERT_DRV(err)
        (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode()) from e
    err, dataSize = nvrtc.nvrtcGetCUBINSize(prog)
    ASSERT_DRV(err)
    data = bytearray(dataSize)
    (err,) = nvrtc.nvrtcGetCUBIN(prog, data)
    ASSERT_DRV(err)
    return data


class CudaInitializer:
    """this class defers cuda initialization until runtime. that way, platforms that
    don't have cuda installed can still load the package."""

    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def initialize_cuda(cls):
        with cls._lock:
            if not cls._initialized:
                libnvrtc_path = os.path.join(
                    nvidia.cuda_nvrtc.lib.__path__[0], "libnvrtc.so.12"
                )
                # I think we need to hold a reference to this library until the next
                # call so that it can be found when cuInit calls dlopen
                libnvrtc = ctypes.CDLL(libnvrtc_path)  # noqa: F841
                (err,) = cuda.cuInit(0)
                ASSERT_DRV(err)
                cls._initialized = True
                err, device_num = cudart.cudaGetDevice()
                ASSERT_DRV(err)
                err, prop = cudart.cudaGetDeviceProperties(device_num)
                ASSERT_DRV(err)
                cls.compute_capability = f"{prop.major}{prop.minor}"
                logger.debug("Compute capability: {}", cls.compute_capability)


class _PSMCKernelBase:
    "PSMC kernel running on a single GPU"

    def __init__(self, M: int, data: jax.Array, double_precision: bool = False):
        CudaInitializer.initialize_cuda()
        assert data.ndim == 2
        assert data.dtype == np.int8
        assert data.min() >= -1
        data = data.clip(-1, 1)
        assert data.max() <= 1
        assert np.all(
            data.max(axis=1) > -1
        ), "data contains observations with all missing values"
        self.double_precision = double_precision
        self._N, self._L = data.shape
        # copy the data onto the gpu once and for all
        err, self._data_gpu = cuda.cuMemAlloc(data.nbytes)
        try:
            ASSERT_DRV(err)
        except CudaError as e:
            logger.debug(f"{data.shape=}")
            raise MemoryError(
                f"While trying to allocate {data.nbytes} bytes on GPU"
            ) from e
        (err,) = cuda.cuMemcpyHtoD(self._data_gpu, data, data.nbytes)
        ASSERT_DRV(err)
        # some more checks
        if M != 16:
            warnings.warn("Performance is optimized when M=16")
        self._M = M
        src = [f"#define M {M}"]
        if double_precision:
            src.append("typedef double FLOAT;")
        else:
            src.append("typedef float FLOAT;")
        src.append(KERNEL_SRC)
        src = "\n".join(src)
        cubin = _compile(src, CudaInitializer.compute_capability)
        err, self._mod = cuda.cuModuleLoadData(cubin)
        ASSERT_DRV(err)
        self._f = {}
        for k in "loglik", "loglik_grad":
            err, f = cuda.cuModuleGetFunction(self._mod, str.encode(k))
            ASSERT_DRV(err)
            self._f[k] = f

        # these will be dynamically allocated later
        self._inds_gpu = self._pa_gpu = self._ll_gpu = self._dlog_gpu = None
        # stream to enqueue commands
        err, self._stream = cuda.cuStreamCreate(0)
        ASSERT_DRV(err)

    def __del__(self):
        # if we die before the constructor has finished, some of these attributes
        # might not exist, so we use hasattr() everywhere.
        # FIXME: I see a weird bug where __del__ is called without initializing cuda.
        # I don't understand how this can happen since it's initialized at the top of
        # the file. everything is surrounded with try/catch blocks to guard against.
        if hasattr(self, "_mod"):
            (err,) = cuda.cuModuleUnload(self._mod)
            ASSERT_DRV(err)
        for a in (
            "_data_gpu",
            "_inds_gpu",
            "_pa_gpu",
            "_ll_gpu",
            "_dlog_gpu",
        ):
            if hasattr(self, a) and getattr(self, a) is not None:
                (err,) = cuda.cuMemFree(getattr(self, a))
                ASSERT_DRV(err)
        if hasattr(self, "_stream"):
            (err,) = cuda.cuStreamDestroy(self._stream)
            ASSERT_DRV(err)

    @property
    def float_type(self):
        if self.double_precision:
            return np.float64
        return np.float32

    def __call__(
        self, pp: PSMCParams, index: int, grad: bool, barrier: threading.Barrier
    ) -> tuple[float, PSMCParams]:
        M = self._M
        N = self._N
        added_S = False
        pa = np.stack(pp, -2)
        inds = np.atleast_1d(index)
        if index.ndim == 0:
            added_S = True
            assert pa.shape == (7, M)
            pa = pa[None]
        S = inds.shape[0]
        assert inds.shape == (S,)
        assert np.all(0 <= inds) & np.all(
            inds < self._N
        ), f"0 <= {inds.min()=} < {inds.max()=} < N"
        # pps should be a stack of params, one for each dataset
        added_B = False
        if pa.ndim == 2:
            assert pa.shape == (7, M)
            pa = np.repeat(pa[None, None], S, axis=1)
            assert pa.shape == (1, S, 7, M)
            added_B = True
        if pa.ndim == 3:
            assert pa.shape == (S, 7, M)
            pa = pa[None]
            added_B = True
        assert pa.ndim == 4
        B = pa.shape[0]
        assert pa.shape == (B, S, 7, M)
        assert np.isfinite(pa).all(), "not all parameters finite"
        dlog = np.zeros([B, S, 7, M], dtype=self.float_type)
        ll = np.zeros([B, S], dtype=np.float64)
        # copy in the needed arrays
        # TODO these memory allocations need only happen once, but they are
        # cheap compared to the kernel call
        # indices array
        inds = inds.astype(np.int64)
        if self._inds_gpu is None:
            err, self._inds_gpu = cuda.cuMemAlloc(inds.nbytes)
            ASSERT_DRV(err)
        # params array
        pa = pa.astype(self.float_type)
        if self._pa_gpu is None:
            err, self._pa_gpu = cuda.cuMemAlloc(pa.nbytes)
            ASSERT_DRV(err)
        # loglik array
        if self._ll_gpu is None:
            err, self._ll_gpu = cuda.cuMemAlloc(ll.nbytes)
            ASSERT_DRV(err)
        # dloglik array
        if self._dlog_gpu is None:
            err, self._dlog_gpu = cuda.cuMemAlloc(dlog.nbytes)
            ASSERT_DRV(err)
        # asynchronously copy them in
        (err,) = cuda.cuMemcpyHtoDAsync(self._inds_gpu, inds, inds.nbytes, self._stream)
        ASSERT_DRV(err)
        (err,) = cuda.cuMemcpyHtoDAsync(self._pa_gpu, pa, pa.nbytes, self._stream)
        ASSERT_DRV(err)
        (err,) = cuda.cuMemcpyHtoDAsync(self._ll_gpu, ll, ll.nbytes, self._stream)
        arg_values = (
            self._data_gpu,
            np.int64(self._L),
            np.int64(N),
            self._inds_gpu,
            self._pa_gpu,
            self._ll_gpu,
        )
        arg_types = (None, ctypes.c_int64, ctypes.c_int64, None, None, None)
        if grad:
            f = self._f["loglik_grad"]
            (err,) = cuda.cuMemcpyHtoDAsync(
                self._dlog_gpu, dlog, dlog.nbytes, self._stream
            )
            ASSERT_DRV(err)
            arg_values += (self._dlog_gpu,)
            arg_types += (None,)
            grid = (B, S, 1)
            block = (7, M, 1)
        else:
            f = self._f["loglik"]
            grid = (B, 1, 1)
            block = (S, 1, 1)
        (err,) = cuda.cuStreamSynchronize(self._stream)
        ASSERT_DRV(err)
        logger.trace("launching kernel in thread={}", threading.get_ident())
        (err,) = cuda.cuLaunchKernel(
            f,
            *grid,
            *block,
            0,  # dynamic shared memory
            self._stream,  # current stream
            (arg_values, arg_types),
            0,
        )
        ASSERT_DRV(err)

        # all threads wait before sync -- without this I was seeing a problem where
        # GPU execution proceeded sequentially across threads -- it's as though
        # streamSynchronize does not release the GIL, though from inspecting the
        # source, it does? idk
        barrier.wait()

        (err,) = cuda.cuStreamSynchronize(self._stream)
        ASSERT_DRV(err)

        ASSERT_DRV(err)
        (err,) = cuda.cuMemcpyDtoHAsync(ll, self._ll_gpu, ll.nbytes, self._stream)
        ASSERT_DRV(err)
        if grad:
            (err,) = cuda.cuMemcpyDtoHAsync(
                dlog, self._dlog_gpu, dlog.nbytes, self._stream
            )
            ASSERT_DRV(err)
        (err,) = cuda.cuStreamSynchronize(self._stream)
        ASSERT_DRV(err)
        if grad:
            dll = PSMCParams(
                b=dlog[..., 0, :],
                d=dlog[..., 1, :],
                u=dlog[..., 2, :],
                # we have to roll the last axis around by 1 because v is stored
                # in positions [1, ..., M-1] in the input array.
                v=np.roll(dlog[..., 3, :], 1, axis=-1),
                emis0=dlog[..., 4, :],
                emis1=dlog[..., 5, :],
                pi=dlog[..., 6, :],
            )
            ret = (ll, dll)
        else:
            ret = ll
        # strip off the additional batch dimensions we added earlier
        if added_B and added_S:
            ret = jax.tree_map(lambda a: a[0, 0, ...], ret)
        elif added_B:
            ret = jax.tree_map(lambda a: a[0, ...], ret)
        elif added_S:
            ret = jax.tree_map(lambda a: a[:, 0, ...], ret)
        return ret


class PSMCKernel:
    """Spread kernel evalution across multiple devices.

    Args:
        - M: discretization level. should always be 16.
        - data: data matrix.
        - double_precision: if True, use float64 on the GPU. (generally not worth it in
            my experience.)
    """

    def __init__(self, M, data, double_precision=False, num_gpus: int = None):
        CudaInitializer.initialize_cuda()
        if num_gpus is not None:
            assert num_gpus > 0
        self.double_precision = double_precision
        self.devices = self._initialize_devices(num_gpus)
        self.M = M

        # Initialize per-GPU resources
        self.gpu_kernels = []
        for device in self.devices:
            (err,) = cudart.cudaSetDevice(device)
            ASSERT_DRV(err)
            self.gpu_kernels.append(_PSMCKernelBase(M, data, double_precision))

    @property
    def float_type(self):
        if self.double_precision:
            return np.float64
        return np.float32

    @singledispatchmethod
    def loglik(self, pp: PSMCParams, index: int):
        log_params = tree_map(jnp.log, pp)
        return _psmc_ll(log_params, index=index, kern=self)

    # convenience overload mostly to help test code
    @loglik.register
    def _(self, dm: phlash.size_history.DemographicModel, index):
        return self.loglik(PSMCParams.from_dm(dm), index)

    def _initialize_devices(self, num_gpus: int):
        """
        Detect and initialize available GPUs.
        """
        err, n = cuda.cuDeviceGetCount()
        ASSERT_DRV(err)
        if num_gpus is not None:
            assert num_gpus > 0
            n = min(num_gpus, n)
        logger.info("Using {} GPUs", n)
        ret = []
        for i in range(n):
            err, cuDevice = cuda.cuDeviceGet(i)
            ASSERT_DRV(err)
            ret.append(cuDevice)
        return ret

    def __call__(
        self, pp: PSMCParams, index: int, grad: bool
    ) -> tuple[float, PSMCParams]:
        def f(a):
            assert np.isfinite(a).all()

        try:
            jax.tree_map(f, pp)
        except Exception:
            logger.debug("pp:{}", pp)
            raise
        # Split indices array across GPUs
        indices = np.atleast_1d(index)
        D = len(self.devices)
        split_indices = np.array_split(indices, D)
        threads = []
        assert D == len(self.gpu_kernels)
        results = [None] * D
        # FIXME which if |indices| < |gpus|
        barrier = threading.Barrier(D)
        for i, (device, gpu_kernel, split_index) in enumerate(
            zip(self.devices, self.gpu_kernels, split_indices)
        ):
            thread = threading.Thread(
                target=self._compute_on_gpu,
                args=(i, device, gpu_kernel, pp, split_index, grad, barrier, results),
            )
            threads.append(thread)
            thread.start()
            logger.trace("spawned thread {}", thread)

        for thread in threads:
            thread.join()

        ret = self._combine_results(results)
        if index.ndim == 0:
            ret = jax.tree_map(np.squeeze, ret)
        return ret

    def _combine_results(self, results):
        """
        Combine results from all GPUs.
        """
        return jax.tree_map(lambda *x: np.concatenate(x), *results)

    def _compute_on_gpu(
        self, i, device, gpu_kernel, pp, split_index, grad, barrier, results
    ):
        """
        Perform computation on a specific GPU.
        """
        cudart.cudaSetDevice(device)
        results[i] = gpu_kernel(pp, split_index, grad, barrier)


@partial(custom_vjp, nondiff_argnums=(2,))
def _psmc_ll(log_params: PSMCParams, index, kern) -> float:
    return _psmc_ll_helper(log_params, index=index, kern=kern, grad=False)


def _psmc_ll_fwd(log_params, index, kern):
    return _psmc_ll_helper(log_params, index=index, kern=kern, grad=True)


def _psmc_ll_helper(log_params: PSMCParams, index, kern, grad):
    params = tree_map(jnp.exp, log_params)
    result_shape_dtype = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64)
    if grad:
        result_shape_dtype = (
            result_shape_dtype,
            PSMCParams(
                *[
                    jax.ShapeDtypeStruct(shape=(params.M,), dtype=kern.float_type)
                    for p in params
                ],
            ),
        )
    return jax.pure_callback(
        kern, result_shape_dtype, pp=params, index=index, grad=grad, vectorized=True
    )


def _psmc_ll_bwd(kern, df, g):
    return tree_map(lambda a: g * a, df), None


_psmc_ll.defvjp(_psmc_ll_fwd, _psmc_ll_bwd)


KERNEL_SRC = r"""
// Shorthands to index into the global gradient array of shape [M, M, 6]
// I experimented with all memory layouts and this one results in the most
// coalesced writes
// Other accessors
typedef signed char int8_t;
typedef long long int64_t;

#define P 7

#define LOG_B 0
#define LOG_D 1
#define LOG_U 2
#define LOG_V 3
#define LOG_E0 4
#define LOG_E1 5
#define LOG_PI 6

#define LOG_X(m, i)   H[i * M * P + m * P + g]  // [M,M,7]
#define LOG_X_STRIDE  M * P

// Shorthands to index into the global params array
#define B(m) p[0 * M + m]
#define D(m) p[1 * M + m]
#define U(m) p[2 * M + m]
#define V(m) p[3 * M + m]
#define EMIS(ob, m) p[(4 + ob) * M + m]
#define PI(m) p[6 * M + m]

template<typename T>
__device__ void matvec(T *v, FLOAT *p, int stride = 1) {
    // in-place O(M) matrix-vector multiply with transition matrix
    FLOAT tmp[M];
    FLOAT x, sum;
    int i;
    sum = 0.;
    for (i = 0; i < M; ++i) {
        x = *(v + stride * i);
        tmp[i] = x * D(i) + sum * V(i);
        sum += U(i) * x;
    }
    sum = 0.;
    for (i = M - 1; i >= 0; --i) {
        x = *(v + stride * i);
        *(v + stride * i) = tmp[i] + sum * B(i);
        sum += x;
    }
}

__device__ FLOAT p_emis(const int8_t ob, FLOAT *p, const int m) {
    if (ob == -1) return 1.;
    return EMIS(ob, m);
}

extern "C"
__global__ void
// log-likelihood function without gradient
loglik(int8_t const *datag,
       const int64_t L,
       const int64_t N,
       const int64_t *inds,
       FLOAT const *pg,  // [B, P, M]
       double *loglik
       ) {
    const int64_t b = blockIdx.x;
    const int64_t S = blockDim.x;
    const int64_t s = threadIdx.x;

    __shared__ FLOAT p[P * M];
    FLOAT h[M], c;
    int m;

    // copy local global parameters to local
    FLOAT const *pgb = &pg[b * S * P * M + s * P * M];
    if (s == 0) {
        memcpy(p, pgb, P * M * sizeof(FLOAT));
    }
    __syncthreads();
    for (m = 0; m < M; ++m) {
        h[m] = PI(m);  // initialize to pi
    }
    double ll = 0.;
    // local variables
    const int8_t *data = &datag[inds[s] * L];
    int8_t ob;
    int64_t ell;
    for (ell = 0; ell < L; ell++) {
        matvec(h, p);
        ob = data[ell];
        c = 0.;
        for (m = 0; m < M; ++m) {
            h[m] *= p_emis(ob, p, m);
            c += h[m];
        }
        for (m = 0; m < M; ++m) h[m] /= c;
        ll += log(c);
    }
    loglik[b * S + s] = ll;
}

extern "C"
__global__ void
__launch_bounds__(7 * M)
// value and gradient of the log-likelihood function
loglik_grad(int8_t const *datag,
          const int64_t L,
          const int64_t N,
          const int64_t *inds,
          FLOAT const *pg,
          double *loglik,
          FLOAT *dlog
         ) {
    const int64_t b = blockIdx.x;
    const int64_t s = blockIdx.y;
    const int64_t S = gridDim.y;
    const int g = threadIdx.x;
    const int m = threadIdx.y;

    __shared__ FLOAT H[P * M * M];
    __shared__ FLOAT p[P * M];
    __shared__ FLOAT h[M];
    __shared__ FLOAT c;
    FLOAT ll;

    // copy local global parameters to local
    FLOAT const *pgb = &pg[b * S * P * M + s * P * M];
    if (g == 0) {
        if (m == 0) {
            memcpy(p, pgb, P * M * sizeof(FLOAT));
            memset(H, 0., sizeof(FLOAT) * P * M * M);
            c = 0.;
            ll = 0.;
        }
        h[m] = PI(m);
    }
    __syncthreads();
    // initialize the H matrix for pi to identity; all others zero.
    // (and technically, it's the gradient w/r/t pi not log_pi)
    LOG_X(m, m) = FLOAT(g == LOG_PI);
    __syncthreads();
    if (g == 0) h[m] = PI(m);  // initialize to pi
    __syncthreads();
    // local variables
    int8_t ob;
    int i, j;
    FLOAT tmp;
    ll = 0.;
    FLOAT sum1, sum2;
    int start, delta;
    if (g == LOG_B) {
        start = M - 1;
        delta = -1;
    } else {
        start = 0;
        delta = +1;
    }
    // main data loop
    const int8_t *data = &datag[inds[s] * L];
    int64_t ell;
    c = 0.;
    __syncthreads();
    for (ell = 0; ell < L; ell++) {
        // read chunks of the data from coalesced global memory
        ob = data[ell];
        // update each derivative matrix
        matvec(&LOG_X(m, 0), p, LOG_X_STRIDE);
        sum1 = 0.;
        sum2 = 0.;
        for (j = start; j >= 0 && j < M; j += delta) {
            // B counts down and everything else counts up
            tmp = (
                      // diag(hr * b)
                      (g == LOG_B) * (m == j) * (sum1 * B(j)) +
                      // diag(h * d)
                      (g == LOG_D) * (m == j) * (D(j) * h[j]) +
                      //
                      (g == LOG_U) * (m < j) * (h[m] * U(m) * V(j)) +
                      //
                      (g == LOG_V) * (j - 1 == m) * (sum2 * V(j))
            );
            sum1 += h[j];
            sum2 += U(j) * h[j];
            LOG_X(m, j) += tmp;
        }
        __syncthreads();
        if (g == 0 && m == 0) matvec(h, p);
        __syncthreads();
        LOG_X(m, m) += (g == LOG_E0) * h[m] * (ob == 0);
        LOG_X(m, m) += (g == LOG_E1) * h[m] * (ob == 1);
        __syncthreads();
        if (g == 0) {
            h[m] *= p_emis(ob, p, m);
            atomicAdd(&c, h[m]);
        }
        __syncthreads();
        if (g == 0) {
            h[m] /= c;
        }
        for (j = 0; j < M; ++j) {
            LOG_X(m, j) *= p_emis(ob, p, j) / c;
        }
        __syncthreads();
        if (g == 0 && m == 0) {
            ll += log(c);
            c = 0.;
        }
        __syncthreads();
    }
    if (g == 0 && m == 0) {
        loglik[b * S + s] = ll;
    }
    // for pi we accumulated dll/dpi instead of dll/dlog(pi) so
    // we have to multiply by
    const FLOAT x = (g == LOG_PI) ? PI(m) : 1.;
    for (i = 0; i < M; ++i) {
        dlog[b * S * P * M + s * P * M + g * M + m] += LOG_X(m, i) * x;
    }
}
"""
