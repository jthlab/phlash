import os

import numpy as np

from eastbay.data import VcfDataset

# from eastbay.data import ChunkedData, _chunk, stdpopsim_dataset


def test_chunk_nopad():
    data = np.arange(36).reshape(3, 12)
    chunked = _chunk(data, 5, 2, False)
    np.testing.assert_allclose(chunked, [[d[:7], d[5:]] for d in data])


def test_chunk_nopad_uneven():
    data = np.arange(27).reshape(9, 3)
    chunked = _chunk(data, 2, 1, False)
    assert chunked.shape == (9, 1, 3)
    np.testing.assert_allclose(chunked[:, 0], data)


def test_chunk_pad():
    data = np.arange(16).reshape(4, 4)
    chunked = _chunk(data, 2, 1, True)
    np.testing.assert_allclose(chunked, [[d[:3], np.append(d[2:], -1)] for d in data])


def test_theta(rng):
    data = (rng.uniform(size=(11, 7)) > 0.2).astype(int)
    overlap = 2
    chunks = _chunk(data, 4, overlap, True)
    ch = ChunkedData(chunks, overlap=overlap, afs=None)
    theta1 = ch.theta
    d = chunks[:, overlap:]
    theta2 = (d == 1).sum() / (d != -1).sum()
    np.testing.assert_allclose(theta1, theta2)


def test_stdpopsim():
    truth, ts = stdpopsim_dataset("SouthMiddleAtlas_1D17", "SouthMiddleAtlas")


def test_vcfdataset():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "sample.bcf")
    vcf = VcfDataset(fn, "1", (25_000_000, 26_000_000), ["NA12878", "NA12889"])
    d = vcf.get_data()
    assert d["het_matrix"].max() == 2
    assert d["het_matrix"].sum() == 256
    assert np.all(d["afs"] == [143, 60, 89])
