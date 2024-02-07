import phlash.data
import phlash.sim
import phlash.size_history


def test_stdpopsim_sim():
    res = phlash.sim.stdpopsim_dataset(
        "HomSap",
        "Zigzag_1S14",
        {"generic": 1},
        contigs=["1"],
        use_scrm=False,
        options={"length_multiplier": 0.01},
    )
    assert isinstance(res, dict)
    assert res.keys() == {"data", "truth"}
    assert isinstance(res["truth"], phlash.size_history.DemographicModel)
    d = res["data"]
    assert isinstance(d, dict)
    assert d.keys() == {"1"}
    assert isinstance(d["1"], phlash.data.Contig)
    assert d["1"].N == 2
    assert d["1"].L == 2489564
