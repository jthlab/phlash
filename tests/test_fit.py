from phlash.fit.ts import TreeSeqFitter

def test_fit_ts(sim_twopop):
    TreeSeqFitter([sim_twopop], mutation_rate=1e-8).fit()
