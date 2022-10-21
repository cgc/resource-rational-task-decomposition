import pandas as pd
import numpy as np
from experiment.analyze import analyze_v1_11_tools as tools
from experiment.analyze import pred_optimal_choice
import scipy.stats

def test_lrtest():
    lrtestrv, llm0, llm1 = tools.ro.r('''
    x = rnorm(10)
    y = 2 * x + 2 + rnorm(length(x))
    df = data.frame(x=x, y=y)
    m0 = lm(y ~ 1, data=df)
    m1 = lm(y ~ x, data=df)
    return(list(as.data.frame(lrtest(m0, m1)), logLik(m0), logLik(m1)))
    ''')
    pylr = pred_optimal_choice.lrtest(llm0[0], llm1[0], dof=1)
    assert np.isclose(pylr.chisq, lrtestrv.rx2['Chisq'][1])
    assert np.isclose(pylr.pr_gt_chisq, lrtestrv.rx2['Pr(>Chisq)'][1])

def test_one_sided_bernoulli_quantile_MC():
    rng = np.random.RandomState(153737859)

    for _ in range(5):
        # Get our probs
        p = scipy.stats.beta.rvs(1, 1, size=300, random_state=rng)
        # Sample from them and compute a mean
        observed = scipy.stats.bernoulli.rvs(p, random_state=rng).mean()

        # Get a seed to make sure we're computing it consistently w/ and w/o broadcasting
        s = rng.randint(0, 2**30)
        r = pred_optimal_choice.one_sided_bernoulli_quantile_MC(p, observed, broadcast=False, seed=s)
        r_bc = pred_optimal_choice.one_sided_bernoulli_quantile_MC(p, observed, broadcast=True, seed=s)
        assert np.isclose(r.quantile, r_bc.quantile), (r.quantile, r_bc.quantile)
        assert .05 < r.quantile < .95, 'this is seed-dependent! but likely'
