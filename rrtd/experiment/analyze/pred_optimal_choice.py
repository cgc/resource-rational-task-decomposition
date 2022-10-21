from scipy.stats.distributions import chi2
import scipy.stats
import scipy.optimize
import scipy.special
import numpy as np
from functools import cached_property
from experiment.analyze import analyze_v1_11_tools as tools
import run_models
import types

def lrtest(L1, L2, *, dof):
    # https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
    def likelihood_ratio(llmin, llmax):
        assert llmin < llmax
        return(2*(llmax-llmin))
    LR = likelihood_ratio(L1,L2)
    p = chi2.sf(LR, dof)
    import types
    return types.SimpleNamespace(
        chisq=LR,
        pr_gt_chisq=p,
    )

def one_sided_bernoulli_quantile_MC(bernoulli_ps, observed, *, n_samples=10_000, broadcast=False, seed=None, log=True):
    rng = np.random.RandomState(seed)
    dist = scipy.stats.bernoulli(bernoulli_ps)
    if broadcast:
        samples = dist.rvs(size=(n_samples, len(bernoulli_ps)), random_state=rng).mean(axis=1)
    else:
        samples = np.array([dist.rvs(random_state=rng).mean() for _ in range(n_samples)])
    assert samples.shape == (n_samples,)

    quantile = (observed > samples).sum() / n_samples
    if log:
        print(f'observed is larger than {(observed > samples).sum()} / {n_samples} samples -- percentile={100*quantile:.02f}%')
        import matplotlib.pyplot as plt
        plt.hist(samples, bins=50)
        plt.axvline(observed, ls='--', c='r')
    return types.SimpleNamespace(
        quantile=quantile,
        mean_samples=np.mean(samples),
        var_samples=np.var(samples),
    )

def minimize(obj, *args, progress=10, **kwargs):
    i = 0
    def fn(x):
        nonlocal i
        i += 1
        if i % progress == 0:
            print(f'iter={i}, cost={obj(x):.2f}, x={x}')
    return scipy.optimize.minimize(obj, *args, callback=fn, **kwargs)

def minimize_from_ll(ll, *args, ll_kw={}, **kwargs):
    # HACK: dividing since that makes optimization more stable
    obj = lambda x: -ll(x, **ll_kw)/len(getattr(ll, 'nchoices', None) or ll_kw['choice_rows'])
    res = minimize(obj, *args, **kwargs)
    import types
    dof = args[0].shape[0]
    ll = ll(res.x, **ll_kw)
    return types.SimpleNamespace(
        result=res,
        ll=ll,
        dof=dof,
        aic=2 * dof - 2 * ll,
    )

class PredictOptimalChoice(object):
    def __init__(self, exp):
        self.exp = exp

    @cached_property
    def optimal_path_choice_df(self):
        return self.exp.optimal_path_choice_df()

    def multinomial_model(self, copy, *, rpar=False):
        r = tools.ro.r
        df = self.optimal_path_choice_df.df
        df = df[df[f'has_probe_choice_{copy}']]

        col = f'probe_choice_match_{copy}'
        rdf = r.convert_rdf_to_dfidx2(tools.convert_df_to_rdf(df[['id', 'chid', 'alt', 'choice', col]]))

        m = r.runmlogit(rdf, r.c(col), rpar=r.c(f'{col}TRUE') if rpar else False)
        r.show(r.summary(m))
        mnull = r.runmlogit(rdf, r.c(), rpar=False)
        r.show(r.lrtest(mnull, m))

    def mc_test(self, copy, *, seed=None):
        df = self.optimal_path_choice_df.df
        df = df[df[f'has_probe_choice_{copy}']]
        col = f'probe_choice_match_{copy}'

        chance = []
        ct = 0

        rows_by_chid = list(df.groupby('chid'))
        for chid, rows in rows_by_chid:
            if (rows[col] & rows.choice).sum():
                ct += 1
            chance.append(rows[col].mean())
        chance = np.array(chance)

        observed = ct/len(rows_by_chid)
        r = one_sided_bernoulli_quantile_MC(chance, observed, n_samples=100000, log=True, seed=seed)
        # variance of average -> variance of sum / length -> 1/length**2 * variance of sum
        # this is a https://en.wikipedia.org/wiki/Poisson_binomial_distribution
        # https://en.wikipedia.org/wiki/Variance#Basic_properties
        print('observed', observed, 'expected mean', np.mean(chance), 'expected variance', np.sum(chance * (1-chance)) / (len(chance)**2))
        print('MC-sampled mean', r.mean_samples, 'MC-sampled variance', r.var_samples)
        return r.quantile

    def make_choice_model_ll_fn(self, model):
        df = self.optimal_path_choice_df.df

        mdp_idx_to_model_scores = {
            mdp_idx: model.model(mdp)['scores']
            for mdp_idx, mdp in enumerate(self.exp.mdps())
        }

        # To avoid the perf implications of reaching into the DF on every loop, I pull out these
        # attributes here.
        choice_rows = [
            dict(
                mdp_idx=rows.mdp_idx.values[0],
                oh=np.stack(rows.alt_onehot.values),
                choice_idx=np.where(rows.choice)[0][0],
            )
            for chid, rows in df.groupby('chid')
        ]

        def log_likelihood(model_score_beta, sg_choice_score, *, mdp_idx_to_model_scores, choice_rows):
            ll = 0
            mdp_idx_to_sg_prob = {
                mdp_idx: scipy.special.softmax(model_score_beta * model_scores)
                for mdp_idx, model_scores in mdp_idx_to_model_scores.items()
            }
            for d in choice_rows:
                choice_among_paths_per_sg = scipy.special.softmax(sg_choice_score * d['oh'], axis=0)
                sg_prob = mdp_idx_to_sg_prob[d['mdp_idx']]
                p_path = choice_among_paths_per_sg @ sg_prob
                assert np.isclose(p_path.sum(), 1)
                ll += np.log(p_path[d['choice_idx']])
            return ll

        return log_likelihood, dict(choice_rows=choice_rows, mdp_idx_to_model_scores=mdp_idx_to_model_scores), [(0, None), (0, None)]
