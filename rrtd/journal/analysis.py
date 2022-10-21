import re
import os
import importlib
from collections import namedtuple
from types import SimpleNamespace
import sys
sys.path.append('../experiment/analyze')
import analyze_v1_11_tools as tools1_11


def pvalue(pv):
    return f'p {tools1_11.apa_pval(pv)}'


def savefig(fn, *args, **kwargs):
    dirname = f'figures'
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    import matplotlib.pyplot as plt

    fn, ext = os.path.splitext(fn)
    exts = [ext]
    if ext in ['', '.*']:
        exts = ['.png', '.eps', '.pdf']
    for ext in exts:
        plt.savefig(f'{dirname}/{fn}{ext}', *args, bbox_inches='tight', dpi=300, **kwargs)


def savesvg(fn, svg, *args, **kwargs):
    from IPython.display import SVG
    dirname = f'figures'
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    orig_svg = svg

    if hasattr(svg, '_repr_image_svg_xml'):
        svg = getattr(svg, '_repr_image_svg_xml')()
    elif isinstance(svg, SVG):
        svg = svg.data

    fn_no_ext, ext = os.path.splitext(fn)
    assert ext in ['.svg']

    svg_file = f'{dirname}/{fn}'
    with open(svg_file, 'w') as f:
        f.write(svg)

    if isinstance(orig_svg, SVG):
        import subprocess
        # subprocess.check_call(['rsvg-convert', svg_file, '-o', f'{dirname}/{fn_no_ext}.pdf'])
        subprocess.check_call(['inkscape', svg_file, '-o', f'{dirname}/{fn_no_ext}.pdf'])
    else:
        with open(f'{dirname}/{fn_no_ext}.pdf', 'wb') as f:
            f.write(orig_svg.pipe(format='pdf'))


_ModelEntryBase = namedtuple('_ModelEntryBase', ['key', 'name', 'color'])
class ModelEntry(_ModelEntryBase):
    @property
    def factors(self):
        if self.key is None:
            return ()
        else:
            return (self.key,)

    @property
    def model(self):
        if self.key is None:
            return
        import run_models
        return getattr(run_models, self.key, None)

models = [
    ModelEntry('iddfs', 'RRTD-IDDFS', None),
    ModelEntry('bfs', 'RRTD-BFS', None),
    ModelEntry('rw', 'RRTD-RW', None),
    ModelEntry('solwayr_partition_greedy_uniformexit_require_all_partitions_have_non_exit', 'Solway et al. (2014)', None),
    ModelEntry('tomov', 'Tomov et al. (2020)', None),
    ModelEntry('qcut', 'QCut', None),
    ModelEntry('log_degree_centrality', 'Degree Cent. (log)', None),
    ModelEntry('log_betweenness_centrality', 'Betweenness Cent. (log)', None),
    ModelEntry('log_stateocc', 'State Occupancy (log)', None),
    ModelEntry(None, 'Random Choice', None),
]

supp_rw_models = [
    ModelEntry('qcut', 'QCut', None),
    ModelEntry('log_degree_centrality', 'Degree Cent. (log)', None),
] + [
    ModelEntry(f'rw_spectral_rank{rank}', f'RRTD-RW (rank {rank})', None)
    for rank in [1, 2, 3, 4, 5, 6, 7]
] + [
    ModelEntry('rw', 'RRTD-RW (full rank)', None),
]

models_ns = SimpleNamespace(**{
    m.key or m.name: m for m in models})

models_without_random = [m for m in models if m.name != 'Random Choice']
models_without_random_or_fixed = [m for m in models if m.model]

ProbeType = namedtuple('ProbeType', ['key', 'verbose', 'name', 'random_effects'])
probes = [
    ProbeType('subgoal', '"What location would you set as a subgoal?"', 'Explicit Probe', True),
    ProbeType('solway2014', '"Choose a location you would visit along the way."', 'Implicit Probe', True),
    ProbeType('busStop', 'Instant Teleportation', 'Teleportation Question', False),
]

def load_exp1(*, nodraw=False):
    version = '1.14'
    tools = importlib.import_module(f'analyze_v{version.replace(".", "_")}_tools')
    rawexp = tools.ExperimentData.init(version)
    included = rawexp.navigation_performance_filter(cutoff=1.75, exclude_len1=True)
    if nodraw:
        included &= rawexp.nodraw_filter(look_never=True)
    exp = rawexp.with_filtered_participants(included)
    exp._rawexp = rawexp
    return exp


class FeatureSelection:
    def __init__(self, mlogits):
        self.mlogits = mlogits

    @classmethod
    def run_analysis(cls, exp):
        from tqdm.auto import tqdm
        r = tools1_11.ro.r

        exp_r = tools1_11.multinomial_data_for_r(exp, modelkeys=[m.key for m in models_without_random_or_fixed], tqdm=tqdm)

        # NOTE we visit models in reverse order to ensure we fit for Random Choice first. Important b/c
        # we need it for LR tests that others models do.
        model_order = models[::-1]
        assert model_order[0].name == 'Random Choice'

        mlogits = {}
        mlogits_full = {}
        for ptype in probes:
            print(ptype)
            mdict = mlogits[ptype.key] = {}
            mdict_full = mlogits_full[ptype.key] = {}
            for mtype in tqdm(model_order):
                m = r.runmlogit(exp_r[ptype.key], r.c(*mtype.factors), rpar=ptype.random_effects)
                mdict_full[mtype.key] = m
                lrtest_models = {}
                # NOTE: we only do lrtests against null model
                if mtype.key is not None:
                    assert None in mdict_full, 'assuming random is first, if this fails, check order of analysis.models'
                    lrtest_models = {(): mdict_full[None]}
                mdict[mtype.key] = tools1_11.convert_model(mtype.factors, m, lrtest_models)

        return cls(mlogits)

    @classmethod
    def load_saved_analysis(cls, fn):
        import joblib
        return cls(joblib.load(fn))

    @classmethod
    def rel_plot(cls, df, *, key, ax):
        import seaborn as sns
        rel_key = f'rel_{key}'
        df[rel_key] = df[key] - df[key].min()
        sns.barplot(x=rel_key, y="coef", data=df, ax=ax)
        ax.set(
            xlabel=f'$\Delta${key}',
            ylabel='',
        )

    def save(self, fn):
        import joblib
        joblib.dump(self.mlogits, fn)

    def plot(self, *, figure_fn=None):
        import matplotlib.pyplot as plt
        import pandas as pd
        def df_from_mlogits(models, modeltypes):
            return pd.DataFrame([
                dict(
                    coef=modeltype.name,
                    AIC=models[modeltype.key]['AIC'],
                    LL=models[modeltype.key]['logLik'],
                )
                for modeltype in modeltypes
            ])

        f, axes = plt.subplots(3, 1, figsize=(5, 2.25*3))
        for ptype, ax in zip(probes, axes):
            #f, ax = plt.subplots(figsize=(4, 2))
            ax.set_title(ptype.name)
            __class__.rel_plot(df_from_mlogits(self.mlogits[ptype.key], models), key='LL', ax=ax)
        plt.tight_layout()
        if figure_fn is not None:
            savefig(figure_fn)

    def table(self, models, *, print_table=True):
        import pandas as pd

        def cellify(cell):
            # Wraps cells using \n appropriately to properly have newlines
            if '\n' not in cell:
                return cell
            return '\makecell{' + (r' \\ '.join([
                el
                for el in cell.split('\n')
            ])) + '}'

        mcdf = []
        for mtype in models:
            row = {'Algorithm': cellify(mtype.name.replace(' (', ' \n('))}
            for ptype in probes:
                mm = self.mlogits[ptype.key][mtype.key]
                beta = mm['coef']['Estimate'][mtype.key]
                beta_se = mm['coef']['Std. Error'][mtype.key]
                lrt = mm['lrtests'][()]
                pv = lrt['Pr(>Chisq)'][1]
                chisq = lrt['Chisq'][1]
                df = lrt['Df'][1]
                elements = [
                    f'$\\beta={beta:.02f}$',
                    f'$SE={beta_se:.02f}$',
                    f'$\\chi^2({int(df)})={chisq:.1f}$',
                    f'${pvalue(pv)}$',
                ]
                # row[ptype.name] = '\makecell{' + (r' \\ '.join([ el for el in elements ])) + '}'
                row[ptype.name] = cellify('\n'.join(elements))

                #row[ptype.name] = f'$\\chi^2({int(df)})={chisq:.1f}$, $p \num{{{pres}}}$'
                #ss.append(f'{ptype.name}: $\\chi^2({int(df)})={chisq:.1f}$, $p {tools1_11.pval_to_pres(pv)}$')
            #print(', '.join(ss))
            #print()
            mcdf.append(row)
        mcdf = pd.DataFrame(mcdf).set_index('Algorithm')

        if print_table:
            with pd.option_context('display.max_colwidth', 999999):
                lines = mcdf.to_latex(escape=False, index_names=False).split('\n')
                print('\n'.join([
                    f'\\rule{{0pt}}{{3em}}{line}[2em]\n\\hline' if r'\\' in line else line
                    for line in lines
                ]))

        return mcdf
