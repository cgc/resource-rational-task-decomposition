import copy, functools, os
from tkinter import SINGLE
import numpy as np
import diskcache
import networkx
import rrtd, bfs_runtime_exact, depth_limited_dfs_analytic, hitting_time, solway_objective, alternatives, automated_design, graph_partition
import solwayr, shannon, search_algorithms
import diskcachetools
import json
from collections import Counter
import prior_envs
import tools

currdir = os.path.dirname(os.path.abspath(__file__))

nx_to_rrtd = automated_design.nx_to_rrtd
rrtd_to_nx = automated_design.rrtd_to_nx
nx_res_to_np = automated_design.nx_res_to_np

def mark(**kwargs):
    def _make_getter(k):
        setattr(mark, k, lambda fn: getattr(fn, k, None))
    def deco(fn):
        for k, v in kwargs.items():
            setattr(fn, k, v)
            _make_getter(k)
        return fn
    return deco

class Models(object):
    def _td(self, mdp):
        return rrtd.all_pairs_shortest_path_distribution(mdp, remove_successor_tasks=True)

    # RRTD algos
    def _rrtd(self, mdp, algo, *, task_decomposition=rrtd.task_decomposition, **kw):
        out = {}
        td = self._td(mdp)
        res = task_decomposition(mdp, td, algo, **kw)
        out['indexed_tdres'] = {tuple(sorted(r['subgoals'])): r['value'] for r in res}
        out['scores'] = rrtd.res_to_arr(mdp, res)
        return out
    def iddfs_sampling(self, mdp):
        fn = functools.partial(rrtd.new_search_sampling_algorithm, value_fn=search_algorithms.iddfs, samples=200)
        return self._rrtd(mdp, fn)
    def iddfs(self, mdp):
        return self._rrtd(mdp, depth_limited_dfs_analytic.iddfs_for_rrtd)
    def bfs(self, mdp):
        return self._rrtd(mdp, bfs_runtime_exact.make_bfs_runtime_exact_algorithm)
    def rw(self, mdp):
        return self._rrtd(
            mdp, rrtd.new_random_walk_algorithm,
            OptionLevelMDP_kwargs=dict(force_subgoal_use=True))
    def dfs(self, mdp):
        algo = functools.partial(depth_limited_dfs_analytic.recursive_dfs_for_rrtd, add_goal_to_visited_count=False)
        return self._rrtd(mdp, algo)

    def _rw_spectral(self, mdp, *, limit=None):
        return self._rrtd(
            mdp, functools.partial(hitting_time.new_random_walk_spectral_algorithm, limit=limit),
            OptionLevelMDP_kwargs=dict(force_subgoal_use=True))
    def rw_spectral_rank1(self, mdp):
        rank = 1
        # Since we skip the first eigenvector, the limit should be be rank+1
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank2(self, mdp):
        rank = 2
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank3(self, mdp):
        rank = 3
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank4(self, mdp):
        rank = 4
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank5(self, mdp):
        rank = 5
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank6(self, mdp):
        rank = 6
        return self._rw_spectral(mdp, limit=rank+1)
    def rw_spectral_rank7(self, mdp):
        rank = 7
        return self._rw_spectral(mdp, limit=rank+1)

    #
    def qcut(self, mdp):
        return dict(scores=rrtd.res_to_arr(mdp, hitting_time.qcut_decomposition(mdp)))
    def log_betweenness_centrality(self, mdp):
        return dict(scores=np.log(nx_res_to_np(
            networkx.algorithms.betweenness_centrality(rrtd_to_nx(mdp), endpoints=True))))
    def log_degree_centrality(self, mdp):
        return dict(scores=np.log(nx_res_to_np(
            networkx.algorithms.degree_centrality(rrtd_to_nx(mdp)))))

    @mark(compressed_cache=True)
    def solwayr_allpartitions(self, mdp, nsamples=10):
        td = self._td(mdp)
        seed = np.random.randint(0, 2**30)
        with solwayr.r_seed_ctx(seed):
            m = solwayr.SolwayModel(mdp, nsamples=nsamples)
        partitions = [
            [p[s] for s in range(len(p))]
            for p in graph_partition.graphenum(mdp)
        ]
        values = [
            dict(partition=p, value=m.logevidence(td, p))
            for p in partitions
        ]
        return dict(
            seed=seed,
            m=m,
            #pypaths=[[o['pypath'] for o in sps] for sps in m.sps_samples],
            values=values,
        )

    @mark(disable_cache=True)
    def solwayr_partition(
        self, mdp, *,
        greedy:bool=False,
        exclude_trivial_partition:bool,
        binary_exit:bool=False,
        uniform_exit:bool=False,
        subgoal_use:bool=False,
        epsilon:float=0,
        exclude_size1_partitions:bool=False,
        require_all_partitions_have_non_exit:bool=False,
    ):
        result = self.solwayr_allpartitions(mdp)
        ds = result['values']

        # First we do any filtering of partitions
        # - could only include bipartitions here too
        if exclude_trivial_partition:
            ds = [d for d in ds if d['partition'] != [0]*len(mdp.state_list)]
        if exclude_size1_partitions:
            ds = [
                d for d in ds
                if min(Counter(d['partition']).values()) > 1
            ]
        if require_all_partitions_have_non_exit:
            new_ds = [
                d for d in ds
                if graph_partition.all_partitions_have_non_exit(mdp, dict(enumerate(d['partition'])))
            ]
            # HACK: We unfortunately can't "require" all partitions to have non-exits.
            # For many graphs (4154 of the 8-node graphs), there unfortunatley isn't a
            # way to guarantee this. However, for those we run experiments on this
            # isn't a concern.
            if new_ds:
                ds = new_ds

        # Assigning weights to partitions. When doing greedy assignment, we simply
        # filter out non-optimal partitions.
        if greedy:
            best = solwayr.maxes(ds, key=lambda d: d['value'])
            weight = np.ones(len(best)) / len(best) # uniform / average over best
        else:
            raise ValueError(f'Invalid setting greedy={greedy}')
        assert np.isclose(weight.sum(), 1), 'Making sure this is a distribution'

        # Now we define mapping of partitions to sg scores
        # some of these are distributions, which use some special code below.
        # the others are directly used as regressors in multinomial regression
        if binary_exit:
            is_dist = False
            def map_partition_to_sg_scores(p):
                return graph_partition.subgoal_choice_binary_exit(mdp, dict(enumerate(p)))
        elif uniform_exit:
            is_dist = True
            def map_partition_to_sg_scores(p):
                return graph_partition.subgoal_choice_uniform_exit(mdp, dict(enumerate(p)))
        elif subgoal_use:
            assert False
            # is_dist = True
            # td = self._td(mdp)
            # def map_partition_to_sg_scores(p):
            #     return m.subgoal_rate(td, p)
        else:
            raise ValueError(f'No mapping of partitions to sg specified.')

        sg_scores_by_partition = np.array([map_partition_to_sg_scores(d['partition']) for d in best])
        if is_dist:
            assert np.allclose(sg_scores_by_partition.sum(axis=1), np.ones(len(best))), 'Making sure these are distributions'
        # Take average of scores
        sg_scores = weight @ sg_scores_by_partition
        if is_dist:
            assert np.isclose(sg_scores.sum(), 1)
            uniform = np.ones(len(mdp.state_list)) / len(mdp.state_list)
            sg_scores = np.log((1 - epsilon) * sg_scores + epsilon * uniform)
        else:
            assert epsilon == 0

        return dict(
            seed=result['seed'],
            scores=sg_scores,
            best=best if len(best) != len(ds) else None, # Avoiding sending back all partitions
        )

    # HACK: this name is wrong! it should be binaryexit
    @mark(disable_cache=True)
    def solwayr_partition_greedy_uniformexit(self, mdp):
        return self.solwayr_partition(mdp, greedy=True, exclude_trivial_partition=True, binary_exit=True)

    @mark(disable_cache=True)
    def solwayr_partition_greedy_uniformexit_exclude_size1_partitions(self, mdp):
        return self.solwayr_partition(mdp, greedy=True, exclude_trivial_partition=True, binary_exit=True, exclude_size1_partitions=True)
    @mark(disable_cache=True)
    def solwayr_partition_greedy_uniformexit_require_all_partitions_have_non_exit(self, mdp):
        return self.solwayr_partition(mdp, greedy=True, exclude_trivial_partition=True, binary_exit=True, require_all_partitions_have_non_exit=True)

    @mark(require_explicit_compute=True, compressed_cache=True)
    def tomov(self, mdp):
        return alternatives.tomov_interop.chunking(mdp)

class ModelsWithCustomTaskDistribution(Models):
    def __init__(self, td):
        self.td = td
    def _td(self, mdp):
        return self.td

class ModelsWithCustomTaskDistributionFromKwargs(Models):
    def __init__(self, td_kwargs):
        self.td_kwargs = td_kwargs
    def _td(self, mdp):
        return rrtd.all_pairs_shortest_path_distribution(mdp, **self.td_kwargs)

def public_methods(inst):
    for propname in dir(inst):
        prop = getattr(inst, propname)
        if callable(prop) and not propname.startswith('_'):
            yield propname, prop

class DiskCacheMiss(Exception):
    pass

class Cache:
    @classmethod
    def cache_key(cls, method, mdp, args, kwargs):
        '''
        We've settled on this cache key for the following reasons:
            - 1) pickling an object isn't a great key b/c there is no canonical version of a
            pickled object -- this is even the case when the pickle version is held constant.
            I've run into a few issues with this across python version updates / platforms.
            - 2) pickling an MDP into a g6 string works for graphs, but the memory overhead
            isn't too much larger to serialize to JSON -- and it makes it possible to run
            these algorithms with other MDPs.
        '''
        assert isinstance(mdp, rrtd.TabularMarkovDecisionProcess)
        mdp_dict = {
            k: v
            for k, v in mdp.__dict__.items()
            # handles _cache_info_*, _cache_*, and _cached_*
            if not k.startswith('_cache')
        }
        return json.dumps(dict(
            method=method.__name__,
            mdp=mdp_dict,
            args=args,
            kwargs=kwargs
        ), sort_keys=True)

    @classmethod
    def mdp_memoize(cls, obj, propname, cache_tag):
        original_function = getattr(obj, propname)

        if mark.disable_cache(original_function):
            return original_function

        # HACK persisting a reference to the original unmemoized function on
        # the object seems to avoid issues with a weird recursion error during
        # multiprocessing via loky. It seems like, without this stored reference,
        # the memoized method replaces `original_function`.
        orig_key = '_nomemo_'+propname
        setattr(obj, orig_key, original_function)

        CACHE = None
        def init():
            nonlocal CACHE
            if CACHE is not None:
                return CACHE

            kw = {} # HACK
            if mark.compressed_cache(original_function):
                kw['disk'] = diskcachetools.CompressedDisk
            try:
                CACHE = diskcache.Cache(f'{currdir}/cache_run_models-v2{cache_tag}/{propname}', eviction_policy='none', **kw)
            except:
                print('Could not initialize cache for', propname, original_function)
                raise
            return CACHE

        def compute_cache_entry(mdp, *args, **kwargs):
            init()
            # HACK we make the key here so that once the cache serializes the arguments, it won't
            # hold onto cached properties (memoized methods) that are added when solving.
            key = cls.cache_key(original_function, mdp, args, kwargs)
            value = None
            if key not in CACHE:
                value = CACHE[key] = original_function(mdp, *args, **kwargs)
            return key, value

        @functools.wraps(original_function)
        def wrapped(mdp, *args, **kwargs):
            init()
            if mark.require_explicit_compute(original_function):
                key = cls.cache_key(original_function, mdp, args, kwargs)
                if key in CACHE:
                    return CACHE[key]
                else:
                    raise DiskCacheMiss()

            key, value = compute_cache_entry(mdp, *args, **kwargs)
            # HACK we avoid this load in compute_cache_entry to make things
            # a bit faster.
            if value is None:
                value = CACHE[key]
            return value
        wrapped.compute = compute_cache_entry
        wrapped._GET_CACHE = init
        wrapped.cache_key = lambda mdp, *args, **kwargs: cls.cache_key(original_function, mdp, args, kwargs)
        return wrapped

    @classmethod
    def memoize_models_instance(cls, obj, cache_tag):
        for propname, _ in list(public_methods(obj)): # copy to avoid any issues when mutating
            fn = Cache.mdp_memoize(obj, propname, cache_tag)
            # we mutate this object to make it easier for methods to reference cached results of other methods
            setattr(obj, propname, fn)


SINGLETON = Models()
Cache.memoize_models_instance(SINGLETON, '')

SINGLETON_WITH_SUCCESSOR_TASKS = ModelsWithCustomTaskDistributionFromKwargs(dict(remove_successor_tasks=False, remove_trivial_tasks=True))
Cache.memoize_models_instance(SINGLETON_WITH_SUCCESSOR_TASKS, '-withsucc')

# HACK: dynamically exporting to make things easier to run.
allmodels = []
allmodels_by_name = {}
for propname, fn in list(public_methods(SINGLETON_WITH_SUCCESSOR_TASKS)): # copy to avoid any issues when mutating
    allmodels.append(fn)
    allmodels_by_name[propname] = fn
    locals()[propname] = fn

def main(*,
    mdps=None, n_jobs=None, models_to_run=None, joblib_parallel_kwargs={},
    chunk_size=100,
    model_filter=None,
):
    import automated_design
    from tqdm.auto import tqdm
    if mdps is None:
        mdps = automated_design.download_bdm_graphs_8c()
    models_to_run = models_to_run or [
        model for model in allmodels
        if not mark.require_explicit_compute(model)
        if not mark.disable_cache(model)
    ]
    if model_filter is not None:
        pred = lambda s: model_filter in s
        if model_filter[0] == '~':
            model_filter = model_filter[1:]
            pred = lambda s: model_filter not in s
        models_to_run = [m for m in models_to_run if pred(m.__name__)]
    print('models_to_run', [m.__name__ for m in models_to_run])
    model_key_cache = {
        model: {hash(k) for k in model._GET_CACHE()} # only holding the hash to reduce memory requirements
        for model in models_to_run
    }
    tasks = list(tools.roundrobin(*[
        tools.chunks([
            (mdp, model) for mdp in mdps
            if hash(model.cache_key(mdp)) not in model_key_cache[model]
        ], chunk_size)
        for model in models_to_run
    ]))
    ct_remaining_tasks = sum(len(t) for t in tasks)
    ct_total_tasks = len(models_to_run) * len(mdps)
    print(f'Remaining tasks {ct_remaining_tasks} of {ct_total_tasks} ({100*ct_remaining_tasks / ct_total_tasks:.2f}%)')
    def dotask(tasks):
        for mdp, model in tasks:
            try:
                model.compute(mdp)
            except DiskCacheMiss:
                pass
            except Exception as e:
                import traceback
                print(f'error on {model} with mdp {automated_design.dump_g6(mdp)} {mdp}')
                traceback.print_exc()
    automated_design.parallel_compute(dotask, tasks, n_jobs=n_jobs, verbose=8, **joblib_parallel_kwargs)

# MDPs used by various experiments
experiment_mdps = [
    automated_design.parse_g6(g)
    for g in '''GCQuus
GQil^[
GCQRUw
GCdcuw
GCQuUs
GQhTVO
GCdbNG
G?B@`w
GCQRVO
GCQeTg
GCRVeg
G?qab[
GCp`e_
GCQTfo
G?`cmg
GCXnbW
G?B@e[
G?`eMs
GCrRUg
G?bBfc
G?ovE[
G?`bcw
GCQbTc
G?`rfG
G?qbfO
G?B@`W
G?bLbW
G?`fAw
G?`Db[
G?`DV_
GCQTnk
GCQrRW
GCQREo
G?ouUW
G?aJeW
GCQfES
GCdbNg
G?qa`[
G?`cvS
G?`fBk'''.split('\n')
]

testing_mdps = experiment_mdps[:4]

prior_envs_f2c = [prior_envs.f2c]

def explicit_compute(**kwargs):
    main(**kwargs, models_to_run=[
        model for model in allmodels
        if mark.require_explicit_compute(model)
    ], joblib_parallel_kwargs=dict(
        prefer='threads'
    ))

def main_all(**kwargs):
    def filtered(instance):
        return [
            model for mname, model in public_methods(instance)
            if not mark.require_explicit_compute(model)
            if not mark.disable_cache(model)
        ]
    main(**kwargs, models_to_run=filtered(SINGLETON) + filtered(SINGLETON_WITH_SUCCESSOR_TASKS))

def tomov_analysis(mdps=None, *, n_jobs, **kwargs):
    automated_design.parallel_compute(allmodels_by_name['tomov'].compute, mdps, n_jobs=n_jobs, verbose=8)

def main_experimentonly(**kwargs):
    main(mdps=experiment_mdps, **kwargs)

def small_test_run(**kwargs):
    main(mdps=experiment_mdps[:4], **kwargs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', nargs='?', default='main', help='name of task to perform: main, main_all, explicit_compute')
    parser.add_argument('--n_jobs', default=15, type=int, help='number of CPUs to use when running')
    parser.add_argument('--mdps', default=None, help='name of variable containing MDPs to use (valid: `experiment_mdps`, `testing_mdps`. defaults to all 8-node connected graphs)')
    parser.add_argument('--chunk_size', default=None, type=int, help='number of mdps to run in a single chunk')
    parser.add_argument('--model_filter', default=None, help='a substring to filter models by -- when the first character is ~, it filters to exclude')
    args = vars(parser.parse_args())
    if args['mdps']:
        args['mdps'] = locals()[args['mdps']]
    else:
        args['mdps'] = automated_design.download_bdm_graphs_8c()
    if args['chunk_size'] is None:
        del args['chunk_size']

    cmd = locals()[args.pop('task')]
    cmd(**args)
