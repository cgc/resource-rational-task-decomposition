from msdm.core.utils.funcutils import method_cache
import copy, collections, os, sys, json, contextlib, itertools
import pandas as pd
import numpy as np
currdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{currdir}/../../')
from experiment import generate
import rrtd, automated_design, run_models
import functools
import betweenness
import types
import pathlib
from dataclasses import dataclass

# HACK: This seems to be required in more recent versions for automatic conversion of numpy.
from rpy2.robjects import pandas2ri
pandas2ri.activate()

PROBE_TYPE_TO_LABEL = {
    'busStop': 'Bus Stop / Instant Teleportation',
    'subgoal': '"What location would you set as a subgoal?"',
    'solway2014': '"Choose a location you would visit along the way."',
}

def onehot(arr, limit):
    rv = np.zeros(limit)
    for el in arr:
        rv[el] = 1
    return rv

def makesavefig(version):
    dirname = f'figures-v{version}'

    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    def savefig(fn, *args, **kw):
        import matplotlib.pyplot as plt
        plt.savefig(f'{dirname}/{fn}', *args, bbox_inches='tight', dpi=300, **kw)
    return savefig

def config_for_condition(allconfig, condition):
    '''
    >>> c = dict(f=dict(g=[dict(h=[0, 1]), dict(h=[2, 3])]), conditionToFactors={'f.g': [None, 0, None, 1], 'f.g.h': [None, 1, None, 0]})
    >>> assert config_for_condition(c, 1) == dict(f=dict(g=dict(h=1)))
    >>> assert config_for_condition(c, 3) == dict(f=dict(g=dict(h=2)))
    >>> corig = dict(f=dict(g=[dict(h=[0, 1]), dict(h=[2, 3])]), conditionToFactors={'f.g': [None, 0, None, 1], 'f.g.h': [None, 1, None, 0]})
    >>> assert c == corig, 'making sure there is no mutation of config'
    '''
    allconfig = dict(allconfig)

    keyidx = []
    for factor, values in allconfig['conditionToFactors'].items():
        idx = values[condition]
        keyidx.append((factor.split('.'), idx))

    # We sort to ensure that shorter keys appear first, so that their
    # conditions are applied before any that are more nested.
    keyidx = sorted(keyidx, key=lambda pair: len(pair[0]))

    for keys, idx in keyidx:
        # We walk the configuration to reach the parent of the current key.
        c = allconfig
        for key in keys[:-1]:
          c[key] = dict(c[key])
          c = c[key]

        # We assign the appropriate value to replace the array of potential values.
        key = keys[-1]
        c[key] = c[key][idx]

    del allconfig['conditionToFactors']
    return allconfig


def load_config(config_path: pathlib.Path):
    return json.loads(config_path.read_text())


def compress_runs(seq, *, as_string=False):
    '''
    >>> assert compress_runs([0, 1, 1, 1, 3], as_string=True) == "['0', '1x3', '3']"
    >>> assert compress_runs([0, 0, 1, 1, 1, 3], as_string=True) == "['0x2', '1x3', '3']"
    '''
    prev = seq[0]
    res = [[prev, 1]]
    for x in seq[1:]:
        if x == prev:
            res[-1][-1] += 1
        else:
            res.append([x, 1])
        prev = x
    if as_string:
        s = []
        for x, ct in res:
            if ct == 1:
                s.append(f'{x}')
            else:
                s.append(f'{x}x{ct}')
        return str(s)
    return res

def get_completed_rows(rawdf, *, debug=False):
    # find the modal sequence. we assume this is the right one.
    ct = collections.defaultdict(list)
    for pid, rows in rawdf.groupby('pid'):
        trial_seq = tuple([row.data['trial_type'] for row in rows.itertuples()])
        ct[trial_seq].append((pid, rows))
    numfirst = [(len(c), v) for v, c in ct.items()]
    count, maxvalue = max(numfirst)
    print(f'Modal sequence length={len(maxvalue)} {compress_runs(maxvalue, as_string=True)}')
    missing = 0
    invalids = []
    dftype = collections.Counter()

    allrows = []
    for count, value in sorted(numfirst, reverse=True):
        pids = [pid for pid, rows in ct[value]]
        for name, pred, sl in [
            ('complete', value == maxvalue, slice(None)), # if they complete the task
            ('skip-survey', value == maxvalue[:-1], slice(None)), # if they skip the survey question (due to server error)
            ('instructions-twice', value[1:] == maxvalue, slice(1, None)), # if they just did the instructions twice.
            # if they just did a couple of things
            ('did-at-most-12-things-then-restart', value[-len(maxvalue):] == maxvalue and len(value) <= len(maxvalue) + 12, slice(-len(maxvalue), None)),
            # they may have done it multiple times, but at least completed it entirely their first time.
            ('retried-after-completion-butusingfirst', value[:len(maxvalue)] == maxvalue, slice(len(maxvalue))),
        ]:
            if pred:
                dftype[name] += len(pids)
                if name not in ('complete', 'skip-survey'):
                    invalids.append(dict(
                        seq=compress_runs(value, as_string=True),
                        category=name,
                        pids=pids,
                    ))
                    if debug:
                        print(f'Invalid sequence {value} has type {name} for pids {pids}')
                    continue
                for pid, rows in ct[value]:
                    rows = [row for row in rows.itertuples()][sl]
                    assert len(rows) in (len(maxvalue), len(maxvalue)-1)
                    if len(rows) == len(maxvalue)-1:
                        # For someone who skipped survey, we make a fake survey data for them.
                        last = rows[-1].copy()
                        last.trial = len(maxvalue)-1
                        last.data = {
                            'responses': '{"Q0":"MISSING_DATA","Q1":"MISSING_DATA","Q2":"MISSING_DATA","Q3":"MISSING_DATA"}',
                            'trial_type': 'survey-text',
                        }
                        rows.append(last)
                        missing += 1
                    allrows.extend(rows)
                break
        else:
            if len(value) < 0.9 * len(maxvalue):
                continue
            t = 'unusable'
            if value[-len(maxvalue):] == maxvalue:
                t = 'unusable: completed after restart'
            elif value == maxvalue[:len(value)]:
                t = 'unusable: incomplete'
            dftype[t] += len(pids)
            invalids.append(dict(
                seq=compress_runs(value, as_string=True),
                category='no-category',
                pids=pids,
            ))
            if debug:
                if len(value) > .95 * len(maxvalue):
                    print('no match, but maybe meaningful?', count, value)
                else:
                    print(f'invalid sequence with pids {pids}')
    if missing:
        print(f'needed to impute survey for {missing}')
    print(f'summary of participant data. total participants {sum(dftype.values())}')
    for k, ct in sorted(dftype.items()):
        print(f'\ttype {k} | # participants {ct}')
    return pd.DataFrame(allrows), pd.DataFrame(invalids)

def setup_data(version):
    experiment_dir = os.getenv('HOME') + '/pu/cocosci-optdisco/'

    datadir = pathlib.Path(experiment_dir) / f'data/human_raw/{version}'

    d = pathlib.Path(currdir) / '.data' / version
    if d.exists():
        return d

    # Copy participant data.
    d.mkdir(exist_ok=True, parents=True)
    for fn in [
        'trialdata.csv',
        'questiondata.csv',
    ]:
        (d / fn).write_bytes((datadir / fn).read_bytes())

    # Copy experiment configuration.
    import subprocess
    data = subprocess.check_output([
        'git',
        '--git-dir', experiment_dir+'/.git',
        'show',
        f'v{version}:static/optdisco/js/configuration/configuration.js',
    ])
    prefix, suffix = b'export default ', b';'
    assert data.startswith(prefix) and data.endswith(suffix)
    data = data[len(prefix):-len(suffix)]
    (d / 'configuration.json').write_bytes(data)

    return d

def load_data(csv_path, filt):
    cols = ['pid', 'trial', 'time_AMBIGUOUS', 'data']
    rawdf = pd.read_csv(csv_path, names=cols)
    rawdf.data = rawdf.data.apply(json.loads)
    df, invalids = get_completed_rows(rawdf[filt(rawdf.pid)])
    df.reset_index(inplace=True, drop=True)
    return df, invalids

def load_questiondata(csv_path, filt, df):
    cols = ['pid', 'key', 'value']
    q_df = pd.read_csv(csv_path, names=cols)
    q_df = q_df[filt(q_df.pid)]
    q_df.reset_index(inplace=True, drop=True)

    pids = set(df.pid.values)
    questiondata = {}
    for pid, rows in q_df.groupby('pid'):
        if pid not in pids:
            continue
        qd = questiondata[pid] = {}
        for row in rows.itertuples(index=False):
            qd[row.key] = row.value

    '''
        # NOTE: This now happens inside ParticipantExperimentData, so it can be overriden more easily
    # Fold survey questions into questiondata
    for pid, rows in grouped_filt_df(df, lambda d: d['trial_type']=='survey-multi-choice'):
        rows = list(rows.iterrows())
        assert len(rows) == 1
        row = rows[0][1]
        responses = json.loads(row.data['responses'])
        assert list(responses.keys()) == ['Q0', 'Q1', 'Q2', 'Q3']
        questiondata[pid]['handedness'] = responses['Q0']
        questiondata[pid]['bigpicture-detail'] = responses['Q1']
        questiondata[pid]['picture'] = responses['Q2']
        questiondata[pid]['draw'] = responses['Q3']
    '''
    return questiondata

def grouped_filt_df(df, pred):
    for pid, rows in df.groupby('pid'):
        filt = rows.data.apply(lambda value: bool(pred(value)))
        yield pid, rows[filt]

def v1_9_to_v1_11(trials):
    newconf = dict(
        graph=[],
        embedding=[],
    )
    for c in trials['conditions']:
        newconf['graph'].append(dict(
            adjacency=c['adjacency'],
            ordering=dict(
                probes_solway2014=c['probes'],
                probes_subgoal=c['probes'],
                navigation=c['navigation'],
            ),
        ))
        newconf['embedding'].append(dict(
            #order=..., not sure if this should be task embed?
            coordinates=c['map_embedding']['coordinates'],
        ))
    newconf['conditionToFactors'] = {
        'graph': [0, 0, 2, 2],
        'embedding': range(4),
    }
    return newconf

class ExperimentData(object):
    @classmethod
    def init(cls, version, *, extra_filt=None):
        filt = lambda pids: ~pids.str.startswith('debug')
        if extra_filt is not None:
            orig_filt = filt
            filt = lambda pids: orig_filt(pids) & extra_filt(pids)

        datadir = setup_data(version)

        trials = load_config(datadir / 'configuration.json')
        if version == '1.9':
            trials = v1_9_to_v1_11(trials)
        df, invalids = load_data(datadir / 'trialdata.csv', filt)
        questiondata = load_questiondata(datadir / 'questiondata.csv', filt, df)
        return cls(version, trials, df, questiondata, invalids=invalids)

    def __init__(
        self, version, trials, df, questiondata, *, invalids=None,
        ParticipantExperimentDataCls=None,
    ):
        if ParticipantExperimentDataCls is None:
            ParticipantExperimentDataCls=ParticipantExperimentData

        self.version = version
        self.trials = trials
        self.df = df
        self.base_questiondata = questiondata
        self.invalids = invalids
        self.participants = [
            ParticipantExperimentDataCls(self, pid, rows)
            for pid, rows in df.groupby('pid')
        ]

    @property
    @method_cache
    def questiondata(self):
        return {p.pid: p.questiondata for p in self.participants}

    def config_for_condition(self, condition):
        return config_for_condition(self.trials, condition)

    def config_idx_for_condition(self, condition):
        return {k: v[condition] for k, v in self.trials['conditionToFactors'].items()}

    def condition_for_pid(self, pid):
        return int(self.questiondata[pid]['condition'])

    def grouped_filt_df(self, pred):
        yield from grouped_filt_df(self.df, pred)

    def group_participants_by(self, projfn):
        grouped = {}
        for p in self.participants:
            grouped.setdefault(projfn(p), []).append(p)
        return grouped

    def with_filtered_participants(self, pid_set):
        return self.__class__.init(self.version, extra_filt=lambda pids: pids.isin(pid_set))

    def navigation_performance_filter(self, *, last_k_trials=15, cutoff=1.75, exclude_len1=True):
        pids = set()
        for p in self.participants:
            trials = list(p.navigation_trials())
            if exclude_len1:
                trials = [(t, r) for t, r in trials if t['optimal_cost'] != 1]
            perf = sum(
                (len(row.data['states'])-1)/task['optimal_cost']
                for task, row in trials[-last_k_trials:]
            ) / last_k_trials
            if perf < cutoff:
                pids.add(p.pid)
        return pids

    def nodraw_filter(self):
        return {
            p.pid
            for p in self.participants
            if p.questiondata['picture'] in ('Did not take picture', 'Rarely looked at picture')
            if p.questiondata['draw'] in ('Did not draw map', 'Rarely looked')
        }

    def mdps(self):
        return[rrtd.Graph(dict(g['adjacency'])) for g in self.trials['graph']]

    def optimal_path_choice_df(self):
        allpath_cache = {}
        for mdpi, mdp in enumerate(self.mdps()):
            # reusing distance matrix
            path_gen = betweenness.make_path_generator_distance(mdp, None)
            for s in mdp.state_list:
                for g in mdp.state_list:
                    allpath_cache[mdpi, s, g] = set(path_gen(dict(start=s, goal=g)))

        ct = collections.Counter()
        chid = 0
        df = []

        for p in self.participants:
            probes = {}
            for t, row in p.probe_trials():
                d = probes.setdefault(row.data['copy'], {})
                d[t['start'], t['goal']] = row.data['states'][0]

            for trial_idx, (t, row) in enumerate(p.navigation_trials()):
                trajectory = tuple(row.data['states'])
                ct['nav:total'] += 1
                # We skip nav trials that were not optimal
                if t['optimal_cost'] != len(trajectory)-1:
                    continue


                s, g = t['start'], t['goal']
                paths = allpath_cache[p.config_idx()['graph'], s, g]
                assert trajectory in paths
                if len(paths) == 1:
                    ct['nav:optimal:onlyOneOptimalPath'] += 1
                    continue
                ct['nav:optimal:multipleOptimalPaths'] += 1

                # Generate choice DF in long form
                for i, path in enumerate(paths):
                    d = dict(
                        mdp_idx=p.config_idx()['graph'],
                        id=p.pid,
                        chid=chid,
                        alt=i,
                        alt_name=','.join(map(str, path)),
                        alt_onehot=onehot(path, limit=len(mdp.state_list)),
                        choice=trajectory == path,
                        trial_idx=trial_idx,
                    )
                    for probe in ['solway2014', 'subgoal']:
                        has_choice = (s, g) in probes[probe]
                        d[f'has_probe_choice_{probe}'] = has_choice
                        d[f'probe_choice_match_{probe}'] = has_choice and probes[probe][s, g] in path
                    df.append(d)
                chid += 1

        return types.SimpleNamespace(
            df=pd.DataFrame(df),
            ct=ct,
        )


class ParticipantExperimentData(object):
    @classmethod
    def probe_tasks_for_condition(cls, c):
        return c['graph']['ordering']['probes_solway2014'] + c['graph']['ordering']['probes_subgoal'] + [dict(start=None, goal=None)]

    def __init__(self, experiment_data, pid, rows):
        self.exp = experiment_data
        self.pid = pid
        self.rows = rows
        self.row_tuples = list(self.rows.itertuples(index=False))
        self.validate()

    @property
    def surveydata(self):
        sd = {}
        rows = list(self.filtered_rows(lambda d: d['trial_type'] == 'survey-multi-choice'))
        assert len(rows) == 1
        row = rows[0]
        responses = json.loads(row.data['responses'])
        assert list(responses.keys()) == ['Q0', 'Q1', 'Q2', 'Q3']
        sd['handedness'] = responses['Q0']
        sd['bigpicture-detail'] = responses['Q1']
        sd['picture'] = responses['Q2']
        sd['draw'] = responses['Q3']
        return sd

    @property
    @method_cache
    def questiondata(self):
        base = self.exp.base_questiondata[self.pid]
        # Backwards compatability: Fold survey questions into questiondata
        return dict(base, **self.surveydata)

    def elapsed_minutes(self):
        elapsed_ms = self.rows.time_AMBIGUOUS.max() - self.rows.time_AMBIGUOUS.min()
        return elapsed_ms / 1000 / 60

    def filtered_rows(self, pred):
        '''
        This yields rows with if data attributes satisfy the supplied predicate.
        '''
        return [row for row in self.row_tuples if pred(row.data)]
        '''
        for row in self.row_tuples:
            if pred(row.data):
                yield row
        '''
        '''
        for row in self.rows[self.rows.data.apply(pred)].itertuples(index=False):
            yield row
        '''
        '''
        for _, row in self.rows[self.rows.data.apply(pred)].iterrows():
            yield row
        '''
        '''
        for _, row in self.rows.iterrows():
            if pred(row.data):
                yield row
        '''

    def validate(self):
        self.config()
        state_set = set(self.mdp().state_list)

        for task, row in self.navigation_trials():
            assert row.data['states'][0] == task['start']
            assert row.data['states'][-1] == task['goal']
            assert len(row.data['states']) >= task['optimal_cost']

        for task, row in self.probe_trials():
            d = row.data
            assert len(d['states']) == 1
            assert d['copy'] in ('solway2014', 'subgoal', 'busStop')
            s = d['states'][0]

            if d['copy'] == 'solway2014':
                invalid = {task['start'], task['goal']}
            elif d['copy'] == 'subgoal':
                invalid = {task['start']}
            else:
                invalid = set()
            assert s not in invalid

            assert set(task['response_options']) == state_set - invalid

    @property
    def condition(self):
        return int(self.questiondata['condition'])

    @method_cache
    def config(self):
        return self.exp.config_for_condition(self.condition)
    @method_cache
    def config_idx(self):
        return self.exp.config_idx_for_condition(self.condition)

    def _get_all_navigation_trials(self):
        '''
        Meant to be overridden by future experiments with filler trials.
        Returns a sequence of pairs of task config and data row.
        '''
        pred = lambda d: d['trial_type'] == 'CircleGraphNavigation' and not d.get('practice')
        rows = list(self.filtered_rows(pred))
        navigation_tasks = self.config()['graph']['ordering']['navigation']
        assert len(rows) == len(navigation_tasks)
        return zip(navigation_tasks, rows)

    def navigation_trials(self, *, return_obj=False):
        navigation_tasks_and_rows = list(self._get_all_navigation_trials())
        for trial_idx, (task, row) in enumerate(navigation_tasks_and_rows):
            task = dict(
                task,
                # HACK: do we want to keep this?????
                trial_idx_for_type=trial_idx,
                trial_idx_for_type_fraction=trial_idx/len(navigation_tasks_and_rows),
            )
            rv = task, row
            if return_obj:
                rv = NavigationTrial(*rv)
            yield rv

    def mdp(self):
        return rrtd.Graph(dict(self.config()['graph']['adjacency']))

    def navigation_state_frequencies(self, *, endpoints=True):
        state_counts = np.zeros(len(self.mdp().state_list))
        for task, row in self.navigation_trials():
            states = row.data['states']
            if not endpoints:
                states = states[1:-1]
            for s in states:
                state_counts[s] += 1
        return state_counts / state_counts.sum()

    def probe_trials(self, *, copy=None, return_obj=False):
        tasks = ParticipantExperimentData.probe_tasks_for_condition(self.config())
        pred = lambda d: d['trial_type'] == 'CirclePathIdentification' and not d.get('practice')
        rows = list(self.filtered_rows(pred))
        mdp = self.mdp()

        assert len(rows) == len(tasks)
        for task, row in zip(tasks, rows):
            if copy is not None and row.data['copy'] != copy:
                continue

            rowcopy = row.data['copy']
            ro = [
                state
                for state in mdp.state_list
                if not (rowcopy == 'solway2014' and state in (task['start'], task['goal']))
                if not (rowcopy == 'subgoal' and state in (task['start'],))
            ]
            rv = dict(task, response_options=ro), row
            if return_obj:
                rv = ProbeTrial(*rv)
            yield rv

    def probe_choice_counts(self, *, copy=None):
        cts = np.zeros(len(self.mdp().state_list))
        for task, row in self.probe_trials():
            if copy is None or row.data['copy'] == copy:
                assert len(row.data['states']) == 1
                cts[row.data['states'][0]] += 1
        return cts

    def subgoal_choice_count_on_explicit_trials(self):
        return sum([
            0 if t.selected == t.goal else 1
            for t in self.probe_trials(copy='subgoal', return_obj=True)
        ])

@dataclass(frozen=True)
class BaseTrial:
    task: ...
    row: ...

    @property
    def start(self): return self.task['start']
    @property
    def goal(self): return self.task['goal']

@dataclass(frozen=True)
class NavigationStateDwell:
    state: ...
    duration_sec: ...
    start_sec: ...

class NavigationTrial(BaseTrial):
    # These are just nice aliases
    @property
    def optimal_action_count(self): return self.task['optimal_cost']

    # Computing things here.
    @property
    def is_filler_trial(self):
        return self.task['optimal_cost'] == 1

    @property
    def action_count(self):
        # State list includes both start and goal state, so we subtract one.
        return len(self.row.data['states']) - 1

    @property
    def action_count_relative_to_optimal(self):
        return self.action_count / self.task['optimal_cost']

    @property
    def is_optimal(self):
        return self.action_count == self.task['optimal_cost']

    @property
    def is_early_trial(self):
        # We define early trials as happening in first half, and late trials as last half.
        return self.task['trial_idx_for_type_fraction'] < 0.5

    @property
    def has_repeated_state_visit(self):
        s = self.row.data['states']
        unique_s = set(s)
        return len(s) != len(unique_s)

    @property
    def initial_planning_time_sec(self):
        '''
        Time between when the navigation interface is shown and
        the first action is taken. See NOTE for trial_duration_sec below.
        '''
        return self.row.data['times'][0] / 1000

    @property
    def trial_duration_sec(self):
        '''
        The total trial duration in seconds.

        NOTE: Must use this measurement of time, since it excludes
        the period of time where we were showing the map. Since that
        map is only shown before we've revealed the current task there's
        no confounding issue here, we just have to make sure we exclude
        that time since we decided to piggy-back the map as a prelude
        within this jsPsych trial type.
        '''
        return self.row.data['times'][-1] / 1000

    def dwell_events(self):
        # Our data format leaves out the offset (of 0) for the initial state.
        start_sec = [0] + self.row.data['times']
        return [
            NavigationStateDwell(
                state=state,
                duration_sec=duration_ms / 1000,
                start_sec=start_ms / 1000,
            )
            for start_ms, duration_ms, state in zip(
                start_sec[:-1],
                np.diff(start_sec),
                self.row.data['states'][:-1],
            )
        ]

    @property
    def was_map_shown(self):
        md = self.row.data.get('mapData')
        # None check is here b/c some older versions failed to include a dictionary for maps when not shown.
        return md is not None and md['showMap']

    @property
    def map_duration_sec(self):
        if not self.was_map_shown:
            # When the map wasn't shown, we make sure to return NaN here so this trial is excluded.
            return np.nan
        return self.row.data['mapData']['rt'] / 1000
        # NOTE: why did I use .get() below? I think mostly to interop with empty object that was default
        # value for .get('mapData', {}). exclude for now?
        # map_rt_sec=md.get('rt', np.nan) / 1000,

    @property
    def map_state_count(self):
        if not self.was_map_shown:
            # When the map wasn't shown, we make sure to return NaN here so this trial is excluded.
            return np.nan
        return len(self.row.data['mapData']['states'])

    @property
    def map_state_hover_duration_sec(self):
        if not self.was_map_shown:
            # When the map wasn't shown, we make sure to return NaN here so this trial is excluded.
            return np.nan
        return sum(self.row.data['mapData']['durations']) / 1000

class ProbeTrial(BaseTrial):
    @property
    def selected(self):
        s = self.row.data['states']
        # In early pilots, participants could select multiple states during probes.
        # However, in this version of the experiment, participants only select one.
        assert len(s) == 1
        return s[0]


def apa_pval(x, *, lower_bound = 0.001):
    '''
    >>> assert apa_pval(0.5) == "= .5"
    >>> assert apa_pval(0.51) == "= .51"
    >>> assert apa_pval(0.49) == "= .49"
    >>> assert apa_pval(0.051) == "= .051"
    >>> assert apa_pval(0.049) == "= .049"
    >>> assert apa_pval(0.0051) == "= .005"
    >>> assert apa_pval(0.0049) == "= .005"
    >>> assert apa_pval(0.0011) == "= .001"
    >>> assert apa_pval(0.0010) == "= .001"
    >>> assert apa_pval(0.0009) == "< .001"
    >>> assert apa_pval(0.00051) == "< .001"
    >>> assert apa_pval(0.00049) == "< .001"
    '''
    assert 0 <= x <= 1
    if x < lower_bound:
        op = '<'
        num = lower_bound
    else:
        op = '='
        num = x
    num_str = f'{num:.3f}'.lstrip('0').rstrip('0')
    return f'{op} {num_str}'


def pval_to_pres(x, *, addstars=False, onlystars=False):
    '''
    >>> assert pval_to_pres(5.1e-1) == "< 1"
    >>> assert pval_to_pres(5.1e-2) == "< 0.1"
    >>> assert pval_to_pres(4.9e-2) == "< 0.05"
    >>> assert pval_to_pres(1e-2) == "< 0.05"
    >>> assert pval_to_pres(0.9e-2) == "< 0.01"
    >>> assert pval_to_pres(5.1e-3) == "< 0.01"
    >>> assert pval_to_pres(4.9e-3) == "< 0.005"
    >>> assert pval_to_pres(5.1e-7) == "< 1e-06"
    >>> assert pval_to_pres(4.9e-7) == "< 5e-07"
    >>> assert pval_to_pres(0) == "< 5e-16"
    >>> assert pval_to_pres(0.4, addstars=True) == "< 0.5"
    >>> assert pval_to_pres(0.09, addstars=True) == "< 0.1 ."
    >>> assert pval_to_pres(0.04, addstars=True) == "< 0.05 *"
    >>> assert pval_to_pres(0.009, addstars=True) == "< 0.01 **"
    >>> assert pval_to_pres(0.0009, addstars=True) == "< 0.001 ***"
    >>> assert pval_to_pres(0.4, onlystars=True) == ""
    >>> assert pval_to_pres(0.0009, onlystars=True) == "***"
    '''
    eps = np.finfo(float).eps
    if x == 0:
        x = eps
    power = np.floor(np.log10(x))
    for value in [
        10 ** (power + np.log10(5)),
        10 ** (power + 1),
    ]:
        if x < value:
            #if value < 1e-15:
            #    value = 1e-15 # AHCK....

            if value >= 1e-3:
                valuepower = int(abs(np.floor(np.log10(value))))
                fmt = '< {:.'+str(valuepower)+'f}'
            else:
                fmt = '< {:.0e}'
            if addstars or onlystars:
                # from R: Signif codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
                stars = (
                    "***" if x < 0.001 else
                    "**" if x < 0.01 else
                    "*" if x < 0.05 else
                    "." if x < 0.1 else
                    ''
                )
                if onlystars:
                    return stars
                fmt = fmt+(' ' if stars else '')+stars
            return fmt.format(value)

def convert_df_to_rdf(df):
    with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
        return ro.conversion.py2rpy(df)

def convert_rdf_to_df(df):
    with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
        return ro.conversion.rpy2py(df)

def nannorm(arr):
    return (arr-np.nanmean(arr))/np.nanstd(arr)

def normdf(df):
    dfnorm = df.copy()
    for c in df.columns.values:
        if df[c].dtype == np.float64:
            dfnorm[c] = nannorm(df[c])
    return dfnorm

@contextlib.contextmanager
def no_conversion():
    with ro.conversion.localconverter(ro.default_converter): # disable pandas conversion temporarily
        yield


def call_by_query(fn, qs):
    '''
    >>> def x(f: int, g: bool, h: float = 3.): return f * h if g else f * 2
    >>> assert call_by_query(x, 'f=3&g=False') == 6
    >>> assert call_by_query(x, 'f=3&g') == 9
    >>> assert call_by_query(x, 'f=3&g=True') == 9
    >>> assert call_by_query(x, 'f=3&g=True&h=-1') == -3
    >>> assert call_by_query({'x': x}, 'x?f=3&g=True&h=-1') == -3
    >>> assert call_by_query({'x': x}, 'x|f=3,g,h=-1') == -3
    '''
    import inspect
    import urllib.parse

    qs = qs.replace('|', '?').replace(',', '&')

    if '?' in qs:
        name, qs = qs.split('?')
        try:
            fn = fn[name]
        except TypeError:
            fn = getattr(fn, name)

    q = urllib.parse.parse_qs(qs, keep_blank_values=True)
    p = inspect.signature(fn).parameters

    def parse_param(p, value):
        assert len(value) == 1
        value = value[0]
        if p.annotation == p.empty:
            return value
        if p.annotation is bool:
            if value.lower() in ('', 'true', 't'):
                return True
            elif value.lower() in ('false', 'f'):
                return False
            else:
                raise ValueError(f'invalid value for bool: {value}')
        return p.annotation(value)

    return fn(**{
        k: parse_param(p[k], v)
        for k, v in q.items()})


def multinomial_data_for_r(exp, *, normalize=True, **kw):
    df = _multinomial_data_as_df(exp, **kw)
    res = {}
    for copy_, rows in df.groupby('copy'):
        if normalize:
            rows = normdf(rows)
        rdf = convert_df_to_rdf(rows)
        with no_conversion():
            res[copy_] = ro.r.convert_rdf_to_dfidx(rdf)
    return res

def _multinomial_data_as_df(
    exp, *,
    navigation_state_frequencies_kw=dict(),
    extra_graph_scores=None,
    modelkeys=None,
    tqdm=None,
):
    if tqdm is None:
        tqdm = lambda x: x
    else:
        try:
            from tqdm.auto import tqdm
        except:
            tqdm = lambda x: x
    mlogitdf = collections.defaultdict(list)

    modelkeys = modelkeys or [
        'iddfs', 'bfs', 'rw',
        'solway_entopt',
        'solwayr_partition_greedy_uniformexit',
        'qcut', 'log_degree_centrality', 'log_betweenness_centrality',
        'tomov',
    ]
    scores_by_graphidx = []
    for g in exp.trials['graph']:
        mdp = rrtd.Graph(dict(g['adjacency']))
        model_results = {}
        for mk in modelkeys:
            try:
                if isinstance(modelkeys, dict) and modelkeys.get(mk):
                    res = modelkeys[mk](mdp)
                elif '?' in mk:
                    name, qs = mk.split('?')
                    res = call_by_query(functools.partial(getattr(run_models, name), mdp), qs)
                else:
                    res = getattr(run_models, mk)(mdp)
            except Exception as e:
                print(f'Error when running model {mk} on graph {mdp}: {type(e)} {str(e)}')
                res = dict(error=str(e))
            model_results[mk] = res
        if extra_graph_scores is not None:
            model_results.update(extra_graph_scores(mdp))
        scores_by_graphidx.append(model_results)

    embedding_distance_corr = {}
    for gi, g in enumerate(exp.trials['graph']):
        mdp = rrtd.Graph(dict(g['adjacency']))
        d = rrtd.floyd_warshall(mdp)
        for ei, e in enumerate(exp.trials['embedding']):
            if 'order' not in e:
                continue
            embedding_distance_corr[gi, ei] = generate.correlate_distance_and_coordinates(
                d, circle_order=e['order'])

    chid = 0
    for p in tqdm(exp.participants):
        graphidx = p.config_idx()['graph']
        embeddingidx = p.config_idx()['embedding']
        logfreq = np.log(p.navigation_state_frequencies(**navigation_state_frequencies_kw))
        mdp = p.mdp()
        g6 = automated_design.dump_g6(mdp)
        for task, row in p.probe_trials():
            selected = row.data['states'][0]
            rowcopy = row.data['copy']
            for state in task['response_options']:
                if rowcopy == 'solway2014':
                    assert state not in (task['start'], task['goal'])
                elif rowcopy == 'subgoal':
                    assert state not in (task['start'],)
                mlogitdf['task'].append(str((task['start'], task['goal'])))
                mlogitdf['chid'].append(chid)
                mlogitdf['alt'].append(state)
                mlogitdf['id'].append(p.pid)
                mlogitdf['copy'].append(rowcopy)
                mlogitdf['choice'].append(state==selected)
                mlogitdf['log_stateocc'].append(logfreq[state])
                for k, v in scores_by_graphidx[graphidx].items():
                    if not ({'error', 'indexed_tdres', 'scores'} & v.keys()):
                        # We use (start, goal) as a key
                        v = v.get((task['start'], task['goal']), None)
                    if v is None or 'error' in v:
                        mlogitdf[k].append(np.nan)
                    elif 'indexed_tdres' in v:
                        use_no_sg_score = rowcopy == 'subgoal' and state == task['goal']
                        mlogitdf[k].append(v['indexed_tdres'][() if use_no_sg_score else (state,)])
                    else:
                        assert 'scores' in v
                        assert k not in ('iddfs', 'bfs'), 'Should have indexed_tdres for these'
                        mlogitdf[k].append(v['scores'][state])
                mlogitdf['graph'].append(g6)
                mlogitdf['graphidx'].append(graphidx)
                mlogitdf['embeddingidx'].append(embeddingidx)
                mlogitdf['embedding_distance_corr'].append(embedding_distance_corr.get((graphidx, embeddingidx), None))
                # HACK HACK HACK
                #mlogitdf['iddfs_invalidsg'].append(scores['iddfs'][state])
                #use_no_sg_score = row.copy_ == 'subgoal' and state == row.goal
                #mlogitdf['iddfs'].append(scores['iddfs'][None if use_no_sg_score else state])
                # HACK end
            '''
            if extra_graph_scores:
                for k, v in extra_graph_scores(mdp, task['start'], task['goal']):
                    for state in task['response_options']:
                        mlogitdf[k].append(v[state])
            '''
            assert len({len(v) for k, v in mlogitdf.items()}) == 1, 'Every column should have same number of entries'
            chid += 1
    return pd.DataFrame(mlogitdf)

def setup_r():
    ro.r('''
    library('mlogit')
    assert = function(pred, msg='') {
        if (!pred) {
            stop(msg);
        }
    }
    convert_rdf_to_dfidx = function(df) {
        dfidx(df, idx = list(c("chid", "id")), idnames = c("chid", "alt"))
    }
    convert_rdf_to_dfidx2 = function(df) {
        mlogit.data(df, shape='long', alt.var='alt', chid.var='chid', id.var='id')
    }
    make_rpar_n = function(factors) {
        setNames(rep('n', length(factors)), factors)
    }
    runmlogit = function(dff, factors, rpar=TRUE, show_summary=FALSE, R=100, part2=c('0'), part3=c('0')) {
        # For more info on what part2/3 mean, look at https://cran.r-project.org/web/packages/mlogit/vignettes/c2.formula.data.html
        # factors/fml/part1 corresponds to alt-/choice-situation specific variables with generic coefficients
        # e.g. transit_choice ~ cost
        # part2 corresponds to choice-situation specific variables with alternative-specific coefficients
        # e.g. transit_choice ~ urban + income
        # part3 corresponds to alt-/choice-situation specific variables with alternative-specific coefficients
        # e.g. transit_choice ~ in_vehicle_travel_time
        if (!any(class(dff)=='dfidx')) {
            dff = convert_rdf_to_dfidx(dff)
        }
        part2 = paste(part2, collapse='+');
        part3 = paste(part3, collapse='+');
        empty_model = length(factors) == 0 && part2 == '0' && part3 == '0';
        #if (!length(factors) && part2!='0' && part3) {
        if (empty_model) {
            m = nullmodel(dff)
        } else {
            fml = as.formula(sprintf('choice ~ %s-1|%s|%s', paste(factors, collapse='+'), part2, part3));
            scalarlogical = length(rpar)==1 && is.logical(rpar)
            if (scalarlogical && rpar==F) {
                rpar = c()
                panel = FALSE
            } else {
                rpar = if(scalarlogical) make_rpar_n(factors) else make_rpar_n(rpar)
                panel = TRUE
            }
            m = eval(bquote(mlogit(
                fml, dff,
                # Hyper-parameters for fitting
                halton=NA, R=R,
                rpar=.(rpar), panel=.(panel))))
        }
        if (show_summary) {
            show(summary(m))
        }
        m
    }
    nullmodel_same_number_options = function(mlogitdf) {
        # A lot of details here don't matter much; the offset is the only really critical piece.

        # Filtering down to only choice=1 rows, since this is in "long" mlogit format.
        choice1 = mlogitdf[mlogitdf$choice==1,]
        # We assume for now that each choice situation has the same number of choices. Should be easy
        # enough to write with tapply, but skipping for now since I don't need it yet.
        assert(nrow(mlogitdf) %% nrow(choice1) == 0, 'if not evenly divisible, then this has a mix of choice situations')
        chance = nrow(choice1)/nrow(mlogitdf)

        # Solved for offset in: chance = 1 / (1 + exp(-offset))
        # 1+exp(-offset) = 1/chance
        offset = -log(1/chance-1)
        glm(choice ~ -1, data=choice1, family=binomial(link='logit'), offset=rep(offset, nrow(choice1)))
    }
    nullmodel = function(mlogitdf) {
        # This model returns a null model corresponding to uniform choice among alternatives.
        # We assume input is in "long" mlogit format and appropriately determine what uniform choice
        # is based on the number of alternatives in every choice setting.
        # We do this by returning a GLM with a fixed offset per choice setting that corresponds
        # to uniform random choice -- this makes it possible to use R's function for lrtests and
        # other comparisons between models.

        # A lot of details here don't matter much; the offset is the only really critical piece.

        # Get # of options per choice situation
        alts_per_chid = tapply(rep(1, nrow(mlogitdf)), mlogitdf$chid, sum)
        chance = 1/alts_per_chid

        # Solved for offset in: chance = 1 / (1 + exp(-offset))
        # 1+exp(-offset) = 1/chance
        offset = -log(1/chance-1)

        # We have to supply some kind of df # of rows equal to # of choice situations, even though
        # we're not going to use the parameters. We do set the choice to true, though, since that's
        # what our offset is relative to.
        fake_df = data.frame(choice=rep(T, length(chance)))

        glm(choice ~ -1, data=fake_df, family=binomial(link='logit'), offset=offset)
    }
    ''')

def r_with_vars(rcode, *, convert_result=False, **kw):
    '''
    This takes some R code and variables used in the R code (as kwargs)
    and runs it by dynamically making an appropriate function. Assumes
    that the variables you are handling are R objects.
    '''
    if kw:
        keys, values = zip(*kw.items())
    else:
        keys, values = [], []
    rv = ro.r('function('+','.join(keys)+') {'+rcode+'}')(*values)
    if convert_result:
        return convert_rdf_to_df(rv)
    else:
        return rv

try:
    import rpy2.robjects as ro
    import rpy2.robjects.pandas2ri
    setup_r()
except Exception as err:
    print("WARNING: Couldn't load R")
    import traceback;traceback.print_exc()

def powerset(iterable, *, proper_subset=False):
    '''
    >>> assert list(powerset([0, 1])) == [(), (0,), (1,), (0, 1)]
    >>> assert list(powerset([0, 1], proper_subset=True)) == [(), (0,), (1,)]
    '''
    s = list(iterable)
    lim = len(s) if proper_subset else len(s)+1
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(lim))

def convert_model(factors, m, mods):
    with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
        return dict(
            coef=ro.r('function(m){as.data.frame(summary(m)$CoefTable)}')(m),
            AIC=ro.r.AIC(m)[0],
            logLik=ro.r.logLik(m)[0],
            random_effects=ro.r.fitted(m, type='parameters'),
            lrtests={
                k: (
                    dict(error=f'nested model not present: {k}')
                    if mods.get(k) is None
                    else dict(error=f'nested model {k} had error: {mods[k]["error"]}')
                    if isinstance(mods[k], dict) and 'error' in mods[k]
                    else ro.r.lrtest(mods[k], m))
                # every subset will be sorted since our list of factors is too
                for k in powerset(factors, proper_subset=True)
            }
        )
