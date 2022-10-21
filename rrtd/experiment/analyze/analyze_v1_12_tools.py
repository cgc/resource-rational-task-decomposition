import numpy as np
from analyze_v1_11_tools import ExperimentData, ParticipantExperimentData
import rrtd

class ParticipantExperimentData(ParticipantExperimentData):
    def validate(self):
        super().validate()

        self.all_connected_to_modal_info = []

        mdp = self.mdp()
        edges = [(s, mdp.next_state(s, a)) for s in mdp.state_list for a in mdp.actions(s)]

        # Making sure front-end logic is working!
        ct = [0] * len(mdp.state_list)
        numchecks = 0
        for ti, (t, row) in enumerate(self.navigation_trials()):
            dyn = row.data.get('dynamicProperties', None)
            if dyn is not None:
                params = dyn['sampleLowOccTrialParams']
                if params['allConnectedToModal']:
                    self.all_connected_to_modal_info.append(dict(trial=ti, task=t, counter=ct))
                assert params['counter'] == ct

                if params['allConnectedToModal']:
                    # actually this is sort of a bummer?
                    maxsum = max(ct[s]+ct[ns] for s, ns in edges)
                    assert ct[dyn['start']]+ct[dyn['goal']] == maxsum
                else:
                    maxval = max(ct)
                    for key in ['start', 'goal']:
                        assert ct[dyn[key]] != maxval
                    maxsum_nonmodal = max(ct[s]+ct[ns] for s, ns in edges if ct[s] != maxval and ct[ns] != maxval)
                    assert ct[dyn['start']]+ct[dyn['goal']] == maxsum_nonmodal
                numchecks += 1

            # increment visit counter
            for s in row.data['states']:
                ct[s] += 1
        assert numchecks == 30, numchecks
        ct = np.array(ct)
        assert np.allclose(ct/ct.sum(), self.navigation_state_frequencies())

    def _get_all_navigation_trials(self):
        D = rrtd.floyd_warshall(self.mdp())

        pred = lambda d: d['trial_type'] == 'CircleGraphNavigation' and not d.get('practice')
        rows = list(self.filtered_rows(pred))
        navigation_tasks = self.config()['graph']['ordering']['navigation']
        assert len(rows) == len(navigation_tasks) * 2
        for t, task in enumerate(navigation_tasks):
            yield task, rows[2*t]
            dyn = rows[2*t+1]
            dyntask = dict(
                start=dyn.data['dynamicProperties']['start'],
                goal=dyn.data['dynamicProperties']['goal'],
            )
            dyntask['optimal_cost'] = D[dyntask['start'], dyntask['goal']]
            # getting distance mat isn't really needed
            assert dyntask['optimal_cost'] == 1
            yield dyntask, dyn

class ExperimentData(ExperimentData):
    def __init__(self, *args, **kwargs):
        p_cls = kwargs.pop('ParticipantExperimentDataCls', ParticipantExperimentData)
        super().__init__(*args, ParticipantExperimentDataCls=p_cls, **kwargs)

    def summarize_adaptive_all_connected(self, *, plt=None):
        nums = [len(p.all_connected_to_modal_info) for p in self.participants]
        tot = sum(nums)
        ns = sum(1 for n in nums if n)
        print(f'All connected report')
        print(f'\t{ns}/{len(nums)} ({100*ns/len(nums):.02f}%) participants affected.')
        print(f'\tOf affected, average # of affected trials {tot/ns:.2f}.')
        print(f'\tOf all participants, average # of affected trials {tot/len(nums):.2f}.')
        if plt is not None:
            plt.hist(nums)
