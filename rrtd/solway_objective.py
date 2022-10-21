#!/usr/bin/env python
# coding: utf-8
import random
import rrtd
import numpy as np
from msdm.algorithms import VectorizedValueIteration
from msdm.core.utils.funcutils import method_cache


class SolwayGraph(rrtd.Graph):
    def __init__(self, *args, disable_noise=False, **kwargs):
#        rrtd.TabularMarkovDecisionProcess.__init__(self, memoize=False)
        super().__init__(*args, **kwargs)
        self._rewards = {}
        self.disable_noise = disable_noise

    def reward(self, s, a, ns):
        if self.disable_noise:
            return super().reward(s, a, ns)
        '''
        Page 6 of Solway et al. 2014
        > each task being specified in terms of a start vertex (state), a goal vertex,
        > and a negative reward associated with traversal of each edge
        > (randomly sampled between -1.01 and 0.99, and held constant across tasks).
        '''
        key = (s, ns)
        if key not in self._rewards:
            v = random.uniform(-1.01, -0.99)
            # Assign to both directions, since it applies to each edge.
            self._rewards[s, ns] = v
            self._rewards[ns, s] = v
        return self._rewards[key]

    @method_cache
    def vi_for_goal(self, g):
        assert not self.is_task_instance, 'Can only call vi on abstract MDP instance to ensure reuse occurs.'
        vi = VectorizedValueIteration()
        res = vi.plan_on(self.for_task(g, g))
        assert res.converged
        return res

class SolwayOptionLevelMDP(rrtd.TaskInstanceMixin, rrtd.DeterministicShortestPathProblem, rrtd.TabularMarkovDecisionProcess):
    def __init__(self, mdp, subgoals):
        self.mdp = mdp
        self.subgoals = list(subgoals)

    @property
    def state_list(self):
        return self.mdp.state_list

    def actions(self, s):
        return self._options(s) + list(self.mdp.actions(s))

    def _options(self, s):
        return [('option', s) for s in self.subgoals]

    def next_state(self, s, a):
        if isinstance(a, tuple) and a[0] == 'option':
            return a[1]
        return self.mdp.next_state(s, a)

    def reward(self, s, a, ns):
        # Since we define self-visits as automatically satisfied (hit time/reward/cost=0)
        # we need a way to disincentivize them. This helps avoid bugs like what affected
        # previous unpublished results with DFS.
        if isinstance(a, tuple) and a[0] == 'option':
            if s == ns:
                return float('-inf')
            return self.mdp.vi_for_goal(ns).V[s]
        return self.mdp.reward(s, a, ns)

def assign_subgoals(mdp, policy):
    '''
    Given an option-level MDP and policy, this function determines a compatible
    hierarchical execution trace following after Solway et al. 2014.
    So, in an MDP with a subgoal to state 2 and a policy leading to the
    state sequence 0, 1, 2, 3, 4, a compatible option-preferring decomposition
    would look like this: [[0, 1], 2, 3, 4].
    '''
    s = mdp.initial_state()
    viable = set(mdp._options(s))

    path = []
    path_since_last_option = [s]

    while not mdp.is_terminal(s):
        # take action & transition
        a = policy.action_dist(s).sample()
        s = mdp.next_state(s, a)
        #path_since_last_option.append(s)

        # If we reach a viable subgoal, then we store the path
        # thus far hierarchically and reset variables.
        if ('option', s) in viable:
            path.append(path_since_last_option)
            path_since_last_option = []
            viable = set(mdp._options(s))
        else:
            # remove options that are not viable at this state.
            # HACK, this requires that we only use options
            # where they can be initiated.
            # We don't perform this filtering at an option, since
            # 1) It shouldn't matter if an option is viable at that option and
            # 2) We change the value of viable once we're at that option anyway.
            viable &= set(mdp._options(s))

        path_since_last_option.append(s)

    # At the end, we just add everything on.
    path.extend(path_since_last_option)

    return path


def solway_phi(mdp, td, subgoals, *, SolwayOptionLevelMDP_class=SolwayOptionLevelMDP, only_count_entrance_options=False, compute_subgoal_rate=False):
    assert isinstance(mdp, SolwayGraph)
    assert sorted(mdp.state_list) == list(range(len(mdp.state_list))), 'need simple state list'

    ol_mdp_dist = SolwayOptionLevelMDP_class(mdp, subgoals)

    # Optimizing! We call these now to keep them cached & avoid computing them after we clone.
    for m in [mdp, ol_mdp_dist]:
        m.transition_matrix
        m.reward_matrix
        m.action_matrix

    option_to_used_map = {
        sg: np.zeros(len(mdp.state_list), dtype=bool)
        for sg in subgoals
    }

    loginvphi = 0
    if compute_subgoal_rate:
        subgoal_rate = np.zeros(len(mdp.state_list))

    for task in td.support:
        # Make task and get solution.
        task_mdp = mdp.for_task(task['start'], task['goal'])
        policy = mdp.vi_for_goal(task['goal']).policy

        # Infer the hierarchical policy
        ol_mdp = ol_mdp_dist.for_task(task['start'], task['goal'])
        try:
            hier = assign_subgoals(ol_mdp, policy)
        except:
            print(task, policy)
            raise

        # Determine phi based on this hierarchical policy.
        # We skip the last state since that's the terminal.
        assert hier[-1] == task['goal']
        # We track when there are no more subgoals
        all_subgoals_done = False
        if compute_subgoal_rate:
            numsgs = sum(1 for el in hier[:-1] if isinstance(el, list))
            if numsgs == 0:
                subgoal_rate[task['goal']] += 1/len(td.support)
        for idx, el in enumerate(hier[:-1]):
            if isinstance(el, list):
                assert not all_subgoals_done
                # Mark all usage of subgoal so we can calculate cost of per-option policy later.
                sg = hier[idx+1]
                sg = sg[0] if isinstance(sg, list) else sg
                if compute_subgoal_rate:
                    subgoal_rate[sg] += 1/(len(td.support)*numsgs)
                for s in el:
                    option_to_used_map[sg][s] = True

                # Sum the value for the hierarchical policy
                loginvphi += np.log(1/len(ol_mdp.actions(el[0])))
            else:
                last_entrance_state = False
                if not all_subgoals_done:
                    all_subgoals_done = True
                    # The first time we no longer have subgoals is also our last
                    # initiation state.
                    last_entrance_state = True

                if last_entrance_state or not only_count_entrance_options:
                    loginvphi += np.log(1/len(ol_mdp.actions(el)))
                else:
                    # For other last states not run within a subgoal, we
                    # merely count the actions in the ground MDP
                    loginvphi += np.log(1/len(mdp.actions(el)))

    # Factoring in per-option cost based on what states have an action set.
    for sg, used_map in option_to_used_map.items():
        for s in mdp.state_list:
            if used_map[s]:
                loginvphi += np.log(1/len(mdp.actions(s)))

    if compute_subgoal_rate:
        assert np.isclose(subgoal_rate.sum(), 1)
        return loginvphi, subgoal_rate

    return loginvphi

def task_decomposition(orig_mdp, td, *, tqdm=lambda x: x, samples=50, resfmt=False, SolwayOptionLevelMDP_class=SolwayOptionLevelMDP, only_count_entrance_options=False):
    adjacency = rrtd.adjacency_ssp(orig_mdp)
    total = np.zeros(len(orig_mdp.state_list))
    for _ in tqdm(range(samples)):
        # sample a new MDP (since weights have random noise for tie-breaking)
        mdp = SolwayGraph(adjacency)
        total += np.array([
            solway_phi(mdp, td, [sg], SolwayOptionLevelMDP_class=SolwayOptionLevelMDP_class, only_count_entrance_options=only_count_entrance_options)
            for sg in mdp.state_list
        ]) / samples
    if resfmt:
        return [dict(subgoals=[sg], value=total[i]) for i, sg in enumerate(mdp.state_list)]
    return total
