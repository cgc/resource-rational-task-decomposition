import sys, os
import rrtd, automated_design
import numpy as np

try:
    import matlab
    import matlab.engine
except ImportError:
    print('tomov_interop could not load matlab')

codedir = os.path.dirname(os.path.abspath(__file__))

eng = None
def init():
    global eng
    if eng is None:
        eng = start_matlab()
    return eng

def start_matlab():
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(codedir, 'chunking'))
    eng.addpath(codedir)
    return eng

def wrapped_matlab(fn, *, show_output=False, silent=False):
    import io
    out = io.StringIO()
    err = io.StringIO()

    def wrapped(*args, **kw):
        if show_output or silent:
            import io
            kw = dict(kw)
            kw['stdout'] = io.StringIO()
            kw['stderr'] = io.StringIO()
        rv = fn(*convert_to_matlab(args), **kw)
        if show_output:
            print(kw['stdout'].getvalue())
            print(kw['stderr'].getvalue())
        return rv
    return wrapped

def convert_to_matlab(arg):
    if isinstance(arg, np.ndarray):
        return matlab.double(arg.tolist())
    elif isinstance(arg, (list, tuple)):
        return [convert_to_matlab(a) for a in arg]
    elif isinstance(arg, (dict,)):
        return {k: convert_to_matlab(v) for k, v in arg.items()}
    else:
        return arg

def convert_from_matlab(v):
  if isinstance(v, dict):
    return {k: convert_from_matlab(el) for k, el in v.items()}
  elif isinstance(v, (list, tuple)):
    return [convert_from_matlab(el) for el in v]
  elif isinstance(v, matlab.double):
    return np.array(v)
  else:
    return v

def mdp_to_D(mdp):
    '''
    Convert one of our MDP's to the structure used in the repo.
    '''
    npedges = []
    for s in mdp.state_list:
        for t in mdp.state_list:
            if s >= t:
                continue
            if any(mdp.next_state(s, a) == t for a in mdp.actions(s)):
                npedges.append([s+1, t+1])

    N = len(mdp.state_list)
    r = init().cell(N, 1)
    return dict(
        name=automated_design.dump_g6(mdp),
        G=dict(
            N=float(N),
            E=rrtd.binary_adjacency_ssp(mdp).astype(np.float64),
            edges=np.array(npedges, dtype=np.float64),
            hidden_E = np.zeros((N, N)),
            hidden_edges=np.array([]),
        ),
        tasks=dict(
            s=np.array([]),
            g=np.array([]),
        ),
        r=r,
    )

def format_result(res):
    nstates = int(res['D']['G']['N'])
    rawcounts = np.bincount(np.array(res['loc'], dtype=np.int).flatten()-1, minlength=nstates)
    counts = rawcounts + 1 # Making sure none are zero
    ps = counts/counts.sum()
    return convert_from_matlab(dict(scores=np.log(ps), ps=ps, counts=rawcounts, detailed=res))

def chunking(mdp):
    eng = init()
    res = wrapped_matlab(eng.rungraph)(mdp_to_D(mdp))
    return format_result(res)
