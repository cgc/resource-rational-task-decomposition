import rrtd
import  numpy as np, networkx, requests, scipy.stats
import networkx.readwrite.graph6
import os, random

currdir = os.path.dirname(os.path.abspath(__file__))


from joblib import Parallel, delayed
def parallel_compute(fn, args, *, n_jobs=4, chunk_size=None, **kw):
    if chunk_size is not None:
        kw['batch_size'] = chunk_size
    print(f'Running task={fn.__name__} #tasks={len(args)} #workers={n_jobs}')
    return Parallel(n_jobs=n_jobs, **kw)(delayed(fn)(arg) for arg in args)

def nx_to_rrtd(g):
    states = list(g.nodes)
    assert sorted(states) == list(range(len(states)))
    return rrtd.Graph({
        node: list(g.adj[node].keys())
        for node in g.nodes
    })

def rrtd_to_nx(g):
    return networkx.Graph(g.adjacency)

def nx_res_to_np(res):
    '''
    Converts the result of a networkx algorithm like betweenness centrality (a dict from node to value)
    to a numpy array when the nodes are a continguous numbered sequence starting at 0.
    '''
    assert set(res.keys()) == set(range(len(res)))
    a = np.zeros(len(res))
    for k, v in res.items():
        a[k] = v
    return a

def download_bdm_graphs(fn, sample_lines=None):
    import networkx
    import requests
    import os

    try:
        os.makedirs(f'{currdir}/graph_cache')
    except:
        pass
    path = f'{currdir}/graph_cache/{fn}'
    if not os.path.exists(path):
        r = requests.get('http://users.cecs.anu.edu.au/~bdm/data/'+fn, stream=True)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in stream_unzipped(r):
                f.write(chunk)

    with open(path, 'rb') as f:
        byte_lines = list(f)
    if sample_lines is not None:
        byte_lines = random.sample(byte_lines, k=sample_lines)
    return [parse_g6(g.strip()) for g in byte_lines]

def stream_unzipped(r):
    # https://github.com/psf/requests/issues/2446#issuecomment-148306231
    import zlib

    def decompress_stream(stream):
        o = zlib.decompressobj(16 + zlib.MAX_WBITS)
        for chunk in stream:
            yield o.decompress(chunk)
        yield o.flush()

    if r.headers['content-type'] == 'application/x-gzip':
        return decompress_stream(r.iter_content(1024))
    else:
        return r.iter_content(1024)

def dump_g6(g):
    g = networkx.Graph(g.adjacency)
    v = networkx.readwrite.graph6.to_graph6_bytes(g).decode('ascii')
    header = '>>graph6<<'
    assert v.startswith(header)
    assert v.endswith('\n')
    return v[len(header):-1]

def parse_g6(g6):
    if isinstance(g6, str):
        g6 = g6.encode('ascii') # ensure we are bytes
    return nx_to_rrtd(networkx.readwrite.graph6.from_graph6_bytes(g6))

def is_complete(mdp):
    for s in mdp.state_list:
        if len(mdp.actions(s)) != len(mdp.state_list)-1:
            return False
    return True

def download_bdm_graphs_8c(*, exclude_complete=False):
    mdps = download_bdm_graphs('graph8c.g6')
    if exclude_complete:
        complete = mdps[-1] # HACK... that's the order it's in!
        # making sure it's the complete graph
        assert is_complete(complete)

        mdps = mdps[:-1]
    return mdps
