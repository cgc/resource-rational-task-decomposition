import rrtd, automated_design


def old_undirected_graph_to_new(graph):
    '''
    Converting from old format (pair of state, edges) to dictionary (from state to edges).
    Also making sure that we convert graphs to avoid directed/undirected issues.
    '''
    g = {
        s: list(nss)
        for s, nss in graph
    }
    for s, nss in graph:
        for ns in nss:
            g.setdefault(ns, [])
            if s not in g[ns]:
                g[ns].append(s)
    # Make sure we sort before returning to keep things consistent.
    return {s: sorted(nss) for s, nss in g.items()}


c0 = set(range(5))
c1 = set(range(5, 10))
c2 = set(range(10, 15))
graph = [
    (0, c0-{0}),
    (1, c0-{1}),
    (2, c0-{2}),
    (3, c0-{3, 4}),
    (4, c0-{3, 4}),

    (5, c1-{5}),
    (6, c1-{6}),
    (7, c1-{7}),
    (8, c1-{8, 9}),
    (9, c1-{8, 9}),

    (10, c2-{10}),
    (11, c2-{11}),
    (12, c2-{12}),
    (13, c2-{13, 14}),
    (14, c2-{13, 14}),

    # cross-community
    (3, [8]),
    (4, [13]),
    (9, [14]),
]
f2a = rrtd.Graph(old_undirected_graph_to_new(graph))

import math
penta_angle = 72 / 180 * math.pi
def penta_pos(angle_mult, base):
    return (
        base[0] + math.sin(angle_mult*penta_angle),
        base[1] + math.cos(angle_mult*penta_angle),
    )

upper = (0, 0)
left = (-5/3, -5/2)
right = (5/3, -5/2)
f2a.pos = [
    penta_pos(4, upper),
    penta_pos(0, upper),
    penta_pos(1, upper),
    penta_pos(3, upper),
    penta_pos(2, upper),

    penta_pos(2, left),
    penta_pos(3, left),
    penta_pos(4, left),
    penta_pos(0, left),
    penta_pos(1, left),

    penta_pos(3, right),
    penta_pos(2, right),
    penta_pos(1, right),
    penta_pos(0, right),
    penta_pos(4, right),
]

# fig. 2C

graph = [
    (0, [1, 2, 3]),
    (2, [1, 3]),
    (4, [1, 3, 5]),
    (5, [6, 8]),
    (7, [6, 8]),
    (9, [6, 7, 8]),
]

assert old_undirected_graph_to_new(graph) == {
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 3],
    3: [0, 2, 4],
    4: [1, 3, 5],
    5: [4, 6, 8],
    6: [5, 7, 9],
    7: [6, 8, 9],
    8: [5, 7, 9],
    9: [6, 7, 8],
}

f2c = rrtd.Graph(old_undirected_graph_to_new(graph))

# must be x (horiz), y (vert)
f2c.pos = [
    (0, 5),
    (2, 5),
    (1, 4),
    (0, 3),
    (2, 3),
    (3, 2),
    (5, 2),
    (4, 1),
    (3, 0),
    (5, 0),
]

# fig. 2D

def gen_xy(w, h):
    for y in range(h):
        for x in range(w):
            yield x, y

def grid(idx=0, w=3, h=3):
    xy_to_node = lambda x,y: idx + x+y*w
    nodes = []
    for x, y in gen_xy(w, h):
        neighbors = []
        if 0 <= x-1: neighbors.append(xy_to_node(x-1,y))
        if x+1 < w: neighbors.append(xy_to_node(x+1,y))
        if 0 <= y-1: neighbors.append(xy_to_node(x,y-1))
        if y+1 < h: neighbors.append(xy_to_node(x,y+1))
        nodes.append((xy_to_node(x,y), neighbors))
    return nodes

graph = grid(0) + grid(10) + [
    (9, [2, 8, 10, 16])
]
f2d = rrtd.Graph(old_undirected_graph_to_new(graph))
f2d.pos = list(gen_xy(3, 3)) + [
    (3, 1)
] + [
    (x+4, y)
    for x, y in gen_xy(3, 3)
]


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
