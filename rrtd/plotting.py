import warnings
import numpy as np

def _state_arg_to_callable(arg, default):
    '''
    Argument processing function to permit the use of callables,
    lists/tuples/dicts/numpy arrays, or constants when specifying
    parameters like state-specific properties like color/opacity/label/etc.
    '''
    if arg is None:
        return default
    if callable(arg):
        return arg
    if hasattr(arg, '__getitem__'):
        return lambda s: arg[s]
    else:
        # it's a constant.
        return lambda s: arg

def edge_penwidth_map(mdp, outrange=(0.2, 1.8)):
    '''
    This complicated function renders costs of edges through varying edge widths. Mainly used
    when visualizing graphs with noise on edges (as in Solway, noise used for tie-breaking)
    '''
    next_states = {
        s: {mdp.next_state(s, a) for a in mdp.actions(s)}
        for s in mdp.state_list}

    rewards = {
        (s, ns): {mdp.reward(s, a, ns) for a in mdp.actions(s) if mdp.next_state(s, a)==ns}
        for s in mdp.state_list
        for ns in next_states[s]
    }

    valid = (
        all(len(rew) == 1 for rew in rewards.values()) and
        all(rewards[s, ns] == rewards[ns, s] for s, ns in rewards.keys() if (ns, s) in rewards)
    )
    NO_PENWIDTH = {pair: '' for pair in rewards.keys()}
    if not valid:
        return NO_PENWIDTH

    reward_values = [list(rewardset)[0] for rewardset in rewards.values()]
    rmin = min(reward_values)
    rmax = max(reward_values)

    if rmin == rmax:
        return NO_PENWIDTH

    return {
        pair: str((list(rewardset)[0] - rmin) / (rmax - rmin) * (outrange[1]-outrange[0]) + outrange[0])
        for pair, rewardset in rewards.items()
    }

# Got these colors from plt.get_cmap('Set1').colors
# Multiplying by 10 is a bit of a hack; our goal here is to make sure we never run out of colors
partition_colors = [c+(1.0,) for c in [
    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    (1.0, 0.4980392156862745, 0.0),
    (1.0, 1.0, 0.2),
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
    (0.6, 0.6, 0.6)]] * 10

def plot_graph(
    mdp, *,
    # to color nodes
    z=None, vmin=None, vmax=None,
    labels=None,
    rgb=None,

    layout='neato',
    title='',
    size=None,
    constant_node_size=False,
    node_scaling=(0.2, 0.3),
    eps=1e-5,
    node_arg={},
    edge_arg={},
    graph_arg={},
    display=False,
    fixed_pos=True,
    pos=None,

    edge_penwidth_map=edge_penwidth_map,

    partition=None, # is this feature creep?
    state_list=None,
):
    import graphviz
    def alpha_to_hex(alpha):
        return '%02x' % (int(alpha*255))

    try:
        initial_state = mdp.initial_state()
    except ValueError:
        initial_state = None

    if partition is None:
        rgbdefval = (
            lambda s:
            (1, 0.5, 0.5) if mdp.is_terminal(s) else
            (0.5, 1, 0.5) if initial_state == s else
            (0.5, 0.5, 1))
    else:
        compressed = []
        g_to_i = {}
        for el, group in (sorted(partition.items()) if isinstance(partition, dict) else enumerate(partition)):
                if group not in g_to_i:
                    g_to_i[group] = len(g_to_i)
                compressed.append(g_to_i[group])
        rgbdefval = lambda s: partition_colors[compressed[s]]
    rgb = _state_arg_to_callable(rgb, rgbdefval)
    labels = _state_arg_to_callable(labels, lambda s: '')

    alphas = None
    if z is not None:
        z = np.array(z)
        vmin = vmin or z.min()
        vmax = vmax or z.max()
        z = np.clip(z, vmin, vmax)
        diff = (vmax - vmin)
        if diff > eps:
            alphas = (z - vmin) / diff
        else:
            warnings.warn(f'Ignoring data because data range {diff} is smaller than eps={eps}')
            alphas = np.zeros(z.shape)
    alphas = _state_arg_to_callable(alphas, lambda s: 0.0)
    tooltips = (lambda s: f'{s}') if z is None else (lambda s: f'{s} - {z[s]}')

    g = graphviz.Digraph()
    g.attr('graph', layout=layout, size=size and str(size), label=title, labelloc='t', **graph_arg)
    next_states = {
        s: {mdp.next_state(s, a) for a in mdp.actions(s)}
        for s in mdp.state_list}
    edge_penwidth_map_ = edge_penwidth_map(mdp)
    for s in state_list or mdp.state_list:
        label = labels(s)
        alpha = 1 if initial_state == s or mdp.is_terminal(s) else alphas(s)
        color = '#'+''.join(alpha_to_hex(c) for c in tuple(rgb(s)) + (alpha,))

        default_args = dict(
            color='black', fillcolor=color, style='filled',
            fontsize=str(8), width=str(0.25), height=str(0.1),
            shape='rect',
            #penwidth=str(7),
        )
        if not label:
            default_args['shape'] = 'circle'

        if constant_node_size is True:
            size = 0.5
        elif constant_node_size is False:
            intercept, slope = node_scaling
            size = intercept + alpha * slope
        else:
            size = constant_node_size
        default_args['width'] = default_args['height'] = str(size)

        if pos is None and hasattr(mdp, 'pos'):
            pos = mdp.pos
        if pos is not None:
            x, y = pos[s]
            default_args['pos'] = f'{x},{y}{"!" if fixed_pos else ""}'

        g.node(str(s), label=label, tooltip=tooltips(s), **dict(
            default_args,
            **node_arg(s) if callable(node_arg) else node_arg))

        for ns in next_states[s]:
            kw = dict(
                dict(penwidth=edge_penwidth_map_[s, ns]),
                **edge_arg(s, ns) if callable(edge_arg) else edge_arg)
            if s in next_states[ns]:
                if s >= ns:
                    continue
                g.edge(str(s), str(ns), dir='none', **kw)
            else:
                g.edge(str(s), str(ns), **kw)

    if display:
        from IPython.display import display
        display(g)
        return None
    return g


def display_graphs(*graphs, columns=4, html=True, cellwidth=90, cellheight=90, cellpadding=15, should_display=True):
    '''
    Future feature:

    If we want to add text annotations per graph, we could insert a <text> node at an appropriate x/y, like
    <text x="5pt" y="85pt" dominant-baseline="text-before-edge">...</text>
    dominant-baseline is important for changing where text sits https://vanseodesign.com/web-design/svg-text-baseline-alignment/
    Adding text annotations this way is better than using graphviz b/c by default graphviz might resize text sort of arbitrarily.

    '''

    import math
    from IPython.display import display_html, display, SVG

    if len(graphs) == 1 and not hasattr(graphs[0], '_repr_image_svg_xml'):
        graphs = graphs[0]

    frac = math.floor(100*1/columns)-1 # HACK
    if html:
        display_html(''.join(
            g._repr_image_svg_xml().replace('<svg', f'<svg style="width: {frac}%; max-height: {frac}vw;"')
            for g in graphs
        ), raw=True)
        return

    cellwidthpad = cellwidth + cellpadding
    cellheightpad = cellheight + cellpadding
    rows = len(graphs) // columns + (0 if len(graphs) % columns == 0 else 1)
    from bs4 import BeautifulSoup
    fmt = f'''
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{cellwidthpad*columns}pt" height="{cellheightpad*rows}pt">
    {"{}"}
    </svg>
    '''
    # # Adding a width/height here causes weird results?
    # fmt = f'''
    # <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
    #     viewBox="0 0 {cellwidthpad*columns} {cellheightpad*rows}">
    # {"{}"}
    # </svg>
    # '''

    # doubly hack... we don't support wrapping.
    #frac = math.floor(100*1/columns)-1 # HACK, to ensure there's no margin issue?

    def to_svg(graph, index):
        parsed = BeautifulSoup(graph._repr_image_svg_xml(), 'xml')
        svg = parsed.find('svg')
        svg.findChild(attrs=dict(fill='white')).replace_with() # Removing white background
        svg['width'] = f'{cellwidth}pt'
        svg['height'] = f'{cellheight}pt'
        svg['x'] = f'{cellpadding/2 + cellwidthpad*(index%columns):.02f}pt'
        svg['y'] = f'{cellpadding/2 + cellheightpad*(index//columns):.02f}pt'
        del svg['xmlns']
        del svg['xmlns:xlink']
        # HACK: can this be done with g tags by parsing internal size out of viewbox
        # and making the appropriate translate / transform?
        return str(svg)

    def _to_svg(graph, index):
        '''
        This is an alternative that transforms the graphs to scale/position them.

        It was initially written assuming that conversion to PDF had artifacts b/c
        we had nested SVG nodes. However, the real issue is the white background that
        graphviz adds.
        '''
        parsed = BeautifulSoup(graph._repr_image_svg_xml(), 'xml')
        svg = parsed.find('svg')
        children = [c for c in svg.children if str(c).strip()]
        assert len(children) == 1
        # NOTE: it's important to pull the child <g> out, instead of wrapping it in another <g>
        g = children[0]
        # Removing white background
        g.findChild(attrs=dict(fill='white')).replace_with()
        x, y, w, h = [float(v) for v in svg['viewBox'].split()]
        assert x == 0 and y == 0

        # rescale appropriately
        frame_aspect = cellwidth / cellheight
        g_aspect = w / h
        ratio = cellwidth / w if g_aspect > frame_aspect else cellheight / h

        # locate appropriately
        x = f'{cellpadding/2 + cellwidthpad*(index%columns):.02f}'
        y = f'{cellpadding/2 + cellheightpad*(index//columns):.02f}'

        g['transform'] += f' scale({ratio}) translate({x} {y})'

        return str(g)

    svg = SVG(fmt.format('\n'.join(to_svg(g, index) for index, g in enumerate(graphs))))
    if should_display:
        display(svg)
    else:
        return svg
