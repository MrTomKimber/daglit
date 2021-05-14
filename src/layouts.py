import digraph

def rescale_positions(positions_d, xy_bounds):
    minx, maxx, miny, maxy = xy_bounds
    xvals, yvals = zip(*[v for v in positions_d.values()])
    minxvals, maxxvals, minyvals, maxyvals = min(xvals), max(xvals), min(yvals), max(yvals)
    xrangvals, yrangvals = maxxvals-minxvals, maxyvals-minyvals
    xrang, yrang = maxx-minx, maxy-miny

    return { k : ((((v[0]-minxvals)/xrangvals)*xrang)+minx, (((v[1]-minyvals)/yrangvals)*yrang)+miny) for k,v in positions_d.items() }

def pad_short_leaf_nodes(dag):
    d = dag.copy()
    v=0

    # Find any nodes that consolidate, rather than branch, and add new virtual leaf nodes to consolidate.
    original_nodes = set(d.nodes)
    for n in original_nodes:
        cps = d.coparents(n)
        if len(cps)>len(d.nodes[n].successors):
            for cp in cps-set(n):
                v=v+1
                d.add_edge((n,"__virtual_node_{v}".format(v=v)))

    t_sort = list(d.topological_sort(True))
    d_map = d.node_depth_map()
    max_depth = max((v for v in d_map.values()))
    print (d_map)

    # Extend early-closing leaf nodes to max-depth
    for t in d.leaf_nodes():
        n=t
        for l in range(d_map[t], max_depth):
            v=v+1
            d.add_edge((n, "__virtual_node_{v}".format(v=v)))
            n="__virtual_node_{v}".format(v=v)
    return d


def tree_layout(dag, hide_virtual=True,reverse=False):

    d = pad_short_leaf_nodes(dag)
    t_sort = list(d.depth_first_sort())[::-1]

    leaves = d.leaf_nodes()
    parents=set()
    finished = False
    posns={}
    ordering={}
    c=0
    leaf_count = len(leaves)
    depths = d.node_depth_map()
    print(depths)
    full_depth = max(depths.values())

    while len(leaves)>0:

        node = [l for l in t_sort if l in leaves].pop()
        parents=parents.union(set(d.nodes[node].predecessors.keys()))
        sibs = d.siblings(node)
        if len(sibs)==0:
            sibs = set([node])
        sibs = sibs - set(posns.keys())
        leaves=leaves-sibs
        print (node, leaves, sibs)
        for i in sorted(sibs):
            c=c+1
            ordering[i]=c
            if len(d.nodes[i].successors)==0:
                if not reverse:
                    posns[i]= ((c/leaf_count), (depths[i]/full_depth))
                else:
                    posns[i]= ((c/leaf_count), 1-(depths[i]/full_depth))
            else:
                children = d.nodes[i].successors
                xpos = sum([posns.get(l,[0,0])[0] for l in children])/len(children)
                if not reverse:
                    ypos = depths[i]/full_depth
                else:
                    ypos = 1- depths[i]/full_depth
                posns[i] = (xpos,ypos)

        if len(leaves)==0:
            leaves = parents-set(posns.keys())
            parents = set()
    if hide_virtual:
        return {k:v for k,v in posns.items() if k in dag.nodes}
    else:
        return posns


def svg_base(viewbox=(0,0,10,10), contents=None):
    viewbox = ", ".join([str(v) for v in viewbox])
    svg_chunk = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink" x="0" y="0"
    width="900" height="900" viewBox="{viewbox}">
      <g id="Layer_1">
        <g>{contents}
        </g>
      </g>
      </svg >
        """.format(viewbox=viewbox, contents=contents)
    return svg_chunk

def css_style():
    css_style = """
        <style>
            text {font-size:0.04em;}
            .small_text {font-size:0.01em;}
            .fill_white { fill: white; }
            .fill_black { fill: black; }
            .fill_red { fill: #FF3333; }
            .fill_orange { fill: #FFAA00; }
            .fill_yellow { fill: #FFFF00; }
            .fill_green { fill: #00FF00; }
            .fill_cyan { fill: #00FFFF; }
            .fill_lightblue { fill: #00AAFF; }
            .fill_blue { fill: #0000FF; }
            .fill_indigo { fill: #4400FF; }
            .fill_violet { fill: #AA00FF; }
            .fill_none { fill: none;}

            .fill_garnet { fill: #ca2020;}
            .fill_buttercup { fill: #dbac01;}
            .fill_jade { fill: #05776a;}
            .fill_topaz { fill: #246bcd;}

            .fill_alpha_none { fill-opacity: 0.0;}
            .fill_alpha_25pc { fill-opacity: 0.25;}
            .fill_alpha_50pc { fill-opacity: 0.5;}
            .fill_alpha_100pc { fill-opacity: 1.0;}

            .stroke_white { stroke: white; }
            .stroke_black { stroke: black; }
            .stroke_red { stroke: #FF3333; }
            .stroke_orange { stroke: #FFAA00; }
            .stroke_yellow { stroke: #FFFF00; }
            .stroke_green { stroke: #00FF00; }
            .stroke_cyan { stroke: #00FFFF; }
            .stroke_lightblue { stroke: #00AAFF; }
            .stroke_blue { stroke: #0000FF; }
            .stroke_indigo { stroke: #4400FF;}
            .stroke_violet { stroke: #AA00FF; }
            .stroke_garnet { stroke: #ca2020;}
            .stroke_buttercup { stroke: #dbac01; }
            .stroke_jade { stroke: #05776a; }
            .stroke_topaz { stroke: #246bcd; }
            .stroke_none { stroke: none;}

            .stroke_width_thin { stroke-width:0.05; }
            .stroke_width_mid { stroke-width:0.1; }
            .stroke_width_wide { stroke-width:0.5; }

        </style>"""
    return css_style
