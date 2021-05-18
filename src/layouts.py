import daglit.digraph

def rescale_positions(positions_d, xy_bounds):
    minx, maxx, miny, maxy = xy_bounds
    xvals, yvals = zip(*[v for v in positions_d.values()])
    minxvals, maxxvals, minyvals, maxyvals = min(xvals), max(xvals), min(yvals), max(yvals)
    xrangvals, yrangvals = maxxvals-minxvals, maxyvals-minyvals
    xrang, yrang = maxx-minx, maxy-miny
    if xrangvals == 0:
        xrangvals = 1
    if yrangvals == 0:
        yrangvals = 1

    return { k : ((((v[0]-minxvals)/xrangvals)*xrang)+minx, (((v[1]-minyvals)/yrangvals)*yrang)+miny) for k,v in positions_d.items() }

def pad_short_leaf_nodes(dag):
    d = dag.copy()
    v=0

    # Find any nodes that consolidate, rather than branch, and add new virtual leaf nodes to consolidate.
    original_nodes = set(d.nodes)
    #for n in original_nodes:
    #    cps = d.coparents(n)
    #    if len(cps)>len(d.nodes[n].successors):
    #        for cp in cps-set(n):
    #            v=v+1
    #            d.add_edge((n,"__virtual_node_{v}".format(v=v)))

    t_sort = list(d.topological_sort(True))
    d_map = d.node_depth_map()
    max_depth = max((v for v in d_map.values()))
    # Extend early-closing leaf nodes to max-depth
    for t in d.leaf_nodes():
        n=t
        for l in range(d_map[t], max_depth):
            v=v+1
            d.add_edge((n, "__virtual_node_{v}".format(v=v)))
            n="__virtual_node_{v}".format(v=v)
    return d


def tree_layout(dag, hide_virtual=True,reverse=False):
    # Build a tree from the leaves up, positioning parents at a point
    # determined as the centre of the x-coordinates of all its children
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

    full_depth = max(depths.values())

    while len(leaves)>0:
        # Pick the lowest node
        node = [l for l in t_sort if l in leaves].pop()
        # Add the parents of this node to the current parents collection
        #parents=parents.union(set(d.nodes[node].predecessors.keys()))
        # Extract all siblings of the current node
        sibs = d.siblings(node)
        if len(sibs)==0:
            sibs = set([node])
        sibs = sibs - set(posns.keys())
        leaves=leaves-sibs

        for i in sorted(sibs):
            c=c+1
            ordering[i]=c
            # Add the parents of this node to the current parents collection
            parents=parents.union(set(d.nodes[i].predecessors.keys()))
            if len(d.nodes[i].successors)==0:
                if not reverse:
                    posns[i]= ((c/leaf_count), (depths[i]/full_depth))
                else:
                    posns[i]= ((c/leaf_count), 1-(depths[i]/full_depth))
            else:
                children = d.nodes[i].successors
                if all([c in posns.keys() for c in children]):
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

def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)

def tree_layout_2(dag, hide_virtual=True,reverse=False):
    # Build a tree from the leaves up, positioning parents at a point
    # determined as the centre of the x-coordinates of all its children
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

    full_depth = max(depths.values())

    while len(leaves)>0:
        # Pick the lowest node
        leaves_l=list(leaves)
        npm=dag.node_proximity_matrix(leaves_l)

        sorted_leaves = [leaves_l[a] for a in argsort([sum(row) for row in npm])][::-1]
        node = [l for l in sorted_leaves].pop()
        for i in sorted_leaves:
            c=c+1
            ordering[i]=c
            # Add the parents of this node to the current parents collection
            parents=parents.union(set(d.nodes[i].predecessors.keys()))
            if len(d.nodes[i].successors)==0: # This is a full leaf node, so lay out in a linear spread
#                print("bottom row", i)
                if not reverse:
                    posns[i]= ((c/leaf_count), (depths[i]/full_depth))
                else:
                    posns[i]= ((c/leaf_count), 1-(depths[i]/full_depth))
            else:
                children = d.nodes[i].successors
                if all([c in posns.keys() for c in children]):
#                    print ( "children of ", i, ":", d.nodes[i].successors)

                    xpos = sum([posns.get(l,[0,0])[0] for l in children])/len(children)
                    if not reverse:
                        ypos = depths[i]/full_depth
                    else:
                        ypos = 1- depths[i]/full_depth
                    posns[i] = (xpos,ypos)


        leaves = parents-set(posns.keys())
        parents = set()
    if hide_virtual:
        return {k:v for k,v in posns.items() if k in dag.nodes}
    else:
        return posns

def render_graph_svg(component_id, graph, posns, hide_virtual=True):
    if component_id is None:
        component_id = graph.name

    node_circle = """<circle class="stroke_topaz fill_lightblue stroke_width_thin" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    open_circle = """<circle class="stroke_black fill_white stroke_width_thin" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    virtual_node_circle = """<circle class="stroke_green fill_avocado stroke_width_thin" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    cext_t = """<text x="{cx}" y="{cy}" text-anchor="middle" alignment-baseline="middle">{text}</text>"""
    nodes_svg = ""
#    edge_line = """<line class="stroke_black stroke_width_thin" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)"/>"""
    edge_line = """<path class="stroke_black stroke_width_thin" d="M{x1} {y1} L{x2} {y2}" marker-end="url(#arrow)"/>"""
    for edge in graph.edges:
        if ("__virtual" not in edge[1] or not hide_virtual):
            edge_start_x, edge_start_y, edge_end_x, edge_end_y = posns[edge[0]][0], posns[edge[0]][1], posns[edge[1]][0], posns[edge[1]][1]
            #nodes_svg = nodes_svg + edge_line.format(x1=edge_start_x, y1=edge_start_y, x2=edge_end_x, y2=edge_end_y )
            x_diff, y_diff = (edge_end_x-edge_start_x), (edge_end_y-edge_start_y)
            line_dist = (x_diff**2 + y_diff**2)**(1/2)
            radius_offset = 1-(0.4/line_dist)
            nodes_svg = nodes_svg + edge_line.format(x1=edge_start_x, y1=edge_start_y, x2=edge_start_x+(x_diff*radius_offset), y2=edge_start_y+(y_diff*radius_offset) )

        else:
            pass


    for node, pos in posns.items():
        if "__virtual" not in node :
            nodes_svg = nodes_svg + node_circle.format(r=0.3, cx=pos[0], cy=pos[1])
            nodes_svg = nodes_svg + cext_t.format(cx=pos[0], cy=pos[1], text=str(node))
        elif not hide_virtual:
            nodes_svg = nodes_svg + virtual_node_circle.format(r=0.3, cx=pos[0], cy=pos[1])
            nodes_svg = nodes_svg + cext_t.format(cx=pos[0], cy=pos[1], text=str(node[-3:]))

    return  """<g id="{component_id}"><g>{contents}</g></g>""".format(component_id=component_id, contents=nodes_svg)


def svg_base(width_height=(500,500), viewbox=(0,0,10,10), contents=None):
    viewbox = ", ".join([str(v) for v in viewbox])
    width, height = width_height
    defs = """<defs>
        <marker id="arrow" markerWidth="1" markerHeight="1" refX="0", refY="3" orient="auto" overflow="visible" markerUnits="strokeWidth">
        <path d="M-7,0 L-7,6 L0,3 z" fill="#fff" stroke="black" fill-opacity="1.0"/>
        </marker>
        </defs>"""
    svg_chunk = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg"

    xmlns:xlink="http://www.w3.org/1999/xlink" x="0" y="0"
    width="{width}" height="{height}" viewBox="{viewbox}">
    {defs}
      {contents}
      </svg >
        """.format(width=width, height=height, viewbox=viewbox, defs=defs, contents=contents)
    return svg_chunk



def css_style():
    css_style = """
        <style>
            text {font-size:0.03em;}
            .small_text {font-size:0.01em;}
            .fill_white { fill: white; }
            .fill_black { fill: black; }
            .fill_red { fill: #FF3333; }
            .fill_orange { fill: #FFAA00; }
            .fill_yellow { fill: #FFFF00; }
            .fill_green { fill: #00FF00; }
            .fill_avocado { fill: #9bd975; }
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
            .stroke_avocado { stroke: #9bd975; }
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

            .stroke_width_fine { stroke-width:0.03; }
            .stroke_width_thin { stroke-width:0.05; }
            .stroke_width_mid { stroke-width:0.1; }
            .stroke_width_wide { stroke-width:0.5; }

        </style>"""
    return css_style
