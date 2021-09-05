


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
            if leaf_count == 0 :
                leaf_count = 1
            if full_depth == 0 :
                full_depth = 1
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



def tree_layout_with_single_final_assignment_layer(dag):
    # Extract the top layers which should form a branching tree,
    # and finally arrange the bottom layer to evenly distribute the remaining nodes
    # in a reasonable manner
    top_dag = dag.subgraph([k for k,n in dag.nodes.items() if not n.out_degree()==0])
    top_dag = daglit.layouts.pad_short_leaf_nodes(top_dag)

    bottom_dag = dag.subgraph([k for k,n in dag.nodes.items() if n.out_degree()==0])


    top_posns=daglit.layouts.tree_layout(top_dag, hide_virtual=False)


    edge_list = [ e for e in dag.edges if (e[0] in bottom_dag.nodes.keys() or e[1] in bottom_dag.nodes.keys())]


    sum_count={}
    for s,f in edge_list:
        if f in sum_count:
            sum_count[f]=(sum_count[f][0]+top_posns[s][0],sum_count[f][1]+1)
        else:
            sum_count[f]=(top_posns[s][0],1)
    for s,f in sum_count.items():
        sum_count[s]=f[0]/f[1]

    d_step = 1 / (len(sum_count))
    d_step_threshold = 1 / (len(sum_count)*2)
    v_step = max(top_dag.node_depth_map().values())
    v_loc = (1 / v_step) * (v_step + 4)
    m_positions=sorted(sum_count.items(), key=lambda x: x[1])
    mp_diffs=[m_positions[i][1]-m_positions[i-1][1] for i in range(1,len(m_positions))]
    if any([p<d_step for p in mp_diffs]):
        bottom_dag_posns = { k[0] : (d_step * (e+1),v_loc)  for e,k in enumerate(m_positions)}
    else:
        bottom_dag_posns = { k[0] : (k[1],v_loc)  for e,k in enumerate(m_positions)}
    posns = {**top_posns, **bottom_dag_posns}


    return posns


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
