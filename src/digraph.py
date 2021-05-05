
class DAGError(Exception):
    "General Purpose Directed Acyclic Graph Error"
    pass

class DAGNotBipartite(Exception):
    "This DAG is not Bipartite"
    pass


class EdgeCreatesCycleError(Exception):
    "Addition of this edge creates a cycle"
    pass


class DAGContainsCycleError(Exception):
    "Directed Acyclic Graph is cyclic"
    pass


class Node():
    def __init__(self, name, successors=None, predecessors=None, data=None):
        if successors is None:
            successors = {}
        if predecessors is None:
            predecessors = {}
        if data is None:
            data = {}
        self.name = name
        self.successors = successors
        self.predecessors = predecessors
        self.data = data
        self._update_neighbors()

    def _update_neighbors(self):
        self.neighbors = {**self.successors , **self.predecessors}

    def in_degree(self):
        return len(self.predecessors)

    def out_degree(self):
        return len(self.successors)

    def degree(self):
        return len(self.predecessors) + len(self.successors)

class Edge():
    def __init__(self, node_from, node_to, data=None):
        self.edge = (node_from, node_to)
        self.node_from = node_from
        self.node_to = node_to
        if data is None:
            data = {}
        self.data = data

class Graph():
    """
    A Graph is a data structure consisting of an ordered pair of sets V and E
    G = (V,E)
    Where V is a set of elements called Vertices (or Nodes), and
    E is another set of elements called Edges, Each edge is a pair taken from V,
    describing endpoints in V belonging to each edge.
    """
    pass

class DiGraph(Graph):
    """
    :class: DiGraph - A DiGraph class represents a Directed (Acyclic) Graph.
    :param name: defaults to none, enables reference by name


    """
    def __init__(self, name=None):
        self.name = name
        self.nodes={}
        self.edges={}



    def __next__(self):
        for k,v in self.nodes.items():
            yield v

    def from_dict(self, nested_dict, root="root"):
        pass

    def add_node(self, name, data=None):
        """Adds a node to the current graph with some unique name and optional data payload
        :param name: hashable, gives the node a name
        :param data: dict, optional dictionary of key-value pairs to be associated with this node
        """
        if data is None:
            data = {}
        if name not in self.nodes.keys():
            node = Node(name, data=data)
            self.nodes[name]=node

    def delete_node(self, name):
        """Removes a named node from the graph, along with any associated edges and information"""
        del self.nodes[name]
        del_candidates=[]
        neighbors_to_update = set()

        for k,n in self.nodes.items():
            if name in n.predecessors.keys():
                del n.predecessors[name]
                neighbors_to_update.add(n.name)

            if name in n.successors.keys():
                del n.successors[name]
                neighbors_to_update.add(n.name)

        for k,e in self.edges.items():
            if name in e.edge:
                del_candidates.append(e)

        for e in del_candidates:
            del self.edges[e.edge]

        for n in neighbors_to_update:
            if n in self.nodes:
                self.nodes[n]._update_neighbors()





    def add_edge(self, edge, data=None):
        """Adds an edge to a DiGraph - if any node in the edge definition isn't already extant, it will be created.
        :param edge: Each edge is defined as a tuple of node-names in the order (from, to)
        :type edge: tuple(from_node, to_node) -
        :param data: A dictionary of key-value pairs to be associated with this edge
        :type data: dict, optional
        :raises EdgeCreatesCycleError: Protection from creating a cycle - where attempted throws EdgeCreatesCycleError
        """
        if data is None:
            data = {}
        new_edge = Edge(*edge, data)
        for e in edge:
            if e not in self.nodes.keys():
                self.add_node(e, data)
        if self.nodes[new_edge.node_to] in self.ancestors(new_edge.node_from):
            raise EdgeCreatesCycleError(edge)
        self.edges[edge]=new_edge
        self.nodes[new_edge.node_from].successors.update({new_edge.node_to:self.nodes[new_edge.node_to]})
        self.nodes[new_edge.node_to].predecessors.update({new_edge.node_from:self.nodes[new_edge.node_from]})

        self.nodes[new_edge.node_from]._update_neighbors()
        self.nodes[new_edge.node_to]._update_neighbors()

    def adjacency_matrix(self, nodelist=None):
        """Return the adjacency matrix for the graph in nested list format, along with a list of node-labels to align to xy positions."""
        if nodelist is None:
            nodelist=list(self.topological_sort(lexicographical=True))
        matrix = [[0] * len(nodelist) for e,n in enumerate(nodelist)]
        for e in self.edges:
            x,y=nodelist.index(e[0]), nodelist.index(e[1])
            matrix[x][y]=matrix[x][y]+1
        return matrix, nodelist


    def root_nodes(self):
        """Retrieve a list of node-names that lie at the end of the DAG - i.e. that have no predecessors"""
        leaves = []
        for k,n in self.nodes.items():
            if len(n.predecessors)==0:
                leaves.append(n.name)
        return leaves

    def leaf_nodes(self):
        """Retrieve a list of node-names that lie at the end of the DAG - i.e. that have no successors"""
        roots=[]
        for k,n in self.nodes.items():
            if len(n.successors)==0:
                roots.append(n.name)
        return roots

    def node_degree(self, degree=None, names=False, in_out=None):
        """:parm degree: None, or int - a selector applied as a filter
        :parm names: bool, return Names if True, or Node objects if False
        :parm in_out: None, bool - specify whether in, out or both degree measures are to be measured
        """
        d_func_map = { None : "degree",
                       "in" : "in_degree",
                       "out" : "out_degree" }

        if degree is not None:
            node_list = []
            for k,n in self.nodes.items():
                #if n.degree()==degree:
                if getattr(n,d_func_map[in_out])()==degree:
                    if names:
                        node_list.append(n.name)
                    else:
                        node_list.append(n)
            return node_list
        else:
            node_d = {}
            for k,n in self.nodes.items():
                #dk=n.degree()
                dk=getattr(n,d_func_map[in_out])()
                if dk in node_d.keys():
                    if names:
                        node_d[dk].append(n.name)
                    else:
                        node_d[dk].append(n)
                else:
                    if names:
                        node_d[dk]=[n.name]
                    else:
                        node_d[dk]=[n]
            return node_d

    def node_in_degree(self, degree=None, names=False):
        return self.node_degree(degree=degree, names=names, in_out="in")

    def node_out_degree(self, degree=None, names=False):
        return self.node_degree(degree=degree, names=names, in_out="out")

    def ancestors(self, node, names=False):
        return self.connected_nodes(node, names=names, direction="up")

    def descendents(self, node, names=False):
        return self.connected_nodes(node, names=names, direction="down")

    def connected_nodes(self, node, names=False, direction=None):
        """For a given node, find all nodes connected to it according to the direction parameter.
        If direction is `up`, then look for all predecessors (ancestors), if direction is `down` then
        find all successors (descendents) reachable from this node, and if direction is `None`,
        return all ancestors and descendents associated with this node.
        """
        accumulator=[]
        processed = set()

        d_func_map = { None : "neighbors",
                       "up" : "predecessors",
                       "down" : "successors"
                       }

        if isinstance(node, str):
            name = node
        elif isinstance(node, Node):
            name = node.name

        if names:
            unprocessed = set([k for k in getattr(self.nodes[name],d_func_map[direction]).keys()])
        else:
            unprocessed = set([v for v in getattr(self.nodes[name],d_func_map[direction]).values()])

        stack = set([i for i in unprocessed])
        while len(stack) > 0:
            for k in stack:
                accumulator.append(k)
                processed.add(k)
                if names:
                    for pk in getattr(self.nodes[k],d_func_map[direction]).keys():
                        unprocessed.add(pk)
                else:
                    for pk in getattr(self.nodes[k.name],d_func_map[direction]).values():
                        unprocessed.add(pk)
                unprocessed.remove(k)
            stack = set([i for i in unprocessed])-set(processed)
        return accumulator

    def regions(self):
        unprocessed = set(self.nodes.keys())
        regions = []
        while len(unprocessed)>0:
            node = unprocessed.pop()
            current_region = set(self.connected_nodes(node,names=True))
            if current_region == set():
                current_region.add(node)
            unprocessed=unprocessed-set(current_region)
            regions.append(current_region)
        return regions

    def subgraph(self, node_collection, name=None):
        sg = DiGraph(name)
        for e in self.edges:
            if e[0] in node_collection and e[1] in node_collection:
                sg.add_edge(e,self.edges[e].data)
                for n in e:
                    sg.nodes[n].data = self.nodes[n].data
        return sg


    def all_simple_paths_between(self, source, target, cutoff=None):
        if cutoff is not None and cutoff < 1:
            return
        if cutoff is None:
            cutoff = len(self.nodes)-1
        visited = [source]
        stack = [iter(self.nodes[source].successors)]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child == target:
                    yield visited + [ target ]
                elif child not in visited:
                    visited.append(child)
                    stack.append(iter(self.nodes[child].successors))
            else:
                if child == target or target in children:
                    yield visited + [ target ]
                stack.pop()
                visited.pop()

    def paths(self):
        for s_node in self.nodes.keys():
            for t_node in self.nodes.keys():
                vp_iter=self.all_simple_paths_between(s_node, t_node)
                for p in vp_iter:
                    yield p
        for singleton in self.node_degree(0, names=True):
            yield singleton


    def full_paths(self):
        for s_node in self.root_nodes():
            for t_node in self.leaf_nodes():
                vp_iter=self.all_simple_paths_between(s_node, t_node)
                for p in vp_iter:
                    yield p
        for singleton in self.node_degree(0, names=True):
            yield singleton

    def edge_flows(self):
        edge_flows_d={}
        for p in self.full_paths():
            for i in range(0,len(p)-1):
                fn, tn = p[i], p[i+1]
                edge_flows_d[(fn,tn)]=edge_flows_d.get((fn,tn),0)+1
        return edge_flows_d

    def longest_paths(self):
        paths = list(self.full_paths())
        plen = [len(p) for p in paths]
        pind = [e for e,i in enumerate(plen) if i==max(plen)]
        return [paths[e] for e in pind]

    def longest_path_between(self, source, target):
        pathlist = list(self.all_simple_paths_between(source, target))
        plen=[len(p) for p in pathlist]
        if len(plen) > 0:
            return pathlist[plen.index(max(plen))]
        else:
            return []

    def shortest_path_between(self, source, target):
        pathlist = list(self.all_simple_paths_between(source, target))
        plen=[len(p) for p in pathlist]
        if len(plen) > 0:
            return pathlist[plen.index(min(plen))]
        else:
            return []

    def internode_distance(self, source, target):
        path = self.shortest_path_between(source, target)
        return len(path)

    def has_cycle(self):
        try:
            s = self.topological_sort()
            return False

        except DAGContainsCycleError:
            return True

    def is_bipartite(self):
        try:
            self.attempt_bi_colouring()
            return True
        except DAGNotBipartite as e:
            return False


    def attempt_bi_colouring(self):
        c_map = {}
        for n in self.nodes.keys():
            if n in c_map or len(self.nodes[n].neighbors)==0:
                continue
            else:
                queue = [n]
                c_map[n]=1

                while queue:
                    v = queue.pop()
                    c = 1 - c_map[v]
                    for w in self.nodes[v].neighbors:

                        if w in c_map:
                            if c_map[w] == c_map[v]:
                                raise DAGNotBipartite(c_map,w,v)
                        else:
                            c_map[w]=c

                            queue.append(w)
        singletons = { k: 0 for k in self.node_degree(0,names=True) }
        c_map.update(singletons)
        return c_map

    def reversed(self):
        """Return a new object with the same node structure, but with all edge directions reversed."""
        return self.copy(reverse_nodes=True)


    def copy(self, reverse_nodes=False):
        """Return a new object with the same node and edge structure - all data components are preserved."""
        cdag = DiGraph(self.name)
        for k,n in self.nodes.items():
            cdag.add_node(k, data=n.data)
        if not reverse_nodes:
            for k,e in self.edges.items():
                cdag.add_edge(k,data=e.data)
        else:
            for k,e in self.edges.items():
                cdag.add_edge((k[1],k[0]),data=e.data)
        return cdag


    def node_depth_map(self,minimum=False):
        if minimum:
            test_val=float('inf')
        else:
            test_val=0
        level_d = {}
        for n in self.depth_first_sort():
            if minimum:
                for l in self.root_nodes():
                    if l == n:
                        level_d[n]=0
                    elif l in self.ancestors(n,names=True):
                        if level_d.get(n,test_val)>len(self.shortest_path_between(l,n))-1:
                            level_d[n]=len(self.shortest_path_between(l,n))-1
            else:

                for l in self.root_nodes():
                    if l == n:
                        level_d[n]=0
                    elif l in self.ancestors(n,names=True):
                        if level_d.get(n,test_val)<len(self.longest_path_between(l,n))-1:
                            level_d[n]=len(self.longest_path_between(l,n))-1
        return level_d

    def topological_sort(self, lexicographical=False):
        ordered_list = []
        if lexicographical:
            root_layer = sorted(self.root_nodes())
        else:
            root_layer = set(self.root_nodes())
        remaining_set = set(self.nodes.keys())
        processed_set = set()
        while len(root_layer)>0:
            if lexicographical:
                root_layer=sorted(root_layer)[::-1]
            n = root_layer.pop()
            remaining_set.remove(n)
            processed_set.add(n)
            yield n
            for m in ( node for node in self.nodes[n].successors if node in remaining_set):
                if all([p in processed_set for p in self.nodes[m].predecessors] + [True]):
                    if lexicographical:
                        root_layer.append(m)
                    else:
                        root_layer.add(m)
        if len(remaining_set)>0:
            raise DAGContainsCycleError

    def breadth_first_sort(self,resolve_children=True):

        next_layer=set()
        processed_set=set()
        seen=set()

        if resolve_children:
            level = set(self.root_nodes())
            while len(level) is not 0:
                for n in level:
                    if n not in processed_set:
                        yield n
                    for child in self.nodes[n].successors:
                        next_layer.add(child)
                    processed_set.add(n)

                level=level-processed_set
                level=level.union(next_layer)

                next_layer = set()
        else:
            level = set(self.leaf_nodes())
            while len(level) is not 0:
                for n in level:
                    if n not in processed_set:
                        yield n
                    for parent in self.nodes[n].predecessors:
                        seen.add(n)
                        if all([c in seen for c in self.nodes[parent].successors ]):
                            next_layer.add(parent)
                    processed_set.add(n)

                level=level-processed_set
                level=level.union(next_layer)

                next_layer = set()




    def depth_first_sort(self, node=None, accumulator=None):

        if accumulator is None:
            accumulator=[]
        if node is None:
            while (len(set(accumulator)) != len(set(self.nodes))) :

                #print(accumulator, len(set(accumulator)) , len(set(self.nodes)), set(self.root_nodes()))
                node = (set(self.root_nodes())-set(accumulator)).pop()
                for next_node in self.depth_first_sort(node):
                    if next_node not in set(accumulator):
                        accumulator.append(next_node)

        else:
            if node not in set(accumulator):
                accumulator.append(node)


            for current_node in self.nodes[node].successors:
                if current_node not in set(accumulator):
                    accumulator.append(current_node)

                    for next_node in self.depth_first_sort(current_node, accumulator):
                        if next_node not in set(accumulator):
                            accumulator.append(next_node)

        return (v for v in accumulator)
