from itertools import chain
from collections import Counter

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

    def flow_class(self):
        class_names = { 0 : "Singleton",
                        1 : "Sink",
                        2 : "Source",
                        3 : "Flow"}
        in_degree  = 0  if len(self.predecessors) == 0 else 1
        out_degree = 0 if len(self.successors) == 0 else 1
        #print ( self.name, in_degree, out_degree)
        io_class = (1 * in_degree) + (2 * out_degree)
        return class_names[io_class]

    def edges(self):
        edgelist = []
        for p in self.predecessors.keys():
            edgelist.append((p,self.name))
        for s in self.successors.keys():
            edgelist.append((self.name, s))
        return edgelist


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
    def __init__(self, name=None, acyclic=True):
        self.name = name
        self.nodes={}
        self.edges={}
        self.acyclic=acyclic



    def __next__(self):
        for k,v in self.nodes.items():
            yield v

    @staticmethod
    def from_dict(node_edges_dict,acyclic=True,name=None):
        d=DiGraph(name,acyclic)
        for n,successors in node_edges_dict.items():
            for s in successors:
                d.add_edge((n,s))
            if len(successors)==0:
                d.add_node(n)
        return d

    def to_dict(self):
        return {n: [s for s in self.nodes[n].successors] for n in self.node_depth_map()  }

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
        if (new_edge.node_to in self.ancestors(new_edge.node_from)) and self.acyclic==True:
            raise EdgeCreatesCycleError(edge)

        self.edges[edge]=new_edge
        self.nodes[new_edge.node_from].successors.update({new_edge.node_to:self.nodes[new_edge.node_to]})
        self.nodes[new_edge.node_to].predecessors.update({new_edge.node_from:self.nodes[new_edge.node_from]})

        self.nodes[new_edge.node_from]._update_neighbors()
        self.nodes[new_edge.node_to]._update_neighbors()

    def adjacency_matrix(self, nodelist=None):
        """Return the adjacency matrix for the graph in nested list format, along with a list of node-labels to align to xy positions."""
        if nodelist is None:
            nodelist=list(self.nodes.keys())
        matrix = [[0] * len(nodelist) for e,n in enumerate(nodelist)]
        for e in self.edges:
            x,y=nodelist.index(e[0]), nodelist.index(e[1])
            matrix[x][y]=matrix[x][y]+1
        return matrix, nodelist

    def transition_matrix(self, nodelist=None,edge_metric=None,transpose=True,randomise_flow=False):
        """Returns the markov transitional matrix associated with the graph.
        If randomise_flow is set, then a diagonal of 1's is inserted into the adjancency matrix to allow for
        movement from each node to any adjacent nodes to become optional and not mandatory. """
        if nodelist is None:
            nodelist=list(self.nodes.keys())
        matrix = [[0] * len(nodelist) for e,n in enumerate(nodelist)]
        for e,o in self.edges.items():
            x,y=nodelist.index(e[0]), nodelist.index(e[1])
            matrix[x][y]=matrix[x][y]+o.data.get(edge_metric,1)
        if randomise_flow:
            for x in range(0,len(nodelist)):
                if sum(matrix[x])>0:
                    matrix[x][x]=sum(matrix[x])
                else:
                    matrix[x][x]=1
        qmatrix=list()
        for r in matrix:
            qmatrix.append([(e/sum(r)) if sum(r)!=0 else 0 for e in r])
        if transpose:
            return _transpose_matrix(qmatrix)
        else:
            return qmatrix

    @staticmethod
    def _transpose_matrix(matrix):
        return list(map(lambda *a: list(a), *matrix))



    def root_nodes(self):
        """Retrieve a list of node-names that lie at the start of the DAG - i.e. that have no predecessors"""
        leaves = []
        for k,n in self.nodes.items():
            if len(n.predecessors)==0:
                leaves.append(n.name)
        return set(leaves)

    def leaf_nodes(self):
        """Retrieve a list of node-names that lie at the end of the DAG - i.e. that have no successors"""
        roots=[]
        for k,n in self.nodes.items():
            if len(n.successors)==0:
                roots.append(n.name)
        return set(roots)

    def nodes_degree(self, degree=None, names=True, in_out=None):
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

    def nodes_in_degree(self, degree=None, names=True):
        return self.nodes_degree(degree=degree, names=names, in_out="in")

    def nodes_out_degree(self, degree=None, names=True):
        return self.nodes_degree(degree=degree, names=names, in_out="out")

    def ancestors(self, node, names=True):
        return set(self.connected_nodes(node, names=names, direction="up"))

    def descendents(self, node, names=True):
        return set(self.connected_nodes(node, names=names, direction="down"))

    # The siblings of a node are those that share a common parent (or parents)
    def siblings(self, node, names=True):
        siblings = set()
        if isinstance(node, Node):
            parents = node.predecessors
        else:
            parents = self.nodes[node].predecessors
        for p in parents:
            for s in self.nodes[p].successors:
                siblings.add(s)
        return siblings

    # coparents are the reverse analog of siblings - a pair (or more) of parents of the same child node
    def coparents(self, node, names=True):
        coparents = set()
        if isinstance(node, Node):
            children = node.successors
        else:
            children = self.nodes[node].successors
        for c in children:
            for s in self.nodes[c].predecessors:
                coparents.add(s)
        return coparents



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
        return set(accumulator)

    def regions(self, containing=None):
        unprocessed = set(self.nodes.keys())
        regions = []
        while len(unprocessed)>0:
            node = unprocessed.pop()
            current_region = set(self.connected_nodes(node,names=True))
            if current_region == set():
                current_region.add(node)
            unprocessed=unprocessed-set(current_region)
            regions.append(current_region)

        if containing is None:
            return regions
        else:
            for reg in regions:
                if containing in reg:
                    return reg
            return None

    # Create a deep copy of the graph, using only the nodes referenced in the node_collection variable
    def subgraph(self, node_collection, name=None):
        sg = DiGraph(name,acyclic=self.acyclic)
        for e in self.edges:
            if e[0] in node_collection and e[1] in node_collection:
                sg.add_edge(e,self.edges[e].data)
                for n in e:
                    sg.nodes[n].data = self.nodes[n].data
        return sg



    def cycle_members(self):
        """Identify which nodes participate in cycles, by finding nodes who are their own descendents.
           Returns a set of (frozen)sets (normal sets don't hash), each consisting of a disjoint cycle membership."""
        loop_members=set()
        processed = set()
        for k,n in self.nodes.items():
            if k not in processed:
                cycle = (self.ancestors(k).intersection(self.descendents(k)))
                if len(cycle)>0:
                    loop_members.add(frozenset(cycle))
                    processed=processed.union(cycle)
        if len (loop_members)==0:
            return set()
        else:
            return loop_members

    def has_cycle(self):
        if len(self.cycle_members())>0:
            return True
        else:
            return False

    def consolidate_cycles(self):
        # Identify existing cycles and consolidate these into virtual "cyclic" nodes.
        # After performing, the remaining graph can be expressed as a DAG without any cycles.
        # The virtual DAG nodes should be unpackable from the virtual storage.
        # This consolidated version of the DAG can be used to help perform layout arrangements by
        # splitting the graph down into component elements.
        cycle_nodes = []
        cyclic_nodes = set(chain(*self.cycle_members()))
        non_cycle_nodes = set([n for n in self.nodes.keys() if n not in cyclic_nodes])
        non_cycle_edges = set([e for e in self.edges.keys() if not any([x in cyclic_nodes for x in e])])
        node_translation_map = {n:n for n in self.nodes.keys()}
        c_graph = DiGraph()
        for n in non_cycle_nodes:
            c_graph.add_node(n, data=self.nodes[n].data)
        for e in non_cycle_edges:
            c_graph.add_edge(e, data=self.edges[e].data)
        cm_list = list(self.cycle_members())

        for e,cycle in enumerate(cm_list):
            c_name = "__cycle_{e}".format(e=e)
            for n in cycle:
                node_translation_map[n]=c_name

        for e,cycle in enumerate(cm_list):
            data_payload = {}
            c_name = "__cycle_{e}".format(e=e)
            for member in cycle:
                node_translation_map[member]=c_name
                data_payload[member]={**{"edges" : self.nodes[member].edges()},
                                         "data" : self.nodes[member].data}

                for edge in self.nodes[member].edges():
                    if (len([n for n in edge if n in non_cycle_nodes])==1 and len([n for n in edge if n in cyclic_nodes])==1):
                        new_edge = (node_translation_map[edge[0]], node_translation_map[edge[1]])
                        c_graph.add_edge(new_edge, data=self.edges[edge].data)
                    elif node_translation_map[edge[0]]!= node_translation_map[edge[1]] and len([n for n in edge if n in non_cycle_nodes])==2:
                        new_edge = (node_translation_map[edge[0]], node_translation_map[edge[1]])
                        c_graph.add_edge(new_edge, data=self.edges[edge].data)
            if not c_name in c_graph.nodes.keys():
                c_graph.add_node(c_name, data={"members" : data_payload})
            else:
                c_graph.nodes[c_name].data={"members" : data_payload}
        return c_graph




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
        singletons = { k: 0 for k in self.nodes_degree(0,names=True) }
        c_map.update(singletons)
        return c_map

    def reversed(self):
        """Return a new object with the same node structure, but with all edge directions reversed."""
        return self.copy(reverse_nodes=True)


    def copy(self, reverse_nodes=False):
        """Return a new object with the same node and edge structure - all data components are preserved."""
        cdag = DiGraph(self.name,self.acyclic)
        for k,n in self.nodes.items():
            cdag.add_node(k, data=n.data)
        if not reverse_nodes:
            for k,e in self.edges.items():
                cdag.add_edge(k,data=e.data)
        else:
            for k,e in self.edges.items():
                cdag.add_edge((k[1],k[0]),data=e.data)
        return cdag


    def topological_sort(self, lexicographical=False):

        ordered_list = []
        if lexicographical:
            root_layer = sorted(self.root_nodes())
        else:
            root_layer = set(self.root_nodes())
        remaining_set = set(self.nodes.keys())
        processed_set = []
        while len(root_layer)>0:
            if lexicographical:
                root_layer=sorted(root_layer)[::-1]
            n = root_layer.pop()
            remaining_set.remove(n)
            processed_set.append(n)
            yield n
            for m in ( node for node in self.nodes[n].successors if node in remaining_set):
                if all([p in processed_set for p in self.nodes[m].predecessors] + [True]):
                    if lexicographical:
                        root_layer.append(m)
                    else:
                        root_layer.add(m)
        if len(remaining_set)>0 :
            raise DAGContainsCycleError(processed_set)





    def simple_arborescence(self):
        node_order = self.topological_sort()
        arb=DiGraph()
        processed=set()
        for n in node_order:
            for m in self.nodes[n].successors:
                if m not in processed:
                    arb.add_edge((n,m))
                    processed.add(m)
            if n not in processed:
                processed.add(n)
                arb.add_node(n)
        return arb


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
        if not self.acyclic:
            # Find and break any extant cycles before proceeding.
            pass
        if accumulator is None:
            accumulator=[]
        if node is None:
            while (len(set(accumulator)) != len(set(self.nodes))) :

                #print(accumulator, len(set(accumulator)) , len(set(self.nodes)), set(self.root_nodes()))
                try:
                    node = (set(self.root_nodes())-set(accumulator)).pop()
                except Exception as e:
                    if not self.acyclic:
                        print ( "Picking node from set", (set(self.nodes.keys())-set(accumulator)) )
                        node = (set(self.nodes.keys())-set(accumulator)).pop()
                    else:
                        raise e
                for next_node in self.depth_first_sort(node,accumulator):

                    if next_node not in set(accumulator) and not self.acyclic:
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
