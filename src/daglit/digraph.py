from itertools import chain
from collections import Counter
import daglit.utils

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

class DAGContainsNoRootNode(Exception):
    "No Root Node Exists"
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

    def __eq__(self, node):
        if not isinstance(node, Node):
            return False
        name_check = self.name == node.name
        successors_check = all([k in set(k for k,v in node.successors.items()) for k,v in self.successors.items() ])
        predecessors_check = all([k  in set(k for k,v in node.predecessors.items()) for k,v in self.predecessors.items()])
        data_check = all([(k,v) in set((x,y) for x,y in node.data.items()) for k,v in self.data.items() ])
        return all([name_check, successors_check, predecessors_check, data_check])



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
    def __eq__(self, edge):
        if not isinstance(edge, Edge):
            return False
        edge_check = self.edge == edge.edge
        data_check = all([(k,v) in set((x,y) for x,y in edge.data.items()) for k,v in self.data.items() ])
        return all([edge_check,data_check])

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

    def __eq__(self, digraph,debug=True):
        name_check = self.name == digraph.name
        nodes_check = all([n==digraph.nodes.get(k,None) for k,n in self.nodes.items()]) and (len(self.nodes) == len(digraph.nodes))
        edges_check = all([e==digraph.edges.get(k,None) for k,e in self.edges.items()]) and (len(self.edges) == len(digraph.edges))
        acyclic_check = self.acyclic==digraph.acyclic
        if debug:
            print([name_check, nodes_check, edges_check, acyclic_check])
        return all([name_check, nodes_check, edges_check, acyclic_check])


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
        """Removes or 'cuts' a named node from the graph, along with any associated edges and information"""
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

    def shrink_node(self, name):
        """Shrinks the node to nothing, all edges leading into the node are
        connected directly with all nodes leading out from the node.
        Performs the task in-place, returning nothing
        """
        edges_in = [e for e in self.edges.keys() if e[1] == name]
        edges_out = [e for e in self.edges.keys() if e[0] == name]
        candidate_new_edges=[]
        for i in edges_in:
            for o in edges_out:
                candidate_new_edges.append((i[0],o[1]))
        self.delete_node(name)
        for e in candidate_new_edges:
            self.add_edge(e)

    # Shrink the graph of any nodes of degree < 3 - doing so should reveal any special
    # configurations such as K3,3 or K5.

    def core_layers(self, preserve=True):
        # Recursively reduces a graph to connection-n layers.
        # Nodes that are pruned are either done so crudely (preserve=False)
        # Or their inward/outward connections are preserved (preserve=True)
        # to maintain the conectitive structure of the graph
        C = self.copy()
        core_number = 0
        remaining=C.nodes.keys()
        layers={}
        # Remove any trailing singletons and end-point-nodes.
        while len(remaining)>0:
            while True:
                if preserve:
                    candidates = C.nodes_degree(core_number)
                else:
                    candidates = [e for k,v in C.nodes_degree().items() if k <= core_number for e in v]
                for c in candidates:
                    layers[c]=core_number
                    if preserve:
                        C.shrink_node(c)
                    else:
                        C.delete_node(c)
                if len(candidates)==0:
                    break


            if all([k>core_number for k in C.nodes_degree().keys()]):
                core_number += 1
            remaining=[(k,v.degree()) for k,v in C.nodes.items() if v.degree()!=0]


        singletons = C.nodes_degree(0)
        if len(singletons)>0:
            for s in singletons:
                C.delete_node(s)
                layers[s]=core_number+1

        return layers


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
        movement from each node to any adjacent nodes to become optional and not mandatory.
        It is customary for the transition matrix to be interpreted in a form that works like a transposition of
        the adjacency matrix, but it's possible to turn this off to align the outputs using transpose=False"""
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
            return daglit.utils._transpose_matrix(qmatrix)
        else:
            return qmatrix




    def root_nodes(self):
        """Retrieve a list of node-names that lie at the start of the DAG - i.e. that have no predecessors"""
        roots = []
        for k,n in self.nodes.items():
            if len(n.predecessors)==0:
                roots.append(n.name)
        return set(roots)

    def leaf_nodes(self):
        """Retrieve a list of node-names that lie at the end of the DAG - i.e. that have no successors"""
        leaves=[]
        for k,n in self.nodes.items():
            if len(n.successors)==0:
                leaves.append(n.name)
        return set(leaves)

    def nodes_degree(self, degree=None, in_out=None):
        """:parm degree: None, or int - a selector applied as a filter
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
                    node_list.append(n.name)

            return node_list
        else:
            node_d = {}
            for k,n in self.nodes.items():
                #dk=n.degree()
                dk=getattr(n,d_func_map[in_out])()
                if dk in node_d.keys():

                    node_d[dk].append(n.name)

                else:
                    node_d[dk]=[n.name]

            return node_d

    def nodes_in_degree(self, degree=None):
        return self.nodes_degree(degree=degree, in_out="in")

    def nodes_out_degree(self, degree=None ):
        return self.nodes_degree(degree=degree,in_out="out")

    def ancestors(self, node):
        return set(self.connected_nodes(node, direction="up"))

    def descendents(self, node):
        return set(self.connected_nodes(node, direction="down"))

    # The siblings of a node are those that share a common parent (or parents)
    # If a node is the sole child of its parent node, then it is a sibling of itself.
    def siblings(self, node):
        siblings = set()
        if isinstance(node, Node):
            parents = node.predecessors
        else:
            parents = self.nodes[node].predecessors
        for p in parents:
            for s in self.nodes[p].successors:
                siblings.add(s)
        return siblings

    # coparents are the reverse analog of siblings - find groups of nodes who share parenthood of the same child nodes
    # As with siblings, coparents always include themselves, should no other matching nodes be present.
    def coparents(self, node ):
        coparents = set()
        if isinstance(node, Node):
            children = node.successors
        else:
            children = self.nodes[node].successors
        for c in children:
            for s in self.nodes[c].predecessors:
                coparents.add(s)
        return coparents



    def connected_nodes(self, node, direction=None):
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

        unprocessed = set([v.name for v in getattr(self.nodes[name],d_func_map[direction]).values()])

        stack = set([i for i in unprocessed])

        while len(stack) > 0:
            for k in stack:
                accumulator.append(k)
                processed.add(k)
                for pk in getattr(self.nodes[k],d_func_map[direction]).values():
                    unprocessed.add(pk.name)
                unprocessed.remove(k)
            stack = set([i for i in unprocessed])-set(processed)

        return set(accumulator)-set([name])

    def regions(self, containing=None):
        unprocessed = set(self.nodes.keys())
        regions = []
        while len(unprocessed)>0:
            node = unprocessed.pop()
            current_region = set(self.connected_nodes(node)).union(set([node]))
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
        for n in self.nodes:
            if n in node_collection and n not in sg.nodes:
                sg.add_node(n, data=self.nodes[n].data)

        return sg

    def compose(self, digraph):
        comp_g = self.copy()
        for e in digraph.edges:
            comp_g.add_edge(e,digraph.edges[e].data)
            for n in e:
                comp_g.nodes[n].data = {**self.nodes.get(n,Node("__dummy")).data, **digraph.nodes[n].data}
        for n in digraph.nodes:
            if n not in comp_g.nodes:
                comp_g.add_node(n, data=digraph.nodes[n].data)
            else:
                comp_g.nodes[n].data = {**self.nodes.get(n,Node("__dummy")).data, **digraph.nodes[n].data}
        return comp_g



    def relabel(self, label_dict):
        relab_g = DiGraph(name,acyclic=self.acyclic)
        for e in self.edges:
            relab_g.add_edge((label_dict[e[0]],label_dict[e[1]]),self.edges[e].data)
            for n in e:
                relab_g.nodes[label_dict[n]].data = self.nodes[n].data
        for n in self.nodes:
            if label_dict[n] not in relab_g.nodes:
                relab_g.add_node(label_dict[n], data=self.nodes[n].data)
        return relab_g

    def possibly_isomorphic(self, digraph,debug=False):
        node_count_test = len(self.nodes)==len(digraph.nodes)
        edge_count_test = len(self.edges)==len(digraph.edges)
        node_degrees_test = {k:len(v) for k,v in self.nodes_degree().items()}=={k:len(v) for k,v in digraph.nodes_degree().items()}
        node_in_degrees_test = {k:len(v) for k,v in self.nodes_in_degree().items()}=={k:len(v) for k,v in digraph.nodes_in_degree().items()}
        node_out_degrees_test = {k:len(v) for k,v in self.nodes_out_degree().items()}=={k:len(v) for k,v in digraph.nodes_out_degree().items()}
        if debug:
            print ( [node_count_test, edge_count_test, node_degrees_test, node_in_degrees_test, node_out_degrees_test] )
        return all([node_count_test, edge_count_test, node_degrees_test, node_in_degrees_test, node_out_degrees_test])




    def cycle_members(self):
        """Identify which nodes participate in cycles, by finding nodes whose ancestors also feature as their descendents.
           Returns a set of (frozen)sets (normal sets don't hash), each consisting of a disjoint cycle membership."""
        loop_members=set()
        processed = set()
        for k,n in self.nodes.items():
            if k not in processed:
                cycle = (self.ancestors(k).intersection(self.descendents(k)))
                if len(cycle)>0:
                    loop_members.add(frozenset(cycle.union(set([k]))))
                    processed=processed.union(cycle.union(set([k])))
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
                #node_translation_map[member]=c_name
                data_payload[member]={**{"edges" : self.nodes[member].edges()},
                                         "data" : self.nodes[member].data}

                for edge in self.nodes[member].edges():
                    if (len([n for n in edge if n in non_cycle_nodes])==1 and len([n for n in edge if n in cyclic_nodes])==1):
                        new_edge = (node_translation_map[edge[0]], node_translation_map[edge[1]])
                        if new_edge[0]!=new_edge[1]:
                            c_graph.add_edge(new_edge, data=self.edges[edge].data)
                    elif node_translation_map[edge[0]]!= node_translation_map[edge[1]] and len([n for n in edge if n in cyclic_nodes])==2:
                        new_edge = (node_translation_map[edge[0]], node_translation_map[edge[1]])
                        if new_edge[0]!=new_edge[1]:
                            c_graph.add_edge(new_edge, data=self.edges[edge].data)


            if not c_name in c_graph.nodes.keys():
                c_graph.add_node(c_name, data={"members" : data_payload})
            else:
                c_graph.nodes[c_name].data={"members" : data_payload}
        print (node_translation_map)
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
        singletons = { k: 0 for k in self.nodes_degree(0) }
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
        if reverse_nodes:
            for k,e in self.edges.items():
                cdag.add_edge((k[1],k[0]),data=e.data)
        else:
            for k,e in self.edges.items():
                cdag.add_edge(k,data=e.data)
        return cdag


    def topological_sort(self, lexicographical=False):
        "Return a strict topological sort - requires the graph to have no cycles"
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

    def core_layers(self):
        """Returns a bottom-up dictionary of nodes and their layer numbers determined by repeated pruning of leaf nodes.
        If a collection of nodes remains after all leaf nodes are pruned, they must be involved in a cycle, clique or other composite arrangement.
        It should work for dis-connected graphs too.
        """
        layer_map = {}
        layer_number = 0
        layer_nodes = self.leaf_nodes()
        working_g = self.copy()
        remaining_nodes = set(self.nodes.keys())
        while not len(layer_nodes)==0:
            for n in layer_nodes:
                layer_map[n]=layer_number
            remaining_nodes=remaining_nodes-set(layer_nodes)
            working_g = working_g.subgraph(remaining_nodes)
            layer_nodes=working_g.leaf_nodes()
            layer_number+=1
        for n in remaining_nodes:
            layer_map[n]=layer_number
        return layer_map

    def min_core_distances(self):
        core_distances={}
        core_layers=self.core_layers()
        max_depth = max([v for v in core_layers.values()])
        core_nodes = set([n for n,v in core_layers.items() if v == max_depth])

        for n in self.nodes.keys():
            if n in core_nodes:
                core_distances[n]=0
            else:
                d = min([len(self.shortest_path_between(c,n))-1 for c in core_nodes])
                core_distances[n]=d
        return core_distances


    def prune_leaves(self):
        leaf_nodes = set(self.leaf_nodes())
        all_nodes = set(self.nodes.keys())
        remaining_nodes = all_nodes-leaf_nodes
        return self.subgraph(remaining_nodes)

    def node_depth_map(self):
        "Returns a dictionary of nodes, and their distances from a root node as determined by a breadth-first scan"
        depth=0
        ndm={}
        layer = self.root_nodes()
        if len(layer)==0:
            raise DAGContainsNoRootNode
        next_layer = set()
        processed_set=set()
        while len(layer)>0:
            for n in layer:
                if n not in processed_set:
                    ndm[n]=depth
                for s in self.nodes[n].successors:
                    if s not in processed_set:
                        next_layer.add(s)
                processed_set.add(n)
            depth=depth+1

            layer=list(set(next_layer)-processed_set)
            next_layer = set()

        return ndm

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

    def all_simple_paths(self,include_singletons=False):
        for s_node in self.nodes.keys():
            for t_node in self.nodes.keys():
                vp_iter=self.all_simple_paths_between(s_node, t_node)
                for p in vp_iter:
                    yield p
        if include_singletons:
            for singleton in self.nodes_degree(0):
                yield [singleton]

    def full_paths(self,include_singletons=False):
        for s_node in self.root_nodes():
            for t_node in self.leaf_nodes():
                vp_iter=self.all_simple_paths_between(s_node, t_node)
                for p in vp_iter:
                    yield p
        if include_singletons:
            for singleton in self.nodes_degree(0):
                yield [singleton]

    def node_flows(self):
        """How often does a node feature in an enumeration of all root to leaf node traversals?
           This gives an indication of the node's centrality.
           Alternately, how many such traversals exist that pass through each node?"""
        return dict(Counter(chain(*self.full_paths(include_singletons=True))))


    def edge_flows(self):
        edge_flows_d={}
        for p in self.full_paths():
            for i in range(0,len(p)-1):
                fn, tn = p[i], p[i+1]
                edge_flows_d[(fn,tn)]=edge_flows_d.get((fn,tn),0)+1
        return edge_flows_d

    def longest_paths(self, include_singletons=True):
        paths = list(self.all_simple_paths(include_singletons=include_singletons))
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

    # Given two nodes in a tree-like arrangement, what is their closest common ancestor?
    # i.e. Assuming a hierarchy, at what point can two distinct elements have a shared classification?
    def lowest_common_ancestors(self,x,y):
        x_ancestors = self.ancestors(x).union(x)
        y_ancestors = self.ancestors(y).union(y)
        ancestry_graph = self.subgraph(x_ancestors.intersection(y_ancestors))
        return ancestry_graph.leaf_nodes()

    def node_proximity_matrix(self, nodelist):
        max_proximity=len(self.nodes)+1
        matrix=[]

        for n in nodelist:
            dn=[]

            for m in nodelist:
                if n!=m:
                    lca_s = self.lowest_common_ancestors(n,m)
                    lca_count=len(lca_s)
                    if lca_count==1:
                        lca=lca_s.pop()
                    elif lca_count>1:
                        lca=lca_s.pop()

                    if lca_count==0:
                        dn.append(max_proximity)
                        #break
                    else:
                        n_dist = self.internode_distance(lca,n)
                        m_dist = self.internode_distance(lca,m)
                        dn.append((n_dist+m_dist)-1)
                else:
                    dn.append(max_proximity)
            matrix.append(dn)

        return matrix

    def simple_arborescence(self):
        "Force a graph into an arborescence - dropping any back edges that may upset the tree structure"
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
            while len(level) != 0:
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
            while len(level) != 0:
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
