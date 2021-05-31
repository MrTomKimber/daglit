import daglit
import pytest


def test_working():
    assert True

def test_template():
    #setup test

    #exercise test

    #verify test
    assert True
    #cleanup

@pytest.fixture
def simple_graph():
    # Test creation of digraph objects - return the finished object
    dg = daglit.DiGraph()
    dg.add_node("A")
    dg.add_node("B")
    dg.add_node("C")
    dg.add_node("D")
    dg.add_node("E")
    assert set(dg.nodes.keys())=={"A", "B", "C", "D", "E"}
    dg.add_edge(("A", "B"))
    dg.add_edge(("A", "C"))
    dg.add_edge(("B", "D"))
    dg.add_edge(("C", "D"))
    assert set(dg.edges.keys())=={("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    return dg

@pytest.fixture
def cyclic_graph():
    dg = daglit.DiGraph.from_dict ({  "a" : ["b"],
                    "b" : ["c"],
                    "c" : ["d"],
                    "d" : ["e"],
                    "e" : ["f"],
                    "f" : ["g"],
                    "g" : ["a"]  }, acyclic=False)
    return dg

@pytest.fixture
def component_graph():
    dg = daglit.DiGraph.from_dict ({  "a" : ["b"],
                    "b" : ["c"],
                    "c" : ["d", "f"],
                    "d" : ["e"],
                    "e" : ["b"],
                    "f" : ["g"],
                    "g" : ["h"],
                    "h" : ["i"],
                    "i" : ["g"]  }, acyclic=False)
    return dg

@pytest.fixture
def component_graph_v2():
    dg = daglit.DiGraph.from_dict ({  "a" : ["b"],
                    "b" : ["c"],
                    "c" : ["d", "g"],
                    "d" : ["e"],
                    "e" : ["b"],
                    "g" : ["h"],
                    "h" : ["i"],
                    "i" : ["g"]  }, acyclic=False)
    return dg

def test_ancestry_digraph_methods(simple_graph):
    dg = simple_graph
    assert dg.ancestors('A')==set()
    assert dg.descendents('A')=={"B", "C", "D"}
    assert dg.ancestors('B')=={"A"}
    assert dg.descendents('B')=={"D"}

def test_from_dict_creation(simple_graph):
    dd = { "A" : ["B", "C"],
           "B" : ["D"],
           "C" : ["D"],
           "D" : [],
           "E" : []}
    dg = daglit.DiGraph.from_dict(dd)
    df = simple_graph
    assert df == dg
    return dg

def test_node_creation():
    dn = daglit.Node(name="test_node")
    assert isinstance(dn, daglit.Node)
    assert dn.name=="test_node"

def test_edge_creation():
    de = daglit.Edge(node_from="A",node_to="B")
    assert isinstance(de, daglit.Edge)
    assert de.edge==("A", "B")


def test_to_dict_creation(simple_graph):
    dd = { "A" : ["B", "C"],
           "B" : ["D"],
           "C" : ["D"],
           "D" : [],
           "E" : []}
    df = simple_graph
    df_d = df.to_dict()
    assert df_d == dd

def test_node_deletion(simple_graph):
    df = simple_graph
    df.delete_node("B")
    assert df.to_dict()=={ "A" : ["C"],
           "C" : ["D"],
           "D" : [],
           "E" : []}

    assert set(df.edges) == set([("A","C"), ("C","D")])

def test_adjacency_matrix(simple_graph):
    df=simple_graph
    mat, keys = df.adjacency_matrix(["A", "B", "C", "D", "E"])
    mat_, keys_ = df.adjacency_matrix()
    assert keys == ["A", "B", "C", "D", "E"]
    assert mat == [[0,1,1,0,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0]]
    assert mat == mat_
    mat_r, keys_r = df.adjacency_matrix(["E", "D", "C", "B", "A"])
    assert mat_r == [[0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,1,0]
                   ]

def test_transition_matrix(simple_graph):
    tm = simple_graph.transition_matrix()
    A=daglit.utils.Array(tm)
    B=daglit.utils.Array(tm)
    for i in range(len(tm)):
        assert tm[i][i]==0

    assert (A@B).T()
    tm2 = simple_graph.transition_matrix(randomise_flow=True)
    for i in range(len(tm2)):
        assert tm2[i][i]>0

    tm3 = simple_graph.transition_matrix(randomise_flow=True, transpose=False)
    for i in range(len(tm3)):
        assert tm3[i][i]>0
    assert tm2 != tm3

def test_root_and_leaf_nodes_degrees(simple_graph):
    assert simple_graph.root_nodes() == { "A", "E"}
    assert simple_graph.leaf_nodes() == { "D", "E"}
    assert simple_graph.nodes_degree() == { 0 : ["E"], 2 : ["A", "B", "C", "D"]}
    assert simple_graph.nodes_in_degree() == { 0 : ["A", "E"], 1 : ["B", "C"], 2 : ["D"]}
    assert simple_graph.nodes_out_degree() == { 0 : ["D", "E"], 1 : ["B", "C"], 2 : ["A"]}

    assert simple_graph.nodes_degree(degree=2) == ["A", "B", "C", "D"]
    assert simple_graph.nodes_in_degree(degree=2) == ["D"]
    assert simple_graph.nodes_out_degree(degree=2) == ["A"]


def test_node_flow_class(simple_graph):
    assert simple_graph.nodes["A"].flow_class()=="Source"
    assert simple_graph.nodes["B"].flow_class()=="Flow"
    assert simple_graph.nodes["C"].flow_class()=="Flow"
    assert simple_graph.nodes["D"].flow_class()=="Sink"
    assert simple_graph.nodes["E"].flow_class()=="Singleton"

def test_node_edges(simple_graph):
    assert simple_graph.nodes["A"].edges()==[('A', 'B'), ('A', 'C')]
    assert simple_graph.nodes["B"].edges()==[('A', 'B'), ('B', 'D')]
    assert simple_graph.nodes["C"].edges()==[('A', 'C'), ('C', 'D')]
    assert simple_graph.nodes["D"].edges()==[('B', 'D'), ('C', 'D')]
    assert simple_graph.nodes["E"].edges()==[]

def test_digraphs_not_equal(simple_graph, cyclic_graph):
    assert not simple_graph == cyclic_graph
    assert simple_graph.has_cycle() == False

def test_ring_is_cyclic(cyclic_graph):
    assert cyclic_graph.has_cycle()
    for k,n in cyclic_graph.nodes.items():
        assert n.flow_class()=="Flow"


def test_acyclic_add_edge_raises_error(simple_graph):
    try:
        simple_graph.add_edge(("D", "A"))
        assert False
    except daglit.EdgeCreatesCycleError:
        assert True

def test_digraph_coparents(simple_graph):
    assert simple_graph.coparents("A") == {"A"}
    assert simple_graph.coparents("B") == {"B", "C"}
    assert simple_graph.coparents("C") == {"B", "C"}
    assert simple_graph.coparents("D") == set()

    assert simple_graph.coparents(simple_graph.nodes["A"]) == {"A"}
    assert simple_graph.coparents(simple_graph.nodes["B"]) == {"B", "C"}
    assert simple_graph.coparents(simple_graph.nodes["C"]) == {"B", "C"}
    assert simple_graph.coparents(simple_graph.nodes["D"]) == set()




def test_digraph_siblings(simple_graph):
    assert simple_graph.siblings("A") == set()
    assert simple_graph.siblings("B") == {"B", "C"}
    assert simple_graph.siblings("C") == {"B", "C"}
    assert simple_graph.siblings("D") == {"D"}

    assert simple_graph.siblings(simple_graph.nodes["A"]) == set()
    assert simple_graph.siblings(simple_graph.nodes["B"]) == {"B", "C"}
    assert simple_graph.siblings(simple_graph.nodes["C"]) == {"B", "C"}
    assert simple_graph.siblings(simple_graph.nodes["D"]) == {"D"}



def test_connected_nodes(simple_graph):
    # The returned nodes from connected_nodes don't include the original node. For regions, this node needs to be added.
    assert simple_graph.connected_nodes("A") == { "B", "C", "D" }
    assert simple_graph.connected_nodes("E") == set()

    assert simple_graph.connected_nodes(simple_graph.nodes["A"]) == { "B", "C", "D" }
    assert simple_graph.connected_nodes(simple_graph.nodes["E"]) == set()



def test_regions(simple_graph):
    # The returned nodes from connected_nodes don't include the original node. For regions, this node needs to be added.
    assert len(simple_graph.regions()) == 2
    for r in simple_graph.regions():
        assert r in [{"A", "B", "C", "D"},  {"E"}]

    a_reg = simple_graph.regions("A")
    assert a_reg == {"A", "B", "C", "D"}

    q_reg = simple_graph.regions("Q")
    assert q_reg is None

def test_subgraph(simple_graph):
    for reg in simple_graph.regions():
        s_graph = simple_graph.subgraph(reg)
        assert set(s_graph.nodes.keys()) == reg

def test_consolidate_0(simple_graph):
    print("Simple Graph Cycle Members")
    print (list(simple_graph.cycle_members()))
    con_graph = simple_graph.consolidate_cycles()
    assert not( simple_graph.has_cycle() )
    assert not( con_graph.has_cycle() )
    assert len(con_graph.nodes)==5
    assert con_graph.root_nodes()=={"A", "E"}
    assert con_graph.leaf_nodes()=={"D", "E"}

def test_consolidate_1(cyclic_graph):
    print("Cyclic Graph Cycle Members")
    print (list(cyclic_graph.cycle_members()))
    con_graph = cyclic_graph.consolidate_cycles()
    assert not( con_graph.has_cycle() )
    assert len(con_graph.nodes)==1
    assert con_graph.root_nodes()=={"__cycle_0"}
    assert con_graph.leaf_nodes()=={"__cycle_0"}

def test_consolidate_2(component_graph):
    print("Component Graph Cycle Members")
    print (list(component_graph.cycle_members()))
    assert component_graph.has_cycle()
    con_graph = component_graph.consolidate_cycles()
    assert not( con_graph.has_cycle() )
    assert len(con_graph.nodes)==4
    assert con_graph.root_nodes()=={"a"}


def test_consolidate_3(component_graph_v2):
    print("Component Graph V2 Cycle Members")
    print (list(component_graph_v2.cycle_members()))
    assert component_graph_v2.has_cycle()
    con_graph = component_graph_v2.consolidate_cycles()
    assert not( con_graph.has_cycle() )
    assert len(con_graph.nodes)==3
    assert con_graph.root_nodes()=={"a"}



def test_graph_analytics_1(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert simple_graph.is_bipartite()
    assert not ( cyclic_graph.is_bipartite() )
    assert not( component_graph.is_bipartite() )
    assert not( component_graph_v2.is_bipartite() )

def test_graph_copy(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    sgc=simple_graph.copy()
    assert sgc==simple_graph
    sgc=cyclic_graph.copy()
    assert sgc==cyclic_graph
    sgc=component_graph.copy()
    assert sgc==component_graph
    sgc=component_graph_v2.copy()
    assert sgc==component_graph_v2

    sgc=simple_graph.copy(reverse_nodes=False)
    assert sgc==simple_graph
    sgc=cyclic_graph.copy(reverse_nodes=False)
    assert sgc==cyclic_graph
    sgc=component_graph.copy(reverse_nodes=False)
    assert sgc==component_graph
    sgc=component_graph_v2.copy(reverse_nodes=False)
    assert sgc==component_graph_v2

def test_graph_copy_reversed(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    sgc=simple_graph.reversed()
    assert sgc.root_nodes()==simple_graph.leaf_nodes()
    sgc=cyclic_graph.reversed()
    assert sgc.root_nodes()==cyclic_graph.leaf_nodes()
    sgc=component_graph.reversed()
    assert sgc.root_nodes()==component_graph.leaf_nodes()
    sgc=component_graph_v2.reversed()
    assert sgc.root_nodes()==component_graph_v2.leaf_nodes()

def test_topological_sorting(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    ts1=list(simple_graph.topological_sort())
    ts2=list(simple_graph.topological_sort(lexicographical=True))
    for node in simple_graph.nodes.keys():
        for descendent in simple_graph.descendents(node):
            assert ts1.index(node)<ts1.index(descendent)
            assert ts2.index(node)<ts2.index(descendent)
    try:
        ts1=list(cyclic_graph.topological_sort())
        assert False
    except daglit.DAGContainsCycleError:
        assert True
    try:
        ts2=list(cyclic_graph.topological_sort(lexicographical=True))
        assert False
    except daglit.DAGContainsCycleError:
        assert True

    try:
        ts1=list(component_graph.topological_sort())
        assert False
    except daglit.DAGContainsCycleError:
        assert True
    try:
        ts2=list(component_graph.topological_sort(lexicographical=True))
        assert False
    except daglit.DAGContainsCycleError:
        assert True

    try:
        ts1=list(component_graph_v2.topological_sort())
        assert False
    except daglit.DAGContainsCycleError:
        assert True
    try:
        ts2=list(component_graph_v2.topological_sort(lexicographical=True))
        assert False
    except daglit.DAGContainsCycleError:
        assert True

def test_node_depth_map(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert simple_graph.node_depth_map() == { "E" : 0, "A" : 0, "B" : 1, "C" : 1, "D" : 2 }
    try:
        cyclic_graph.node_depth_map()
        assert False
    except daglit.DAGContainsNoRootNode:
        reduced_cyclic=cyclic_graph.consolidate_cycles()
        assert reduced_cyclic.node_depth_map()=={'__cycle_0': 0}
    assert component_graph.node_depth_map()=={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'f': 3, 'g': 4, 'e': 4, 'h': 5, 'i': 6}
    assert component_graph_v2.node_depth_map() == {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'g': 3, 'h': 4, 'e': 4, 'i': 5}

def test_all_simple_paths(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert list(simple_graph.all_simple_paths(include_singletons=True))==[['A', 'B'],
                                                   ['A', 'C'],
                                                   ['A', 'B', 'D'],
                                                   ['A', 'C', 'D'],
                                                   ['B', 'D'],
                                                   ['C', 'D'],
                                                   ['E']]
    assert list(cyclic_graph.all_simple_paths())==[['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e', 'f'], ['a', 'b', 'c', 'd', 'e', 'f', 'g'], ['b', 'c', 'd', 'e', 'f', 'g', 'a'], ['b', 'c'], ['b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['b', 'c', 'd', 'e', 'f'], ['b', 'c', 'd', 'e', 'f', 'g'], ['c', 'd', 'e', 'f', 'g', 'a'], ['c', 'd', 'e', 'f', 'g', 'a', 'b'], ['c', 'd'], ['c', 'd', 'e'], ['c', 'd', 'e', 'f'], ['c', 'd', 'e', 'f', 'g'], ['d', 'e', 'f', 'g', 'a'], ['d', 'e', 'f', 'g', 'a', 'b'], ['d', 'e', 'f', 'g', 'a', 'b', 'c'], ['d', 'e'], ['d', 'e', 'f'], ['d', 'e', 'f', 'g'], ['e', 'f', 'g', 'a'], ['e', 'f', 'g', 'a', 'b'], ['e', 'f', 'g', 'a', 'b', 'c'], ['e', 'f', 'g', 'a', 'b', 'c', 'd'], ['e', 'f'], ['e', 'f', 'g'], ['f', 'g', 'a'], ['f', 'g', 'a', 'b'], ['f', 'g', 'a', 'b', 'c'], ['f', 'g', 'a', 'b', 'c', 'd'], ['f', 'g', 'a', 'b', 'c', 'd', 'e'], ['f', 'g'], ['g', 'a'], ['g', 'a', 'b'], ['g', 'a', 'b', 'c'], ['g', 'a', 'b', 'c', 'd'], ['g', 'a', 'b', 'c', 'd', 'e'], ['g', 'a', 'b', 'c', 'd', 'e', 'f']]

    assert list(component_graph.all_simple_paths())==[['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'f'], ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'f', 'g'], ['a', 'b', 'c', 'f', 'g', 'h'], ['a', 'b', 'c', 'f', 'g', 'h', 'i'], ['b', 'c', 'd', 'e', 'b'], ['b', 'c'], ['b', 'c', 'd'], ['b', 'c', 'f'], ['b', 'c', 'd', 'e'], ['b', 'c', 'f', 'g'], ['b', 'c', 'f', 'g', 'h'], ['b', 'c', 'f', 'g', 'h', 'i'], ['c', 'd', 'e', 'b'], ['c', 'd', 'e', 'b', 'c'], ['c', 'd'], ['c', 'f'], ['c', 'd', 'e'], ['c', 'f', 'g'], ['c', 'f', 'g', 'h'], ['c', 'f', 'g', 'h', 'i'], ['d', 'e', 'b'], ['d', 'e', 'b', 'c'], ['d', 'e', 'b', 'c', 'd'], ['d', 'e', 'b', 'c', 'f'], ['d', 'e'], ['d', 'e', 'b', 'c', 'f', 'g'], ['d', 'e', 'b', 'c', 'f', 'g', 'h'], ['d', 'e', 'b', 'c', 'f', 'g', 'h', 'i'], ['f', 'g'], ['f', 'g', 'h'], ['f', 'g', 'h', 'i'], ['e', 'b'], ['e', 'b', 'c'], ['e', 'b', 'c', 'd'], ['e', 'b', 'c', 'f'], ['e', 'b', 'c', 'd', 'e'], ['e', 'b', 'c', 'f', 'g'], ['e', 'b', 'c', 'f', 'g', 'h'], ['e', 'b', 'c', 'f', 'g', 'h', 'i'], ['g', 'h', 'i', 'g'], ['g', 'h'], ['g', 'h', 'i'], ['h', 'i', 'g'], ['h', 'i', 'g', 'h'], ['h', 'i'], ['i', 'g'], ['i', 'g', 'h'], ['i', 'g', 'h', 'i']]

    assert list(component_graph_v2.all_simple_paths())==[['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'g'], ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'g', 'h'], ['a', 'b', 'c', 'g', 'h', 'i'], ['b', 'c', 'd', 'e', 'b'], ['b', 'c'], ['b', 'c', 'd'], ['b', 'c', 'g'], ['b', 'c', 'd', 'e'], ['b', 'c', 'g', 'h'], ['b', 'c', 'g', 'h', 'i'], ['c', 'd', 'e', 'b'], ['c', 'd', 'e', 'b', 'c'], ['c', 'd'], ['c', 'g'], ['c', 'd', 'e'], ['c', 'g', 'h'], ['c', 'g', 'h', 'i'], ['d', 'e', 'b'], ['d', 'e', 'b', 'c'], ['d', 'e', 'b', 'c', 'd'], ['d', 'e', 'b', 'c', 'g'], ['d', 'e'], ['d', 'e', 'b', 'c', 'g', 'h'], ['d', 'e', 'b', 'c', 'g', 'h', 'i'], ['g', 'h', 'i', 'g'], ['g', 'h'], ['g', 'h', 'i'], ['e', 'b'], ['e', 'b', 'c'], ['e', 'b', 'c', 'd'], ['e', 'b', 'c', 'g'], ['e', 'b', 'c', 'd', 'e'], ['e', 'b', 'c', 'g', 'h'], ['e', 'b', 'c', 'g', 'h', 'i'], ['h', 'i', 'g'], ['h', 'i', 'g', 'h'], ['h', 'i'], ['i', 'g'], ['i', 'g', 'h'], ['i', 'g', 'h', 'i']]


def test_full_paths(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert list(simple_graph.full_paths())==[["A", "B", "D"],["A", "C", "D"]]
    assert list(cyclic_graph.full_paths())==[]
    assert list(component_graph.full_paths())==[]
    assert list(component_graph_v2.full_paths())==[]
    assert list(simple_graph.full_paths(include_singletons=True))==[["A", "B", "D"],["A", "C", "D"],["E"]]


def test_node_flows(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert simple_graph.node_flows()=={"A" : 2, "B" : 1, "C" : 1, "D" : 2, "E" : 1}
    assert cyclic_graph.node_flows()=={}
    assert component_graph.node_flows()=={}
    assert component_graph_v2.node_flows()=={}

def test_edge_flows(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert simple_graph.edge_flows()=={('A', 'B'): 1, ('A', 'C'): 1, ('B', 'D'): 1, ('C', 'D'): 1}
    assert cyclic_graph.edge_flows()=={}
    assert component_graph.edge_flows()=={}
    assert component_graph_v2.edge_flows()=={}

def test_longest_paths(simple_graph, cyclic_graph, component_graph, component_graph_v2):
    assert list(simple_graph.longest_paths())==list(simple_graph.full_paths())
    assert list(cyclic_graph.longest_paths())==[['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                                                ['b', 'c', 'd', 'e', 'f', 'g', 'a'],
                                                ['c', 'd', 'e', 'f', 'g', 'a', 'b'],
                                                ['d', 'e', 'f', 'g', 'a', 'b', 'c'],
                                                ['e', 'f', 'g', 'a', 'b', 'c', 'd'],
                                                ['f', 'g','a', 'b', 'c', 'd', 'e' ],
                                                ['g', 'a', 'b', 'c', 'd', 'e', 'f']]
