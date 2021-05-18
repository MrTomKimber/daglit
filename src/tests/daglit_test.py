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
    assert (A@B).T()
