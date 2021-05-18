"""
Tests written per pytest conventions.
https://docs.pytest.org/en/6.2.x/goodpractices.html
"""
import pytest
import daglit
import random

def test_working():
    assert True

def test_template():
    #setup test

    #exercise test

    #verify test
    assert True
    #cleanup

def test_daglit_digraph_fundamentals():
    dg = daglit.DiGraph("test")
    assert dg.name=="test"
    dg.add_node("A")
    dg.add_node("B")
    dg.add_node("C")
    assert len(dg.nodes)==3
    assert len(dg.edges)==0
    dg.add_edge(("A", "B"))
    dg.add_edge(("B", "C"))
    assert len(dg.nodes)==3
    assert len(dg.edges)==2
    dg.delete_node("B")
    assert len(dg.nodes)==2
    assert len(dg.edges)==0
    assert len(dg.nodes["A"].neighbors)==0
    assert len(dg.nodes["C"].neighbors)==0
    dg.add_node("B")
    assert len(dg.nodes)==3
    assert len(dg.edges)==0
    assert len(dg.nodes["A"].neighbors)==0
    assert len(dg.nodes["B"].neighbors)==0
    assert len(dg.nodes["C"].neighbors)==0
    dg.add_edge(("A", "B"))
    assert len(dg.nodes["A"].neighbors)==1
    assert len(dg.nodes["B"].neighbors)==1
    assert len(dg.nodes["C"].neighbors)==0
    dg.add_edge(("B", "C"))
    assert len(dg.nodes["A"].neighbors)==1
    assert len(dg.nodes["B"].neighbors)==2
    assert len(dg.nodes["C"].neighbors)==1
    cols = dg.attempt_bi_colouring()
    assert cols == {"A":1, "B":0, "C":1}
    try:
        dg.add_edge(("C", "A"))
        assert False
    except daglit.EdgeCreatesCycleError:
        assert True
    assert dg.is_bipartite()
    assert not(dg.has_cycle())
    assert len(dg.node_degree(0))==0
    assert len(dg.node_degree(1))==2
    assert len(dg.node_degree(2))==1
    assert dg.ancestors("A")==[]
    assert set(dg.descendents("A",names=True))==set(["B","C"])
    assert set(dg.descendents("A",names=False))==set([dg.nodes["B"],dg.nodes["C"]])
    assert dg.descendents("C")==[]
    assert set(dg.ancestors("C",names=True))==set(["A","B"])
    assert set(dg.ancestors("C",names=False))==set([dg.nodes["A"],dg.nodes["B"]])
    ts_iter = dg.topological_sort()
    assert next(ts_iter)=="A"
    assert next(ts_iter)=="B"
    assert next(ts_iter)=="C"
    try:
        next(ts_iter)
        assert False
    except StopIteration:
        assert True
    rg=dg.reversed()
    ts_iter = rg.topological_sort()
    assert next(ts_iter)=="C"
    assert next(ts_iter)=="B"
    assert next(ts_iter)=="A"
    try:
        next(ts_iter)
        assert False
    except StopIteration:
        assert True

    assert dg.root_nodes() == ["A"]
    assert rg.root_nodes() == ["C"]
    assert dg.leaf_nodes() == ["C"]
    assert rg.leaf_nodes() == ["A"]

    ndm = dg.node_depth_map()
    assert ndm == { "A" : 0, "B" : 1, "C" : 2}

    ndm = rg.node_depth_map()
    assert ndm == { "C" : 0, "B" : 1, "A" : 2}

    del ndm

    paths = list(dg.full_paths())
    assert paths == [["A", "B", "C"]]

    paths = [tuple(p) for p in dg.paths()]
    assert set(paths) == {("A", "B"), ("B", "C"), ("A", "B", "C")}

@pytest.mark.repeat(1)
def test_daglit_digraph_random():
    alphabet="ABC"
    dag = daglit.DiGraph("test")
    nodes=set()
    for e in range(0,random.randint(20,100)):
        nodes.add("".join(random.choices(alphabet,k=5)))
    nodes=list(nodes)
    for e in range(0,random.randint(0,100)):
        try:
            edge = (random.choice(nodes), random.choice(nodes))
            dag.add_edge(edge)
        except daglit.EdgeCreatesCycleError:
            pass
        except Exception as e:
            print(e)
            raise e
            assert False

    try:
        bptest = dag.is_bipartite()
        no_cycle = dag.has_cycle()

        assert bptest in (True, False)
        assert not( no_cycle )
        #print (list([len(p) for p in dag.full_paths()]))
        #print (list([p for p in dag.longest_paths()]))
    except Exception as e:

        assert False
