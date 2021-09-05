import daglit.layouts.layouts as layouts
from daglit.utils import PermutationCycler
from collections import OrderedDict
from functools import reduce
import random


class LayeredLayout(layouts.LayoutHelper):
    def __init__(self, dag, **kwargs):
        super().__init__(dag, **kwargs)
        if "style" not in kwargs:
            self.style=True
        else:
            self.style=kwargs['style']

        self.working_dag = self.fill_cross_layer_paths()
        layer_d=self.layer_assignment()
        print(layer_d)
        s_groups = self.extract_sibling_groups()
        psg=self.extract_permutable_groups()
        self.permutable_siblings_d = OrderedDict([(k,list(v)) for k,v in psg.items()])
        p_list = [len(v) for k,v in self.permutable_siblings_d.items()]
        self.layout_permutations = PermutationCycler(p_list)

    def analyse_edge_crossings(self, **kwargs):
        if "sample_rate" in kwargs:
            sample_size = int(self.layout_permutations.size * kwargs.get("sample_rate"))
        elif "sample_size" in kwargs:
            sample_size = min([self.layout_permutations.size , kwargs.get("sample_size")])
        else:
            sample_size=self.layout_permutations.size

        if sample_size >= self.layout_permutations.size:
            sample_ps = range(0,self.layout_permutations.size)
        else:
            sample_ps=sorted(random.sample(range(0,self.layout_permutations.size),sample_size))
        if "stop_threshold" in kwargs:
            stop_threshold=kwargs['stop_threshold']
        else:
            stop_threshold=0
        fitness_d=OrderedDict()
        layouts_d=OrderedDict()
        for p in sample_ps:
            perm = self.layout_permutations.permute(self.permutable_siblings_d, p)
            parent_row=["__start__"]
            unprocessed=set(self.working_dag.nodes.keys())-set(["__start__"])
            layer_number = 0
            layout = OrderedDict()
            new_row=[]

            while len(unprocessed)>0:
                if len(parent_row)>0:
                    key = parent_row.pop()
                    for e in perm.get(key,[]):

                        new_row.append(self.permutable_siblings_d[key][e])
                else:
                    parent_row=new_row[::-1]
                    layout[layer_number]=[i for i in new_row]
                    layer_number+=1
                    new_row=[]
                unprocessed=unprocessed-set(new_row)
            layout[layer_number]=[i for i in new_row]


            loc_d=OrderedDict()
            for k,v in layout.items():
                for e,i in enumerate(v):
                    loc_d[i]=(e,k)
            fitness=self.edge_crossing_count(loc_d)
            fitness_d[p]=fitness
            layouts_d[p]=layout
            if fitness <= stop_threshold:
                break
        return fitness_d, layouts_d


    def best_layout(self, **kwargs):

        if "stop_threshold" not in kwargs:
            stop_threshold=0
        else:
            stop_threshold=kwargs["stop_threshold"]

        if "sample_size" not in kwargs:
            sample_size=5000
        else:
            sample_size=sample_size
        fit_d, layout_d = self.analyse_edge_crossings(sample_size=5000, stop_threshold=stop_threshold)
        minv, maxv = min(fit_d.values()), max(fit_d.values())
        best_fits = [(k,v) for k,v in fit_d.items() if v==minv]
        worst_fits = [(k,v) for k,v in fit_d.items() if v==maxv]
        best_index=best_fits.pop()[0]
        best_layout = layout_d[best_index]
        loc_d={}
        for k,v in best_layout.items():
            for e,i in enumerate(v):
                loc_d[i]=(e,k)
        return best_layout

    def layout(self):
        loc_d = self.raw_layout()



    def edge_crossing_count(self, locs):
        matrix=[]
        key=[]
        key_row=[]
        for i,e in enumerate(self.working_dag.edges):
            e_locs = (locs[e[0]], locs[e[1]])
            key_row.append(e)
            row=[]
            for j,t in enumerate(self.working_dag.edges):
                t_locs = (locs[t[0]], locs[t[1]])
                e_nodes=set(e)
                t_nodes=set(t)
                if len(e_nodes.intersection(t_nodes))!=0:
                    row.append(0)
                else:
                    row.append(edge_intersect(e_locs,t_locs) * 1)
                    if edge_intersect(e_locs,t_locs):
                        #print(e, t)
                        pass
            matrix.append(row)
        return sum([sum(r) for r in matrix])


    def get_layer_groups(self):
        layer_d=self.layer_assignment()
        layer_groups={}
        for n,l in layer_d.items():
            if l in layer_groups:
                layer_groups[l].append(n)
            else:
                layer_groups[l]=[n]
        return layer_groups

    def fill_cross_layer_paths(self, style="baseline"):
        temp_dag = self.working_dag
        if style.lower()=="baseline":
            baseline = True
        else:
            baseline = False
        layer_d=self.layer_assignment()
        finished=False
        # Show the cross_layer_counts for each edge
        v_nodes=0
        while finished==False:
            ed_list = [(edge,distance) for edge, distance in self.cross_layer_edge_lengths(layer_d).items() if distance > 1]
            edge, distance = ed_list.pop()
            if distance > 1:
                temp_dag.split_edge(edge, ["__v_{n}".format(n=e+v_nodes) for e,n in enumerate(range(0,1))])
                v_nodes = v_nodes + 1
                layer_d=self.layer_assignment()

            if any([distance>1 for edge, distance in self.cross_layer_edge_lengths(layer_d).items()]):
                finished=False
            else:
                finished=True
        return temp_dag

    def layer_assignment(self):
        working_dag = self.working_dag.copy()
        layer_d = {}
        layer = -1
        if self.style:
            while len(working_dag.nodes.items())>0:
                layer = layer + 1
                this_layer = [k for k,n in working_dag.nodes.items() if n.in_degree() == 0]
                layer_d = { **layer_d, **{n:layer for n in this_layer}}
                for n in this_layer:
                    working_dag.delete_node(n)
        else:
            while len(working_dag.nodes.items())>0:
                layer = layer + 1
                this_layer = [k for k,n in working_dag.nodes.items() if n.out_degree() == 0]
                layer_d = { **layer_d, **{n:layer for n in this_layer}}
                for n in this_layer:
                    working_dag.delete_node(n)
            max_layer=max(layer_d.values())
            layer_d ={k:max_layer-v for k,v in layer_d.items()}
        return layer_d

    def cross_layer_edge_lengths(self,layer_d):
        clel_d = {}
        for e,edge in self.working_dag.edges.items():
            clel_d[e] = layer_d[edge.node_to] - layer_d[edge.node_from]
        return clel_d

    def extract_sibling_groups(self):
        s_groups=set()
        for n in self.working_dag.leaf_nodes():
            self.working_dag.add_edge((n,"__start__"))
        for k,n in self.working_dag.nodes.items():
            s_groups.add(frozenset(self.working_dag.reversed().siblings(k)))
        self.working_dag.delete_node("__start__")
        return s_groups


    def extract_permutable_groups(self):
        layer_groups = self.get_layer_groups()
        top_down_layers = [(k,v) for k,v in layer_groups.items()][::-1]
        s_groups=self.extract_sibling_groups()

        processed_items=set()
        canonical_group_order = []
        sibling_group_dict = OrderedDict()
        for l, items in top_down_layers:
            layer_items = [i for i in items]
            while len(layer_items)>0:
                seed = layer_items.pop()
                group_pick = [group for group in s_groups if seed in group][0]
                canonical_group_order.append(group_pick)
                shared_parents=reduce(set.intersection,[set(self.working_dag.nodes[n].successors.keys()) for n in group_pick])
                if len(shared_parents)>0:
                    shared_parent= shared_parents.pop()
                    if shared_parent in sibling_group_dict.keys():
                        assert False
                    else:
                        sibling_group_dict[shared_parent]=group_pick
                else:
                    if "__start__" in sibling_group_dict.keys():
                        assert False
                    else:
                        sibling_group_dict["__start__"]=group_pick
        #            pass
                for element in group_pick:
                    if element in processed_items:
                        #print (element, "exists!")
                        pass
                    processed_items.add(element)

                layer_items = list(set(layer_items)-processed_items)
        return sibling_group_dict


def ccw(A,B,C):
    Ax,Ay=A
    Bx,By=B
    Cx,Cy=C
    return (Cy-Ay) * (Bx-Ax) > (By-Ay) * (Cx-Ax)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def edge_intersect(E,F):
    A,B = E
    C,D = F
    return intersect(A,B,C,D)



    return sum([sum(r) for r in matrix])
