import daglit.digraph
from daglit.layouts  import transforms
from random import random
from math import sqrt

class NodePositioningError(Exception):
    pass

class LayoutHelper(object):
    def __init__(self, dag, **kwargs):
        self.original_dag = dag
        self.working_dag = dag.copy()
        self.kwargs=kwargs

    def raw_layout_to_locations(self, layout):
        loc_d={}
        for k,v in layout.items():
            for e,i in enumerate(v):
                loc_d[i]=(e,k)
        return loc_d


    def smoothed_layout_to_locations(self, layout, max_iterations=2000, spread=0.5):
        force_lattice = self.working_dag.copy()


        force_lattice = daglit.digraph.DiGraph("forces",True)#Allow Cycles in the force lattice
        for k,n in self.working_dag.nodes.items():
            force_lattice.add_node(k, data=n.data)
        for k,e in self.working_dag.edges.items():
            force_lattice.add_edge(k,data=e.data)
        for layer, content in layout.items():
            for e in range(0,len(content)-1):
                force_lattice.add_edge((content[e], content[e+1]), data={"edge_type" : "lattice_link"})

        #for e,ed in force_lattice.edges:
        #    print(e,ed)
        node_forces = { n : 0 for n in force_lattice.nodes}
        iteration=0

        loc_d = self.raw_layout_to_locations(layout)



        while (sum([abs(c) for c in node_forces.values()])>1 or iteration==0) and iteration<max_iterations:
            iteration+=1
            node_forces = { n : 0 for n in force_lattice.nodes}
            for k,e in force_lattice.edges.items():
                edge_dist = loc_d[e.node_to][0] - loc_d[e.node_from][0]
                if e.data.get("edge_type", "graph_edge")=="graph_edge":
                    node_forces[e.node_from] = (node_forces[e.node_from] + (edge_dist*(1-spread)))
                    node_forces[e.node_to] = (node_forces[e.node_to] - (edge_dist*spread))
#                    for neighbour in [n for ed in force_lattice.nodes[e.node_from].edges() for n in ed]:
#                        node_forces[neighbour] = node_forces[neighbour] + (1*edge_dist)/5
#                    for neighbour in [n for ed in force_lattice.nodes[e.node_to].edges() for n in ed]:
#                        node_forces[neighbour] = node_forces[neighbour] - (1*edge_dist)/5

            #nf_copy = node_forces.copy()
            for k,e in force_lattice.edges.items():
                edge_dist = loc_d[e.node_to][0] - loc_d[e.node_from][0]
                if e.data.get("edge_type", "graph_edge")=="lattice_link":
                    n_degree = max([force_lattice.nodes[e.node_from].degree(), force_lattice.nodes[e.node_to].degree()])*1

                    node_forces[e.node_from] = (node_forces[e.node_from] - (n_degree*(1/((edge_dist)**1))))
                    node_forces[e.node_to] = (node_forces[e.node_to] + (n_degree*(1/((edge_dist)**1))))

            #print(node_forces)
            for k,v in loc_d.items():
                f_sign = sign(node_forces[k])
                #print(f_sign/10)
                loc_d[k]=(loc_d[k][0] + f_sign/10, loc_d[k][1])
        print(sum([abs(c) for c in node_forces.values()]),iteration)
        return loc_d

    @staticmethod
    def snap_to_lattice(locs_xy_d, width=None, height=None):
        ok=False
        while ok is not True:
            try:
                if width is None:
                    width=len(locs_xy_d)

                if height is None:
                    height=len(locs_xy_d)

                x_vals, y_vals = [xy for xy in zip(*[xy for xy in locs_xy_d.values()])]
                minx,miny=min(x_vals), min(y_vals)
                x_vals = [x-minx for x in x_vals]
                y_vals = [y-miny for y in y_vals]

                maxx,maxy= max(x_vals), max(y_vals)
                xstep,ystep = maxx/(width-1), maxy/(height-1)
                lattice_x = [(xstep * e) - (xstep/2) for e in range(0,width+1)]
                lattice_y = [(ystep * e) - (ystep/2) for e in range(0,height+1)]
                snap_xy_d = {}
                for e,(k,p) in enumerate(locs_xy_d.items()):

                    snap_xy_d[k]=([(x_vals[e])>l for l in lattice_x].index(False)-1,
                                  [(y_vals[e])>l for l in lattice_y].index(False)-1 )

                inverse={}
                for k,p in snap_xy_d.items():
                    if p not in inverse:
                        inverse[p]=[k]
                    else:
                        inverse[p].append(k)
                if any([len(v)>1 for v in inverse.values()]):
                    raise NodePositioningError("More than one nodes occupy the same position. Increase the width of your snap-to-grid.")
                ok=True
            except NodePositioningError:
                print("width+1")
                width=width+1
        return snap_xy_d





# Given a series of permutable objects, generate a collection of individual permutation
# keys for each object, determined by a single identifier from zero to the max count of perms
# This is essentially a ticker-counter, consisting a number of wheels that 'tick' their
# neighbour wheels by one element after a full revolution.
class PermutationCycler(object):
    def __init__(self, t_list):
        self.terms = [1]+t_list
        self.factors=[1]+[factorial(t) for t in t_list]
        c=1
        c_factors=[c]
        for f in self.factors:
            c=c*f
            c_factors.append(c)
        self.wheels = c_factors
        self.size = reduce(mul, [c for c in self.factors])

    # Return a permutation for all the tickers associated with an index.
    def code(self, numeric):
        if numeric >= self.size :
            raise IndexError("Permutation larger than collection allows.")
        r=[]
        for e,w in enumerate(self.wheels):
            if e>0:
                leftover = numeric%w
                r.append(int(leftover/self.wheels[e-1]))
                numeric = numeric - leftover

        return r[1:]

    def permutations(self, code_number):
        code = self.code(code_number)
        perms=[]
        for e,c in enumerate(code):
            perms.append(PermutationCycler.generatePermutationByIndex(self.terms[e+1],c))
        return perms

    @staticmethod
    def permutationCount(n):
        return factorial(n)

    @staticmethod
    def generatePermutationByIndex(n, i):
        p = list(range(0,n))
        r = []
        for k in list ( range ( 1, n+1 )):
            f = factorial(n-k)
            d = (i // f)
            r.append (p[d])
            p.remove(p[d])
            i = i % f
        return r

    @staticmethod
    def identifyPermutation(p):
        s = list(range(0,len(p)))
        i = 0
        c = 0
        for k in p:
            f = factorial(len(s)-1)
            d = s.index(k)
            c = c + ( d * f )
            s.remove(k)
        return c

def sign(number):
    if number >= 0:
        return 1
    else:
        return -1




def layered_dag_layout(dag, style="baseline"):
    pass
