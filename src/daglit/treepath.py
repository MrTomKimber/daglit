import copy
from functools import reduce
import operator
import json
import pandas as pd
import hashlib
import networkx as nx

class TreePath(object):
    def __init__(self, obj: object):
        self.data = copy.deepcopy(obj)

    @staticmethod
    def _hashpath(path, length=64):
        return hashlib.sha256(bytes(str(tuple(path)).encode())).hexdigest()[0:length]

    def _calc_node_stats(self, length=64):
        paths_by_length = sorted(self.all_paths(), key = lambda x : len(x))
        centrality = {}
        hashmap = {}
        inv_hashmap = {}
        node_types = {}
        leaf_paths = set([tuple(p) for p in self.leaf_paths()])
        for p in paths_by_length:
            tuple_p = tuple(p)
            hashmap[tuple_p]=self._hashpath(p)
            inv_hashmap[self._hashpath(p)]=tuple_p
            if tuple_p in leaf_paths:
                node_types[tuple_p]="leaf"
            elif len(p)==1:
                node_types[tuple_p]="root"
            else:
                node_types[tuple_p]="node"
        self.nodemap=hashmap
        self.inv_nodemap=inv_hashmap
        self.node_type_map=node_types

    def to_dag(self):
        dag = nx.DiGraph()
        edge_set = set()
        if not hasattr(self, "nodemap"):
            self._calc_node_stats()
        for k,v in self.nodemap.items():
            dag.add_node(k, data={"path" : list[v], "label" : v[-1], "type" : self.node_type_map[k]})
            if self.node_type_map[k] != "root":
                for e,p in enumerate(v[1:]):
                    edge = ( self.nodemap.get(tuple(v[0:e+1]))), self.nodemap(.get(tuple(v[0:e+2]))) )
                    if edge not in edge_set:
                        dag.add_edge(*edge)
                        edge_set.add(edge)
        return dag

    def get(self, path : list):
        return reduce(operator.getitem, path, self.data)

    def set(self, path: list, value):
        self.get(path[:-1])[path[-1]] = value

    def matching_paths(self, path: list):
        all_paths = self.all_paths()
        matching_paths = []
        if len(path)>max([len(p) for p in all_paths]):
            return []
        else:
            for p in all_paths:
                if len(p)>=len(path):
                    blocks = [(i, i+len(path)) for i in self._path_indices(p,path[0])]
                    for s,f in blocks:
                        if f<=len(p):
                            if p[s:f]==path:
                                matching_paths.append(p)
        return matching_paths

    def path_exists(self, path):
        try:
            a=self.get(path)
            return True
        except (KeyError, IndexError):
            return False

    def all_paths(self, path=None, paths=None):
        return self._all_paths(self.data, path, paths)

    def keys(self, path=None):
        if path is None:
            path = []
        keys = list(self._iterKeys(self.get(path)))
        if len(keys)>0 and keys != [None]:
            return keys
        else:
            return None

    def leaf_paths(self):
        paths = self._all_paths(self.data)
        l_paths = []
        for p in paths:
            if not isinstance(self.get(p), (dict, list)):
                l_paths.append(p)
        return l_paths

    def paths_matching_content(self, candidates: list):
        if not isinstance(candidates, list):
            candidates = [candidates]
        paths = self._all_paths(self.data)
        c_paths = []
        for p in paths:
            if p[-1] in candidates or self.get(p) in candidates:
                c_paths.append(p)
        return c_paths

    def member_class(self, path):
        if not isinstance(path, list):
            path = [path]
        return type(self.get(path[0:-1]))

    @staticmethod
    def _iterKeys(variable):
        if isinstance(variable, (str, int, float, bool)):
            yield None
        elif isinstance(variable, dict):
            for k in variable.keys():
                yield k
        elif isinstance(variable, (list, tuple)):
            for k in range(0,len(variable)):
                yield k
        else:
            raise StopIteration

    @classmethod
    def _all_paths(cls, data, path=None, paths=None):
        if path is None:
            path=[]
        if paths is None:
            paths=[]
        keys = list(cls._iterKeys(data))
        for k in keys:
            try:
                paths.append(path+[k])
            except ValueError:
                print (k)
            if isinstance(data[k], (dict, list)):
                paths.extend(cls._all_paths(data[k], path + [k], []))
            else:
                pass
        return paths


    @staticmethod
    def _shortest_common_path(path_list):
        best=0
        shortest_length = min([len(p) for p in path_list])
        for loc in range(0, shortest_length):
            if all([p[loc]==path_list[0][loc] for p in path_list]):
                best = loc
            else:
                break
        return path_list[0][0:best+1]

    @staticmethod
    def walk(node, path=None):
        if path is None:
            path=[]
        empty_dict = {}
        for key, item in node.items():
            if isinstance(item,dict):
                if item != empty_dict:
                    for w in walk(item, path + [key]):
                        yield w
                else:
                    yield path, key
            else:
                yield item

    @staticmethod
    def _path_indices(path, element):
        indices=[]
        last_i=0
        finished=False
        while not finished:
            try:
                i = path.index(element, last_i)
                last_i = i+1
                indices.append(i)
            except ValueError:
                finished=True
        return indices
