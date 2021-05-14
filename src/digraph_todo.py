

    # Don't trust this one - it relies on shaky assumptions
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

    def node_flows(self):
        """How often does a node feature in an enumeration of all root to leaf node traversals?
           This gives an indication of the node's centrality.
           Alternately, how many such traversals exist that pass through each node?"""
        return dict(Counter(chain(*self.full_paths())))


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


    def node_depth_map(self,minimum=False):
        if minimum:
            test_val=float('inf')
        else:
            test_val=0
        level_d = {}
        dfl_nodelist = list(self.depth_first_sort())
        for n in dfl_nodelist:
            if minimum:
                for l in self.root_nodes():
                    if l == n:
                        level_d[n]=0
                    elif l in self.ancestors(n,names=True):
                        if level_d.get(n,test_val)>len(self.shortest_path_between(l,n))-1:
                            level_d[n]=len(self.shortest_path_between(l,n))-1
                        else:
                            print ("1", n,l)
                    else:
                        print ("2", n,l,self.ancestors(l,names=True))
            else:

                for l in self.root_nodes():
                    if l == n:
                        level_d[n]=0
                    elif l in self.ancestors(n,names=True):
                        if level_d.get(n,test_val)<len(self.longest_path_between(l,n))-1:
                            level_d[n]=len(self.longest_path_between(l,n))-1
                        else:
                            print ("3", n,l)
                    else:
                        if not self.acyclic:
                            print( l, n, "4")
                            node_set = set(self.nodes.keys())-set(level_d.keys())

        if len(level_d)!=len(self.nodes):
            print ("nodes missing")
            print ("original_nodelist ", dfl_nodelist)
            missing_set = set(self.nodes.keys())-set(level_d.keys())
            print ("missing", missing_set)
            while len(missing_set)>0:
                n = missing_set.pop()
                if minimum:
                    for l in missing_set:
                        if l == n:
                            level_d[n]=0
                        elif l in self.ancestors(n,names=True):
                            if level_d.get(n,test_val)>len(self.shortest_path_between(l,n))-1:
                                level_d[n]=len(self.shortest_path_between(l,n))-1
                            else:
                                print ("1", n,l)
                        else:
                            print ("2", n,l,self.ancestors(l,names=True))
                else:

                    for l in missing_set:
                        if l == n:
                            level_d[n]=0
                        elif l in self.ancestors(n,names=True):
                            if level_d.get(n,test_val)<len(self.longest_path_between(l,n))-1:
                                level_d[n]=len(self.longest_path_between(l,n))-1
                            else:
                                print ("3", n,l)
                        else:
                            if not self.acyclic:
                                print( l, n, "4")
                                node_set = set(self.nodes.keys())-set(level_d.keys())

                #missing_set = set(self.nodes.keys())-set(level_d.keys())
                #print (missing_set, self.ancestors(list(missing_set)[0]), self.descendents(list(missing_set)[0]))
        print ( level_d )
        return level_d
