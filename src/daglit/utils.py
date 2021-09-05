from math import factorial
from functools import reduce
from operator import mul

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

    def permute(self, permutation_dict, p):
        return {k:self.permutations(p)[e] for e,(k,v) in enumerate(permutation_dict.items()) }

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






## Matrix Multiplication
def _matmul(M,N):
    # Perform matrix Multiplication (dot product)
    M_shape, N_shape =(len(M), len(M[0])), (len(N), len(N[0]))
    R_shape = (M_shape[0], N_shape[1])
    assert len(M)==len(N[0])
    R = [[0] * len(N[0]) for n in range(0,len(M))]

    for i in range(0,len(M)):
        for j in range(0,len(N[0])):
            for k in range(0,len(M[0])):
                R[i][j]=R[i][j]+(M[i][k]*N[k][j])
    return R


def _transpose_matrix(matrix):
    # Transpose an nxm matrix into an mxn matrix
    return list(map(lambda *a: list(a), *matrix))

class Array():
    def __init__(self, iterable):
        self.array = list(iterable)
        self.shape = len(self.array), len(self.array[0])

    def __repr__(self):
        repr_str = "\n".join([str(row) for row in self.array])
        return repr_str

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            selected = self.array[index[0]]
            if len(index)>1:
                for c in index[1:]:
                    selected = selected[c]
            return selected

        return self.array[index]

    def unravel(self):
        for row in self.array:
            for column in row:
                yield column

    def T(self):
        return Array(_transpose_matrix(self.array))

    def __matmul__(self,B):
        return Array(_matmul(self.array,B))
