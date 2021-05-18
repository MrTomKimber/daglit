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
