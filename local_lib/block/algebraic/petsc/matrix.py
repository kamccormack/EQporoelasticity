from __future__ import division

"""Module implementing algebraic operations on PETSc matrices: Diag, InvDiag
etc, as well as the collapse() method which performs matrix addition and
multiplication.
"""

from block.block_base import block_base
from petsc4py import PETSc

class diag_op(block_base):
    """Base class for diagonal Epetra operators (represented by an Epetra vector)."""
    from block.object_pool import shared_vec_pool

    def __init__(self, v):
        import dolfin
        if isinstance(v, dolfin.GenericVector):
            v = v.down_cast().vec()
        assert isinstance(v, PETSc.Vec)
        self.v = v

    def _transpose(self):
        return self

    def matvec(self, b):
        try:
            b_vec = b.down_cast().vec()
        except AttributeError:
            return NotImplemented

        x = self.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for %s matvec, %d != %d'%(self.__class__.__name__,len(x),len(b)))

        x.down_cast().vec().pointwiseMult(self.v, b_vec)
        return x

    transpmult = matvec

    def matmat(self, other):
        try:
            from block.block_util import isscalar
            if isscalar(other):
                x = self.v.copy()
                x.scale(other)
                return diag_op(x)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                C = other.mat().copy()
                C.diagonalScale(L=self.v)
                return matrix_op(C)
            else:
                x = self.v.copy()
                x.pointwiseMult(self.v, other.vec())
                return diag_op(x)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    def add(self, other, lscale=1.0, rscale=1.0):
        from block.block_util import isscalar
        from PyTrilinos import Epetra
        try:
            if isscalar(other):
                x = self.v.copy()
                x.shift(rscale*other)
                return diag_op(x)
            other = other.down_cast()
            if isinstance(other, matrix_op):
                return other.add(self, lscale=rscale, rscale=lscale)
            else:
                x = self.v.copy()
                x.axpby(lscale, rscale, other.vec())
                return diag_op(x)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    @shared_vec_pool
    def create_vec(self, dim=1):
        from dolfin import PETScVector
        if dim > 1:
            raise ValueError('dim must be <= 1')
        return PETScVector(self.v.copy())

    def down_cast(self):
        return self
    def vec(self):
        return self.v

    def __str__(self):
        return '<%s %dx%d>'%(self.__class__.__name__,len(self.v),len(self.v))

class matrix_op(block_base):
    """Base class for Epetra operators (represented by an Epetra matrix)."""
    from block.object_pool import vec_pool

    def __init__(self, M, _transposed=False):
        assert isinstance(M, PETSc.Mat)
        self.M = M
        self.transposed = _transposed

    def _transpose(self):
        # Note: An object with transposed set should only be used internally
        # (in matmat() / matvec() / ...), and not returned to the user.
        return matrix_op(self.M, not self.transposed)

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        if self.transposed:
            domainlen = self.M.size[0]
        else:
            domainlen = self.M.size[1]
        x = self.create_vec(dim=0)
        if len(b) != domainlen:
            raise RuntimeError(
                'incompatible dimensions for %s matvec, %d != %d'%(self.__class__.__name__,domainlen,len(b)))
        if self.transposed:
            self.M.multTranspose(b.down_cast().vec(), x.down_cast().vec())
        else:
            self.M.mult(b.down_cast().vec(), x.down_cast().vec())
        return x

    def transpmult(self, b):
        return self._transpose().matvec(b)

    def matmat(self, other):
        from block.block_util import isscalar
        try:
            if isscalar(other):
                C = self.M.copy()
                C.scale(other)
                return matrix_op(C, self.transposed)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                if self.transposed and other.transposed:
                    return other.transpose().matmat(self.transpose()).transpose()

                if self.transposed:
                    C = self.M.transposeMatMult(other.mat())
                elif other.transposed:
                    C = self.M.matTransposeMult(other.mat())
                else:
                    C = self.M.matMult(other.mat())

                return matrix_op(C)
            else:
                C = self.M.copy()
                if self.transposed:
                    C.diagonalScale(L=other.vec())
                else:
                    C.diagonalScale(R=other.vec())
                return matrix_op(C, self.transposed)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    def add(self, other, lscale=1.0, rscale=1.0):
        try:
            other = other.down_cast()
        except AttributeError:
            #raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))
            pass

        if hasattr(other, 'mat'):
            if self.transposed and other.transposed:
                return add(self, other, lscale=lscale, rscale=rscale).transpose()

            A = self.M
            if self.transposed:
                A = self.M.transpose()
            if lscale != 1.0:
                A = A.copy()
                A.scale(lscale)
            B = other.mat()
            if other.transposed:
                B = B.transpose()
            if rscale != 1.0:
                B = B.copy()
                B.scale(rscale)

            return matrix_op(A+B)
        else:
            lhs = self.matmat(lscale)
            D = Diag(lhs).add(other, rscale=rscale)
            lhs.M.setDiagonal(D.vec())
            return lhs

    @vec_pool
    def create_vec(self, dim=1):
        from dolfin import PETScVector
        if self.transposed:
            dim = 1-dim
        if dim == 0:
            m = self.M.createVecRight()
        elif dim == 1:
            m = self.M.createVecLeft()
        else:
            raise ValueError('dim must be <= 1')
        return PETScVector(m)

    def down_cast(self):
        return self
    def mat(self):
        return self.M

    def __str__(self):
        format = '<%s transpose(%dx%d)>' if self.transposed else '<%s %dx%d>'
        return format%(self.__class__.__name__, self.M.size[0], self.M.size[1])

class Diag(diag_op):
    """Extract the diagonal entries of a matrix"""
    def __init__(self, A):
        diag_op.__init__(self, A.down_cast().mat().getDiagonal())

class InvDiag(Diag):
    """Extract the inverse of the diagonal entries of a matrix"""
    def __init__(self, A):
        Diag.__init__(self, A)
        self.v.reciprocal()

class LumpedInvDiag(diag_op):
    """Extract the inverse of the lumped diagonal of a matrix (i.e., sum of the
    absolute values in the row)."""
    def __init__(self, A):
        v = A.create_vec().down_cast().vec()
        a = A.down_cast().mat()
        for row in xrange(*a.getOwnershipRange()):
            data = a.getRow(row)[1]
            v.setValue(row, sum(abs(d) for d in data))
        v.assemblyBegin()
        v.assemblyEnd()
        v.reciprocal()
        diag_op.__init__(self, v)

def _collapse(x):
    # Works by calling the matmat(), transpose() and add() methods of
    # diag_op/matrix_op, depending on the input types. The input is a tree
    # structure of block.block_mul objects, which is collapsed recursively.

    # This method knows too much about the internal variables of the
    # block_mul objects... should convert to accessor functions.

    from block.block_compose import block_mul, block_add, block_sub, block_transpose
    from block.block_mat import block_mat
    from block.block_util import isscalar
    from dolfin import GenericMatrix
    if isinstance(x, (matrix_op, diag_op)):
        return x
    elif isinstance(x, GenericMatrix):
        return matrix_op(x.down_cast().mat())
    elif isinstance(x, block_mat):
        if x.blocks.shape != (1,1):
            raise NotImplementedError("collapse() for block_mat with shape != (1,1)")
        return _collapse(x[0,0])
    elif isinstance(x, block_mul):
        factors = map(_collapse, x)
        while len(factors) > 1:
            A = factors.pop(0)
            B = factors[0]
            if isscalar(A):
                C = A*B if isscalar(B) else B.matmat(A)
            else:
                C = A.matmat(B)
            factors[0] = C

        return factors[0]
    elif isinstance(x, block_add):
        A,B = map(_collapse, x)
        if isscalar(A) and isscalar(B):
            return A+B
        else:
            return B.add(A) if isscalar(A) else A.add(B)
    elif isinstance(x, block_sub):
        A,B = map(_collapse, x)
        if isscalar(A) and isscalar(B):
            return A-B
        else:
            return B.add(A, lscale=-1.0) if isscalar(A) else A.add(B, rscale=-1.0)
    elif isinstance(x, block_transpose):
        A = _collapse(x.A)
        return A if isscalar(A) else A._transpose()
    elif isscalar(x):
        return x
    else:
        raise NotImplementedError("collapse() for type '%s'"%str(type(x)))

def collapse(x):
    """Compute an explicit matrix representation of an operator. For example,
    given a block_ mul object M=A*B, collapse(M) performs the actual matrix
    multiplication.
    """
    # Since _collapse works recursively, this method is a user-visible wrapper
    # to print timing, and to check input/output arguments.
    from time import time
    from dolfin import PETScMatrix, info, warning
    T = time()
    res = _collapse(x)
    if getattr(res, 'transposed', False):
        # transposed matrices will normally be converted to a non-transposed
        # one by matrix multiplication or addition, but if the transpose is the
        # outermost operation then this doesn't work.
        res.M.transpose()
    info('computed explicit matrix representation %s in %.2f s'%(str(res),time()-T))

    result = PETScMatrix(res.M)

    # Sanity check. Cannot trust EpetraExt.Multiply always, it seems.
    from block import block_vec
    v = x.create_vec()
    block_vec([v]).randomize()
    xv = x*v
    err = (xv-result*v).norm('l2')/(xv).norm('l2')
    if (err > 1e-3):
        raise RuntimeError('collapse computed wrong result; ||(a-a\')x||/||ax|| = %g'%err)

    return result


def create_identity(vec, val=1):
    raise NotImplementedError()

    """Create an identity matrix with the layout given by the supplied
    GenericVector. The values of the vector are NOT used."""
    import numpy
    from PyTrilinos import Epetra
    from dolfin import EpetraMatrix
    rowmap = vec.down_cast().vec().Map()
    graph = Epetra.CrsGraph(Epetra.Copy, rowmap, 1, True)
    indices = numpy.array([0], dtype=numpy.intc)
    for row in rowmap.MyGlobalElements():
        indices[0] = row
        graph.InsertGlobalIndices(row, indices)
    graph.FillComplete()

    matrix = EpetraMatrix(graph)
    indices = numpy.array(rowmap.MyGlobalElements())
    if val == 0:
        matrix.zero(indices)
    else:
        matrix.ident(indices)
        if val != 1:
            matrix *= val

    return matrix
