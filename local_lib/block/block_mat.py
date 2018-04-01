from __future__ import division
from block_base import block_container
from block_vec import block_vec

class block_mat(block_container):
    """Block of matrices or other operators. Empty blocks doesn't need to be set
    (they may be None or zero), but each row must have at least one non-empty block.

    As a special case, the diagonal may contain scalars which act as the
    (scaled) identity operators."""

    def __init__(self, m, n=None, blocks=0):
        if n is None:
            blocks = m
            m = len(blocks)
            n = len(blocks[0]) if m else 0
        block_container.__init__(self, (m,n), blocks)

    def matvec(self, x):
        from dolfin import GenericVector, GenericMatrix
        from block_util import isscalar
        import numpy
        m,n = self.blocks.shape
        y = block_vec(m)

        for i in range(m):
            for j in range(n):
                if isinstance(self[i,j], (numpy.ndarray, numpy.matrix)):
                    z = numpy.matrix(self[i,j]) * numpy.matrix(x[j].array()).transpose()
                    z = numpy.array(z).flatten()
                    if y[i] is None: 
                        y[i] = x[j].copy()
                        y[i].resize(len(z))
                    y[i][:] += z[:]
                    continue
                if self[i,j] is None or self[i,j]==0 \
                        or x[j] is None or (isscalar(x[j]) and x[j]==0):
                    # Skip multiply if zero
                    continue
                if self[i,j] == 1:
                    # Skip multiply if identity
                    z = x[j]
                else:
                    # Do the block multiply
                    if isinstance(self[i,j], GenericMatrix):
                        z = self[i,j].create_vec(dim=0)
                        self[i,j].mult(x[j], z)
                    else:
                        z = self[i,j] * x[j]
                        if z == NotImplemented: return NotImplemented
                if not isinstance(z, (GenericVector, block_vec)):
                    # Typically, this happens when for example a
                    # block_vec([0,0]) is used without calling allocate() or
                    # setting BCs. The result is a Matrix*scalar=Matrix. One
                    # way to fix this issue is to convert all scalars to a
                    # proxy class in block_vec.__init__, and let this proxy
                    # class have a __rmul__ that converts to vector on demand.
                    # (must also stop conversion anything->blockcompose for
                    # this case)
                    raise RuntimeError(
                        'unexpected result in matvec, %s\n-- possibly because a block_vec contains scalars ' \
                        'instead of vectors, use create_vec() or allocate()' % type(z))
                if y[i] is None:
                    y[i]  = z
                else:
                    if len(y[i]) != len(z):
                        raise RuntimeError(
                            'incompatible dimensions in block (%d,%d) -- %d, was %d'%(i,j,len(z),len(y[i])))
                    y[i] += z
        return y

    def transpmult(self, x):
        import numpy
        from dolfin import GenericVector

        m,n = self.blocks.shape
        y = block_vec(self.blocks.shape[0])

        for i in range(n):
            for j in range(m):
                if self[j,i] is None or self[j,i]==0:
                    # Skip multiply if zero
                    continue
                if self[i,j] == 1:
                    # Skip multiply if identity
                    z = x[j]
                elif numpy.isscalar(self[j,i]):
                    # mult==transpmult
                    z = self[j,i]*x[j]
                else:
                    # Do the block multiply
                    z = self[j,i].transpmult(x[j])
                if not isinstance(z, (GenericVector, block_vec)):
                    # see comment in matvec
                    raise RuntimeError(
                        'unexpected result in matvec, %s\n-- possibly because RHS contains scalars ' \
                        'instead of vectors, use create_vec() or allocate()' % type(z))
                if y[i] is None:
                    y[i] = z
                else:
                    if len(y[i]) != len(z):
                        raise RuntimeError(
                            'incompatible dimensions in block (%d,%d) -- %d, was %d'%(i,j,len(z),len(y[i])))
                    y[i] += z
        return y

    def copy(self):
        import block_util
        m,n = self.blocks.shape
        y = block_mat(m,n)
        for i in range(m):
            for j in range(n):
                y[i,j] = block_util.copy(self[i,j])
        return y

    def create_vec(self, dim=1):
        """Create a vector of the correct size and layout for b (if dim==0) or
        x (if dim==1)."""
        xx = block_vec(self.blocks.shape[dim])
        xx.allocate(self, dim)
        return xx

    def scheme(self, name, inverse=None, **kwargs):
        """Return a block scheme (block relaxation method). The input block_mat
        is normally expected to be defined with inverses (or preconditioners)
        on the diagonal, and the normal blocks off-diagonal. For example, given
        a coefficient matrix

          AA = block_mat([[A, B],
                          [C, D]]),

        the Gauss-Seidel block preconditioner may be defined as for example

          AApre = block_mat([[ML(A), B],
                             [C, ML(D)]]).scheme('gs')

        However, for the common case of using the same preconditioner for each
        block, the block preconditioner may be specified in a simpler way as

          AApre = AA.scheme('gs', inverse=ML)

        where "inverse" defines a class or method that may be applied to each
        diagonal block to form its (approximate) inverse.

        The returned block operator may be a block_mat itself (this is the case
        for Jacobi), or a procedural operator (this is the case for the
        Gauss-Seidel variants).

        The scheme may be one of the following (aliases in brackets):

          jacobi [jac]
          gauss-seidel [gs]
          symmetric gauss-seidel [sgs]
          truncated gauss-seidel [tgs]
          sor
          ssor
          tsor

        and they may take keyword arguments. For the Gauss-Seidel-like methods,
        'reverse' and 'w' (weight) are supported, as well as 'truncated' and
        'symmetric'.
        """
        from block_scheme import blockscheme
        if inverse is not None:
            m,n = self.blocks.shape
            mat = block_mat(m,n)
            mat[:] = self[:]
            for i in range(m):
                mat[i,i] = inverse(mat[i,i])
        else:
            mat = self
        return blockscheme(mat, name, **kwargs)

    @staticmethod
    def diag(A, n=0):
        """Create a diagonal block matrix, where the entries on the diagonal
        are either the entries of the block_vec or list A (if n==0), or n
        copies of A (if n>0). For the case of extracting the diagonal of an
        existing block matrix, use D=A.scheme('jacobi') instead.

        If n>0, A is replicated n times and may be either an operator or a
        block_vec/list. If it is a list, the entries of A define a banded
        diagonal. In this case, len(A) must be odd, with the diagonal in the
        middle.
        """
        if n==0:
            n = len(A)
            mat = block_mat(n,n)
            for i in range(n):
                mat[i,i] = A[i]
        else:
            if isinstance(A, (list, tuple, block_vec)):
                if len(A)%2 != 1:
                    raise ValueError('The number of entries in the banded diagonal must be odd')
            else:
                A = [A]
            mat = block_mat(n,n)
            m = len(A)
            for i in range(n):
                for k in range(m):
                    j = i-m//2+k
                    if not 0 <= j < n:
                        continue
                    mat[i,j] = A[k]
        return mat

    def block_simplify(self):
        """Try to convert identities to scalars, recursively. A fuller
        explanation is found in block_transform.block_simplify.
        """
        from block_util import isscalar
        from block_transform import block_simplify
        m,n = self.blocks.shape
        res = block_mat(m,n)
        # Recursive call
        for i in range(m):
            for j in range(n):
                res[i,j] = block_simplify(self[i,j])
        # Check if result after recursive conversion is the (scaled) identity
        v0 = res.blocks[0,0]
        if m != n:
            return res
        for i in range(m):
            for j in range(n):
                block = res.blocks[i,j]
                if not isscalar(block):
                    return res
                if i==j:
                    if block!=v0:
                        return res
                else:
                    if block!=0:
                        return res
        return v0
