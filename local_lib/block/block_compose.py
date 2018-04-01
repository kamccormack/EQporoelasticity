from __future__ import division

"""Classes that define algebraic operations on matrices by deferring the actual
action until a vector is present. The classes are normally not used directly,
but instead returned implicitly by mathematical notation. For example,

M = A*B+C

returns the object

M = block_add(block_mul(A,B), C),

and the actual calculation is done later, once w=M*v is called for a vector v:

w=block_add.__mul__(M, v)
--> a = block_mul.__mul__((A,B), v)
----> b = Matrix.__mul__(B, v)
----> a = Matrix.__mul__(A, b)
--> b = Matrix.__mul__(C, v)
--> w = a+b
"""

from block_base import block_base

class block_mul(block_base):
    def __init__(self, A, B):
        """Args may be blockoperators or individual blocks (scalar or Matrix)"""
        A = A.chain if isinstance(A, block_mul) else [A]
        B = B.chain if isinstance(B, block_mul) else [B]
        self.chain = A+B

    def __mul__(self, x):
        for op in reversed(self.chain):
            from dolfin import GenericMatrix, GenericVector
            if isinstance(op, GenericMatrix) and isinstance(x, GenericVector):
                y = op.create_vec(dim=0)
                op.mult(x, y)
                x = y
            else:
                x = op * x
            if x == NotImplemented:
                return NotImplemented
        return x

    def transpmult(self, x):
        from block_util import isscalar
        for op in self.chain:
            if isscalar(op):
                if op != 1:
                    x = op*x
            else:
                x = op.transpmult(x)
        return x

    def create_vec(self, dim=1):
        """Create a vector of the correct size and layout for b (if dim==0) or
        x (if dim==1)."""

        # Use try-except because even if op has a create_vec method, it may be
        # a composed operator and its create_vec method may fail if it cannot
        # find a component to create the vector.
        #
        # The operator may be scalar, in which case it doesn't change the
        # dimensions, but it may also be something that reduces to a
        # scalar. Hence, a simple is_scalar is not generally sufficient, and we
        # just try the next operator in the chain. This is not completely safe,
        # in particular it may produce a vector of the wrong size if non-square
        # numpy matrices are involved.

        if dim==0:
            for op in self.chain:
                try:
                    return op.create_vec(dim)
                except AttributeError:
                    pass
        if dim==1:
            for op in reversed(self.chain):
                try:
                    return op.create_vec(dim)
                except AttributeError:
                    pass
        # Use AttributeError, because that's what the individual operators use,
        # and an outer create_vec will try the next candidate
        raise AttributeError('failed to create vec, no appropriate reference matrix')

    def block_collapse(self):
        """Create a block_mat of block_muls from a block_mul of
        block_mats. See block_transform.block_collapse."""
        from block_mat import block_mat
        from block_util import isscalar
        from block_transform import block_collapse, block_simplify

        # Reduce all composed objects
        ops = map(block_collapse, self.chain)

        # Do the matrix multiply, blockwise. Note that we use
        # block_mul(A,B) rather than A*B to avoid any implicit calculations
        # (e.g., scalar*matrix->matrix) -- the result will be transformed by
        # block_simplify() in the end to take care of any stray scalars.
        while len(ops) > 1:
            B = ops.pop()
            A = ops.pop()

            if isinstance(A, block_mat) and isinstance(B, block_mat):
                m,n = A.blocks.shape
                p,q = B.blocks.shape
                C = block_mat(m,q)
                for row in range(m):
                    for col in range(q):
                        for i in range(n):
                            C[row,col] += block_mul(A[row,i],B[i,col])
            elif isinstance(A, block_mat) and isscalar(B):
                m,n = A.blocks.shape
                C = block_mat(m,n)
                for row in range(m):
                    for col in range(n):
                        C[row,col] = block_mul(A[row,col],B)
            elif isinstance(B, block_mat) and isscalar(A):
                m,n = B.blocks.shape
                C = block_mat(m,n)
                for row in range(m):
                    for col in range(n):
                        C[row,col] = block_mul(A,B[row,col])
            else:
                C = block_mul(A,B)
            ops.append(C)
        return block_simplify(ops[0])

    def block_simplify(self):
        """Try to combine scalar terms and remove multiplicative identities,
        recursively. A fuller explanation is found in block_transform.block_simplify.
        """
        from block_util import isscalar
        from block_transform import block_simplify
        operators = []
        scalar = 1.0
        for op in self.chain:
            op = block_simplify(op)
            if isscalar(op):
                scalar *= op
            else:
                operators.append(op)
        if scalar == 0:
            return 0
        if scalar != 1 or len(operators) == 0:
            operators.insert(0, scalar)
        if len(operators) == 1:
            return operators[0]
        ret = block_mul(None, None)
        ret.chain = operators
        return ret

    def __str__(self):
        return '{%s}'%(' * '.join(op.__str__() for op in self.chain))

    def __iter__(self):
        return iter(self.chain)
    def __len__(self):
        return len(self.chain)
    def __getitem__(self, i):
        return self.chain[i]

class block_transpose(block_base):
    def __init__(self, A):
        self.A = A
    def matvec(self, x):
        return self.A.transpmult(x)
    def transpmult(self, x):
        return self.A.__mul__(x)

    def create_vec(self, dim=1):
        return self.A.create_vec(1-dim)

    def block_collapse(self):
        """See block_transform.block_collapse."""
        from block_transform import block_collapse, block_simplify
        from block_mat import block_mat
        A = block_collapse(self.A)
        if not isinstance(A, block_mat):
            return block_transpose(A)
        m,n = A.blocks.shape
        ret = block_mat(n,m)
        for i in range(m):
            for j in range(n):
                ret[j,i] = block_transpose(A[i,j])
        return block_simplify(ret)

    def block_simplify(self):
        """Try to simplify the transpose, recursively. A fuller explanation is
        found in block_transform.block_simplify.
        """
        from block_util import isscalar
        from block_transform import block_simplify
        A = block_simplify(self.A)
        if isscalar(A):
            return A
        if isinstance(A, block_transpose):
            return A.A
        return block_transpose(A)

    def __str__(self):
        return '<block_transpose of %s>'%str(self.A)
    def __iter__(self):
        return iter([self.A])
    def __len__(self):
        return 1
    def __getitem__(self, i):
        return [self.A][i]

# It's probably best if block_sub and block_add do not allow coercion into
# block_mul, since that might mess up the operator precedence. Hence, they
# do not inherit from block_base. As it is now, self.A*x and self.B*x must be
# reduced to vectors, which means all composed multiplies are finished before
# __mul__ does anything.

class block_sub(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def _combine(self, y, z):
        y -= z

    def _action(self, x, transposed):
        from block_mat import block_vec
        from block_util import mult
        from dolfin import GenericVector
        if not isinstance(x, (GenericVector, block_vec)):
            return NotImplemented
        y = mult(self.A, x, transposed)
        z = mult(self.B, x, transposed)
        if len(y) != len(z):
            raise RuntimeError(
                'incompatible dimensions in matrix subtraction -- %d != %d'%(len(y),len(z)))
        self._combine(y, z)
        return y

    def __mul__(self, x):
        return self._action(x, transposed=False)

    def transpmult(self, x):
        return self._action(x, transposed=True)

    def __neg__(self):
        return block_sub(self.B, self.A)

    def __add__(self, x):
        return block_add(self, x)

    def __sub__(self, x):
        return block_sub(self, x)

    def create_vec(self, dim=1):
        """Create a vector of the correct size and layout for b (if dim==0) or
        x (if dim==1)."""
        try:
            return self.A.create_vec(dim)
        except AttributeError:
            return self.B.create_vec(dim)

    def block_simplify(self):
        """Try to combine scalar terms and remove additive identities,
        recursively. A fuller explanation is found in block_transform.block_simplify.
        """
        from block_util import isscalar
        from block_transform import block_simplify
        A = block_simplify(self.A)
        B = block_simplify(self.B)
        if isscalar(A) and A==0:
            return -B
        if isscalar(B) and B==0:
            return A
        return A-B

    def block_collapse(self):
        """Create a block_mat of block_adds from a block_add of block_mats. See block_transform.block_collapse."""
        from block_mat import block_mat
        from block_util import isscalar
        from block_transform import block_collapse, block_simplify

        A = block_collapse(self.A)
        B = block_collapse(self.B)

        # The self.__class__(A,B) used below works for both block_sub and
        # block_add, and any scalar terms are combined by the final call to
        # block_simplify().
        if isinstance(A, block_mat) and isinstance(B, block_mat):
            m,n = A.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = self.__class__(A[row,col], B[row,col])
        elif isinstance(A, block_mat) and isscalar(B):
            m,n = A.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = self.__class__(A[row,col], B) if row==col else A[row,col]
        elif isinstance(B, block_mat) and isscalar(A):
            m,n = B.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = self.__class__(A, B[row,col]) if row==col else B[row,col]
        else:
            C = self.__class__(A, B)
        return block_simplify(C)

    def __str__(self):
        return '{%s - %s}'%(self.A.__str__(), self.B.__str__())

    def __iter__(self):
        return iter([self.A, self.B])
    def __len__(self):
        return 2
    def __getitem__(self, i):
        return [self.A, self.B][i]

class block_add(block_sub):
    def _combine(self, y, z):
        y += z

    def block_simplify(self):
        """Try to combine scalar terms and remove additive identities,
        recursively. A fuller explanation is found in block_transform.block_simplify.
        """
        from block_util import isscalar
        from block_transform import block_simplify
        A = block_simplify(self.A)
        B = block_simplify(self.B)
        if isscalar(A) and A==0:
            return B
        if isscalar(B) and B==0:
            return A
        return A+B

    def __neg__(self):
        return block_mul(-1, self)

    def __str__(self):
        return '{%s + %s}'%(self.A.__str__(), self.B.__str__())
