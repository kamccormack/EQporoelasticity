from block_mat import block_mat
from block_compose import block_mul

def block_kronecker(A, B):
    """Create the Kronecker (tensor) product of two matrices. The result is
    returned as a product of two block matrices, (A x Ib) * (Ia x B), where Ia
    and Ib are the appropriate identities (A=A*Ia, B=Ib*B). This will often
    limit the number of repeated applications of the inner operators.

    Note: If the A operator is a DOLFIN matrix and B is not a block_mat, A will
    be expanded into a block_mat (one block per matrix entry). This will not
    work in parallel. Apart from that, no blocks are expanded, i.e., only the
    block matrices are expanded in the Kronecker product.

    To form the Kronecker sum, you can extract (A x Ib) and (Ia x B) like this:
      C,D = block_kronecker(A,B); sum=C+D
    Similarly, it may be wise to do the inverse separately:
      C,D = block_kronecker(A,B); inverse = some_invert(D)*ConjGrad(C)
    """
    from block_util import isscalar
    import dolfin

    if isinstance(A, dolfin.GenericMatrix) and not isinstance(B, block_mat):
        A = block_mat(A.array())
    assert isinstance(A, block_mat) or isinstance(B, block_mat)

    if isinstance(B, block_mat):
        p,q = B.blocks.shape
        C = block_mat.diag(A, n=p)
    else:
        C = A

    if isinstance(A, block_mat):
        m,n = A.blocks.shape
        if isinstance(B, block_mat):
            print "Note: block_kronecker with two block matrices is probably buggy"
            D = block_mat(n,n)
            for i in range(n):
                for j in range(n):
                    # A scalar can represent the scaled identity of any
                    # dimension, so no need to diagonal-expand it here. We
                    # don't do this check on the outer diagonal expansions,
                    # because it is clearer to always return two block matrices
                    # of equal dimension rather than sometimes a scalar.
                    b = B[i,j]
                    D[i,j] = b if isscalar(b) else block_mat.diag(b, n=m)
        else:
            D = block_mat.diag(B, n=m)
    else:
        D = B

    return block_mul(C,D)

def block_simplify(expr):
    """Return a simplified (if possible) representation of a block matrix or
    block composition. The simplification does the following basic steps:
    - Convert scaled identity matrices to scalars
    - Combine scalar terms in compositions (2*2==4)
    - Eliminate additive and multiplicative identities (A+0=A, A*1=A)
    """
    if hasattr(expr, 'block_simplify'):
        return expr.block_simplify()
    else:
        return expr


def block_collapse(expr):
    """Turn a composition /inside out/, i.e., turn a composition of block
    matrices into a block matrix of compositions.
    """
    if hasattr(expr, 'block_collapse'):
        return expr.block_collapse()
    else:
        return expr
