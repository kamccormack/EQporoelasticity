from __future__ import division

"""These classes are typically not used directly, but returned by a call to
block_mat.scheme().
"""

from block_base import block_base

def block_jacobi(op):
    from block_mat import block_mat
    m,n = op.blocks.shape
    assert m==n
    mat = block_mat(m,n)
    for i in range(m):
        mat[i,i] = op[i,i]
    return mat

class block_gs(block_base):
    def __init__(self, op, reverse=False, truncated=False, symmetric=False, w=1.0):
        self.op = op
        self.range = range(len(op))
        if reverse:
            self.range.reverse()
        if symmetric:
            self.matvec = self.matvec_symmetric
        elif truncated:
            self.matvec = self.matvec_truncated
        else:
            self.matvec = self.matvec_full
        self.weight = w

    def sor_weighting(self, b, x):
        if self.weight != 1:
            for i in self.range:
                x[i] *= self.weight
                x[i] += (1-self.weight) * b[i]

    def matvec_full(self, b):
        x = b.copy()
        for i in self.range:
            for j in self.range:
                if j==i:
                    continue
                if self.op[i,j]:
                    x[i] -= self.op[i,j]*x[j]
            x[i] = self.op[i,i]*x[i]
        self.sor_weighting(b, x)
        return x

    def matvec_truncated(self, b):
        x = b.copy()
        for k,i in enumerate(self.range):
            for j in self.range[:k]:
                if self.op[i,j]:
                    x[i] -= self.op[i,j]*x[j]
            x[i] = self.op[i,i]*x[i]
        self.sor_weighting(b, x)
        return x

    def matvec_symmetric(self, b):
        x = b.copy()
        for k,i in enumerate(self.range):
            for j in self.range[:k]:
                if self.op[i,j]:
                    x[i] -= self.op[i,j]*x[j]
            x[i] = self.op[i,i]*x[i]
        rev_range = list(reversed(self.range))
        for k,i in enumerate(rev_range):
            for j in rev_range[:k]:
                if self.op[i,j]:
                    x[i] -= self.op[i,i]*self.op[i,j]*x[j]
        self.sor_weighting(b, x)
        return x

def blockscheme(op, scheme='jacobi', **kwargs):
    scheme = scheme.lower()
    if scheme == 'jacobi' or scheme == 'jac':
        return block_jacobi(op)

    if scheme in ('gauss-seidel', 'gs', 'sor'):
        return block_gs(op, **kwargs)

    if scheme in ('symmetric gauss-seidel', 'sgs', 'ssor'):
        return block_gs(op, symmetric=True, **kwargs)

    if scheme in ('truncated gauss-seidel', 'tgs', 'tsor'):
        return block_gs(op, truncated=True, **kwargs)

    raise TypeError('unknown scheme "%s"'%scheme)
