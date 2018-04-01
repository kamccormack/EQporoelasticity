from __future__ import division
import dolfin
from block_mat import block_mat
from block_vec import block_vec
import numpy

class block_bc(list):
    """This class applies Dirichlet BCs to a block matrix. It is not a block operator itself."""
    def __init__(self, lst, symmetric):
        list.__init__(self, lst)
        # Clean up self, and check arguments
        for i in range(len(self)):
            if self[i] is None:
                self[i] = []
            elif not hasattr(self[i], '__iter__'):
                self[i] = [self[i]]
        self.symmetric = symmetric

    def apply(self, A):
        if not isinstance(A, block_mat):
            raise RuntimeError('A is not a block matrix')

        # Create rhs_bc with a copy of A before any symmetric modifications
        rhs_bc = block_rhs_bc(self, A.copy() if self.symmetric else None)

        self._apply(A, A.create_vec() if self.symmetric else None)

        return rhs_bc

    def _apply(self, A, b):
        for i,bcs in enumerate(self):
            for bc in bcs:
                for j in range(len(self)):
                    if i==j:
                        if numpy.isscalar(A[i,i]):
                            # Convert to a diagonal matrix, so that the individual rows can be modified
                            from block_assemble import _new_square_matrix
                            A[i,i] = _new_square_matrix(bc, A[i,i])
                        if self.symmetric:
                            bc.zero_columns(A[i,i], b[i], 1.0)
                        else:
                            bc.apply(A[i,i])
                    else:
                        if numpy.isscalar(A[i,j]):
                            if A[i,j] != 0.0:
                                dolfin.error("can't modify block (%d,%d) for BC, expected a GenericMatrix" % (i,j))
                        else:
                            bc.zero(A[i,j])
                        if self.symmetric:
                            if numpy.isscalar(A[j,i]):
                                if A[j,i] != 0.0:
                                    dolfin.error("can't modify block (%d,%d) for BC, expected a GenericMatrix" % (j,i))
                            else:
                                bc.zero_columns(A[j,i], b[j])

class block_rhs_bc(list):
    def __init__(self, bc, A):
        list.__init__(self, bc)
        self.A = A;

    def apply(self, b):
        if not isinstance(b, block_vec):
            raise RuntimeError('not a block vector')

        if self.A is not None:
            b.allocate(self.A)
        else:
            from block_util import isscalar, _create_vec
            for i,bcs in enumerate(self):
                for bc in bcs:
                    if isscalar(b[i]) and b[i] == 0.0:
                        v = _create_vec(bc)
                        if v is not None:
                            v.zero()
                            b[i] = v


        if self.A is not None:
            # First, collect a vector containing all non-zero BCs. These are required for
            # symmetric modification.
            b_mod = b.copy()
            b_mod.zero()
            for i,bcs in enumerate(self):
                for bc in bcs:
                    bc.apply(b_mod[i])

            # The non-zeroes of b_mod are now exactly the x values (assuming the
            # matrix diagonal is in fact 1). We can thus create the necessary modifications
            # to b by just multiplying with the un-symmetricised original matrix.
            b -= self.A * b_mod

        # Apply the actual BC dofs to b. (This must be done after the symmetric
        # correction above, since the correction might also change the BC dofs.)
        for i,bcs in enumerate(self):
            for bc in bcs:
                bc.apply(b[i])

        return self
