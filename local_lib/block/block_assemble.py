from __future__ import division
#from block import *
from block_util import block_tensor, isscalar, wrap_in_list, create_vec_from

def block_assemble(lhs, rhs=None, bcs=None,
                   symmetric=False, signs=None, symmetric_mod=None):
    """
    Assembles block matrices, block vectors or block systems.
    Input can be arrays of variational forms or block matrices/vectors.

    Arguments:

            symmetric : Boundary conditions are applied so that symmetry of the system
                        is preserved. If only the left hand side of the system is given,
                        then a matrix represententing the rhs corrections is returned
                        along with a symmetric matrix.

        symmetric_mod : Matrix describing symmetric corrections for assembly of the
                        of the rhs of a variational system.

                signs : An array to specify the signs of diagonal blocks. The sign
                        of the blocks are computed if the argument is not provided.
    """
    error_msg = {'incompatibility' : 'A and b do not have compatible dimensions.',
                 'symm_mod error'  : 'symmetric_mod argument only accepted when assembling a vector',
                 'not square'      : 'A must be square for symmetric assembling',
                 'invalid bcs'     : 'Expecting a list or list of lists of DirichletBC.',
                 'invalid signs'   : 'signs should be a list of length n containing only 1 or -1',
                 'mpi and symm'    : 'Symmetric application of BC not yet implemented in parallel'}
    # Check arguments
    from numpy import ndarray
    has_rhs = True if isinstance(rhs, ndarray) else rhs != None
    has_lhs = True if isinstance(rhs, ndarray) else rhs != None

    if symmetric:
        from dolfin import MPI, mpi_comm_world
        if MPI.size(mpi_comm_world()) > 1:
            raise NotImplementedError(error_msg['mpi and symm'])
    if has_lhs and has_rhs:
        A, b = map(block_tensor,[lhs,rhs])
        n, m = A.blocks.shape
        if not ( isinstance(b,block_vec) and  len(b.blocks) is m):
            raise TypeError(error_msg['incompatibility'])
    else:
        A, b = block_tensor(lhs), None
        if isinstance(A,block_vec):
            A, b = None, A
            n, m = 0, len(b.blocks)
        else:
            n,m = A.blocks.shape
    if A and symmetric and (m is not n):
        raise RuntimeError(error_msg['not square'])
    if symmetric_mod and ( A or not b ):
        raise RuntimeError(error_msg['symmetric_mod error'])
    # First assemble everything needing assembling.
    from dolfin import assemble
    assemble_if_form = lambda x: assemble(x, keep_diagonal=True) if _is_form(x) else x
    if A:
        A.blocks.flat[:] = map(assemble_if_form,A.blocks.flat)
    if b:
        #b.blocks.flat[:] = map(assemble_if_form, b.blocks.flat)
        b = block_vec(map(assemble_if_form, b.blocks.flat))
    # If there are no boundary conditions then we are done.
    if bcs is None:
        if A:
            return [A,b] if b else A
        else:
            return b

    # check if arguments are forms, in which case bcs have to be split
    from ufl import Form
    if isinstance(lhs, Form):
        from splitting import split_bcs
        bcs = split_bcs(bcs, m)
    # Otherwise check that boundary conditions are valid.
    if not hasattr(bcs,'__iter__'):
        raise TypeError(error_msg['invalid bcs'])
    if len(bcs) is not m:
        raise TypeError(error_msg['invalid bcs'])
    from dolfin import DirichletBC
    for bc in bcs:
        if isinstance(bc,DirichletBC) or bc is None:
            pass
        else:
            if not hasattr(bc,'__iter__'):
                raise TypeError(error_msg['invalid bcs'])
            else:
                for bc_i in bc:
                    if isinstance(bc_i,DirichletBC):
                        pass
                    else:
                        raise TypeError(error_msg['invalid bcs'])
    bcs = [bc if hasattr(bc,'__iter__') else [bc] if bc else bc for bc in bcs]
    # Apply BCs if we are only assembling the righ hand side
    if not A:
        if symmetric_mod:
            b.allocate(symmetric_mod)
        for i in xrange(m):
            if bcs[i]:
                if isscalar(b[i]):
                    b[i], val = create_vec_from(bcs[i][0]), b[i]
                    b[i][:] = val
                for bc in bcs[i]: bc.apply(b[i])
        if symmetric_mod:
            b.allocate(symmetric_mod)
            b -= symmetric_mod*b
        return b
    # If a signs argument is passed, check if it is valid.
    # Otherwise guess.
    if signs and symmetric:
        if ( hasattr(signs,'__iter__')  and len(signs)==m ):
            for sign in signs:
                if sign not in (-1,1):
                    raise TypeError(error_msg['invalid signs'])
        else:
            raise TypeError(error_msg['invalid signs'])
    elif symmetric:
        from numpy.random import random
        signs = [0]*m
        for i in xrange(m):
            if isscalar(A[i,i]):
                signs[i] = -1 if A[i,i] < 0 else 1
            else:
                x = A[i,i].create_vec(dim=1)
                x.set_local(random(x.local_size()))
                signs[i] = -1 if x.inner(A[i,i]*x) < 0 else 1
    # Now apply boundary conditions.
    if b:
        b.allocate(A)
    elif symmetric:
        # If we are preserving symmetry but don't have the rhs b,
        # then we need to store the symmetric corretions to b
        # as a matrix which we call A_mod
        b, A_mod = A.create_vec(), A.copy()
    for i in xrange(n):
        if bcs[i]:
            for bc in bcs[i]:
                # Apply BCs to the diagonal block.
                if isscalar(A[i,i]):
                    A[i,i] = _new_square_matrix(bc,A[i,i])
                    if symmetric:
                        A_mod[i,i] = A[i,i].copy()
                if symmetric:
                    bc.zero_columns(A[i,i],b[i],signs[i])
                    bc.apply(A_mod[i,i])
                elif b:
                    bc.apply(A[i,i],b[i])
                else:
                    bc.apply(A[i,i])
                # Zero out the rows corresponding to BC dofs.
                for j in range(i) + range(i+1,n):
                    if A[i,j] is 0:
                        continue
                    assert not isscalar(A[i,j])
                    bc.zero(A[i,j])
                # If we are not preserving symmetry then we are done at this point.
                # Otherwise, we need to zero out the columns as well
                if symmetric:
                    for j in range(i) + range(i+1,n):
                        if A[j,i] is 0:
                            continue
                        assert not isscalar(A[j,i])
                        bc.zero_columns(A[j,i],b[j])
                        bc.zero(A_mod[i,j])

    result = [A]
    if symmetric:
        for i in range(n):
            for j in range(n):
                A_mod[i,j] -= A[i,j]
        result += [A_mod]
    if b:
        result += [b]
    return result[0] if len(result)==1 else result


def block_symmetric_assemble(forms, bcs):
    return block_assemble(forms,bcs=bcs,symmetric=True)

def _is_form(form):
    from dolfin.cpp import Form as cpp_Form
    from ufl.form import Form as ufl_Form
    return isinstance(form, (cpp_Form, ufl_Form))

def _new_square_matrix(bc, val):
    from dolfin import TrialFunction, TestFunction
    from dolfin import assemble, Constant, inner, dx
    import numpy
    V = bc.function_space()
    u,v = TrialFunction(V),TestFunction(V)
    Z = assemble(Constant(0)*inner(u,v)*dx)
    if val != 0.0:
        lrange = range(*Z.local_range(0))
        idx = numpy.ndarray(len(lrange), dtype=numpy.intc)
        idx[:] = lrange
        Z.ident(idx)
        if val != 1.0:
            Z *= val
    return Z
