from __future__ import division

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
(block_vec, GenericVector))), or returns a block_mul(self, other) object which defers the
action until there is a proper vector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as
needed.
"""

from block_mat import block_mat
from block_vec import block_vec
from block_compose import block_mul, block_add, block_sub, block_transpose
from block_transform import block_kronecker, block_simplify, block_collapse
from block_assemble import block_assemble, block_symmetric_assemble
from block_bc import block_bc
from block_util import issymmetric

def _init():
    import dolfin
    from object_pool import vec_pool, store_args_ref
    from block_base import block_container

    # To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
    # methods into dolfin.(Generic)Matrix

    def check_type(obj1, obj2):
        if isinstance(obj2, block_container):
            raise TypeError('cannot apply dolfin operators on block containers:\n\t%s\nand\n\t%s'%(obj1,obj2))
        return True

    def inject_matrix_method(name, meth):
        setattr(dolfin.GenericMatrix, name, meth)
        setattr(dolfin.Matrix, name, meth)

    def inject_vector_method(name, meth):
        setattr(dolfin.GenericVector, name, meth)
        setattr(dolfin.Vector, name, meth)

    def wrap_mul(self, other):
        if isinstance(other, dolfin.GenericVector):
            ret = self.create_vec(dim=0)
            self.mult(other, ret)
            return ret
        else:
            check_type(self, other)
            return block_mul(self, other)
    inject_matrix_method('__mul__', wrap_mul)

    inject_matrix_method('__add__', lambda self, other: check_type(self, other) and block_add(self, other))
    inject_matrix_method('__sub__', lambda self, other: check_type(self, other) and block_sub(self, other))
    inject_matrix_method('__rmul__', lambda self, other: check_type(self, other) and block_mul(other, self))
    inject_matrix_method('__radd__', lambda self, other: check_type(self, other) and block_add(other, self))
    #inject_matrix_method('__rsub__', lambda self, other: check_type(self, other) and block_sub(other, self))
    inject_matrix_method('__neg__', lambda self : block_mul(-1, self))

    # Inject a new transpmult() method that returns the result vector (instead of output parameter)
    old_transpmult = dolfin.GenericMatrix.transpmult
    def transpmult(self, x, y=None):
        check_type(self, x)
        if y is None:
            y = self.create_vec(dim=1)
        old_transpmult(self, x, y)
        return y
    inject_matrix_method('transpmult', transpmult)

    # Inject a create() method that returns the new vector (instead of resize() which uses out parameter)
    def create_vec(self, dim=1):
        """Create a vector that is compatible with the matrix. Given A*x=b:
        If dim==0, the vector can be used for b (layout like the rows of A);
        if dim==1, the vector can be used for x (layout like the columns of A)."""
        vec = dolfin.Vector()
        self.init_vector(vec, dim)
        return vec

    inject_matrix_method('create_vec', vec_pool(create_vec))

    # HACK: The problem is that create_vec uses a pool of free vectors, but the internal
    # (shared_ptr) reference in Function is not visible in Python. This creates an explicit
    # Python-side reference to the Vector, so it's not considered re-usable too soon.
    dolfin.Function.__init__ = store_args_ref(dolfin.Function.__init__)

    # For the Trilinos stuff, it's much nicer if down_cast is a method on the
    # object. FIXME: Follow new dolfin naming? Invent our own?
    if hasattr(dolfin, 'as_backend_type'):
        inject_matrix_method('down_cast', dolfin.as_backend_type)
        inject_vector_method('down_cast', dolfin.as_backend_type)
    else:
        # Old name (before Sept-2012)
        inject_matrix_method('down_cast', dolfin.down_cast)
        inject_vector_method('down_cast', dolfin.down_cast)

    if not hasattr(dolfin.GenericMatrix, 'init_vector'):
        inject_matrix_method('init_vector', dolfin.GenericMatrix.resize)

    def T(self):
        from block_compose import block_transpose
        return block_transpose(self)
    inject_matrix_method('T', property(T))

    # Make sure PyTrilinos is imported somewhere, otherwise the types from
    # e.g. GenericMatrix.down_cast aren't recognised (if using Epetra backend).
    # Not tested, but assuming the same is true for the PETSc backend.
    for backend in ['PyTrilinos', 'petsc4py']:
        try:
            __import__(backend)
        except ImportError:
            pass

_init()
