def _init():
    import block.algebraic
    class active_backend(object):
        name = 'petsc'
        def __call__(self):
            import sys
            return sys.modules[self.__module__]
    if block.algebraic.active_backend and block.algebraic.active_backend.name != 'petsc':
        raise ImportError('another backend is already active')
    block.algebraic.active_backend = active_backend()

    import dolfin
    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    import petsc4py, sys
    petsc4py.init(sys.argv)
_init()

from precond import *
from matrix import *
