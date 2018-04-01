from __future__ import division

from block.block_base import block_base

serial_solvers = ['Klu', 'Lapack', 'Umfpack', 'Taucs', 'Superlu']
parallel_solvers = ['Superludist', 'Mumps', 'Dscpack', 'Pardiso', 'Paraklete']

class AmesosSolver(block_base):
    __doc__ = \
    """Trilinos interface to direct solvers. The available solvers depend on
    your Trilinos installation, but the default (Klu) is usually available.

    Serial solvers: %s

    Parallel solvers: %s

    Call AmesosSolver.query() to list currently available solvers.
    """%(serial_solvers, parallel_solvers)

    default_solver = 'Klu'

    def __init__(self, A, solver=default_solver, parameters={}):
        from PyTrilinos import Epetra, Amesos
        from dolfin import info
        from time import time
        self.A = A # Keep reference
        T = time()
        self.problem = Epetra.LinearProblem()
        self.problem.SetOperator(A.down_cast().mat())
        self.solver = Amesos.Factory().Create(solver, self.problem)
        if (self.solver == None):
            raise RuntimeError('Failed to create Amesos solver: '+solver)

        # This prevents a use-after-free crash for the MPI communicator. It
        # enforces that the solver is destroyed before A.
        self.solver.A = A

        self.solver.SetParameters(parameters)
        if self.solver is None:
            raise RuntimeError("Unknown solver '%s'"%solver)
        err = self.solver.SymbolicFactorization()
        if err != 0:
            raise RuntimeError("Amesos " + solver + " symbolic factorization failed, err=%d"%err)
        err = self.solver.NumericFactorization()
        if err != 0:
            raise RuntimeError("Amesos " + solver + " numeric factorization failed, err=%d"%err)
        info('constructed direct solver (using %s) in %.2f s'%(solver,time()-T))

    @staticmethod
    def query(which=None):
        """Return list of available solver backends, or True/False if given a solver name"""
        from PyTrilinos import Amesos
        factory = Amesos.Factory()
        if which is None:
            avail = []
            for s in serial_solvers + parallel_solvers:
                if factory.Query(s):
                    avail.append(s)
            return avail
        return factory.Query(which)

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented()
        if self.A.size(0) != len(b):
            raise RuntimeError(
                'incompatible dimensions for Amesos matvec, %d != %d'%(len(self.b),len(b)))
        x = self.A.create_vec(dim=1)

        self.problem.SetLHS(x.down_cast().vec())
        self.problem.SetRHS(b.down_cast().vec())
        err = self.solver.Solve()
        self.problem.SetLHS(None)
        self.problem.SetRHS(None)

        if err != 0:
            raise RuntimeError("Amesos solve failed, err=%d"%err)
        return x

    def __str__(self):
        return '<%s solver for %s>'%(self.__class__.__name__, str(self.A))

class MumpsSolver(AmesosSolver):
    """Specialized Amesos-Mumps solver. The matrix_type argument may be 'SPD',
    'symmetric' or 'general'.

    NOTE 1: This functionality (matrix_type) is documented, but disabled in
    trilinos (see Amesos_Control.h)

    NOTE 2: The key that is used in the actual trilinos code is "MatrixProperty"
    but the documentation says "MatrixType".
    """
    def __init__(self, A, matrix_type='general', parameters={}):
        new_parameters = {'MatrixType': matrix_type, 'MatrixProperty': matrix_type}
        new_parameters.update(parameters)
        super(MumpsSolver, self).__init__(A, solver='Mumps', parameters=new_parameters)

