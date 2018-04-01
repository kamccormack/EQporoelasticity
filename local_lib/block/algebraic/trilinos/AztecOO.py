from __future__ import division

from block.block_base import block_base

class AztecSolver(block_base):
    def __init__(self, A, tolerance=1e-5, maxiter=300, solver='cg', precond=None):
        from PyTrilinos import AztecOO
        self.A = A # Keep reference
        self.solver = getattr(AztecOO, 'AZ_'+solver)
        if isinstance(precond, str):
            self.precond = getattr(AztecOO, 'AZ_'+precond)
        else:
            self.precond = precond
        self.tolerance = tolerance
        self.maxiter = maxiter

    def matvec(self, b):
        from PyTrilinos import AztecOO
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for AztecOO matvec, %d != %d'%(len(x),len(b)))

        solver = AztecOO.AztecOO(self.A.down_cast().mat(), x.down_cast().vec(), b.down_cast().vec())
        #solver.SetAztecDefaults()
        solver.SetAztecOption(AztecOO.AZ_solver, self.solver)
        if self.precond:
            if hasattr(self.precond, 'down_cast'):
                solver.SetPrecOperator(self.precond.down_cast())
            else:
                # doesn't seem to work very well
                solver.SetAztecOption(AztecOO.AZ_precond, self.precond)
                # the following are from the example with precond='dom_decomp'
                solver.SetAztecOption(AztecOO.AZ_subdomain_solve, AztecOO.AZ_ilu)
                solver.SetAztecOption(AztecOO.AZ_overlap, 1)
                solver.SetAztecOption(AztecOO.AZ_graph_fill, 1)

        solver.SetAztecOption(AztecOO.AZ_output, 0)
        solver.Iterate(self.maxiter, self.tolerance)
        return x

    def __str__(self):
        return '<%s solver for %s>'%(self.__class__.__name__, str(self.A))
