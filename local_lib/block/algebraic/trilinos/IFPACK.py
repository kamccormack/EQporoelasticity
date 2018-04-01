from __future__ import division

from block.block_base import block_base

class IFPACK(block_base):

    errcode = {1 : "Generic Error (called method or function returned an error)",
               2 : "Input data not valid (wrong parameter, out-of-bounds, wrong dimensions, matrix is not square,...)",
               3 : "Data has not been correctly pre-processed",
               4 : "Problem encountered during application of the algorithm (division by zero, out-of-bounds, ...)",
               5 : "Memory allocation error",
               22: "Matrix is numerically singular",
               98: "Feature is not supported",
               99: "Feature is not implemented yet (check Known Bugs and Future Developments, or submit a bug)"}

    params = {}

    def __init__(self, A, overlap=0, params={}):
        from PyTrilinos.IFPACK import Factory
        from dolfin import info
        from time import time

        self.A = A # Keep reference to avoid delete

        T = time()
        prectype = self.prectype
        if overlap == 0:
            prectype += ' stand-alone' # Skip the additive Schwarz step

        self.prec = Factory().Create(prectype, A.down_cast().mat(), overlap)
        if not self.prec:
            raise RuntimeError("Unknown IFPACK preconditioner '%s'"%prectype)

        paramlist = {'schwartz: combine mode' : 'Add'} # Slower than 'Zero', but symmetric
        paramlist.update(self.params)
        paramlist.update(params)

        assert (0 == self.prec.SetParameters(paramlist))
        assert (0 == self.prec.Initialize())
        err = self.prec.Compute()
        if err:
            raise RuntimeError('Compute returned error %d: %s'%(err, self.errcode.get(-err)))
        info('Constructed %s in %.2f s'%(self.__class__.__name__,time()-T))

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for AztecOO matvec, %d != %d'%(len(x),len(b)))

        err = self.prec.ApplyInverse(b.down_cast().vec(), x.down_cast().vec())
        if err:
            raise RuntimeError('ApplyInverse returned error %d: %s'%(err, self.errcode.get(-err)))
        return x

    def down_cast(self):
        return self.prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

# "point relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_PointRelaxation>
# "block relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_BlockRelaxation>
# "Amesos" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_Amesos>.
# "IC" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_IC>.
# "ICT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ICT>.
# "ILU" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILU>.
# "ILUT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILUT>.
# otherwise, Create() returns 0.

class DD_Jacobi(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'Jacobi'}
Jacobi = DD_Jacobi

class DD_GaussSeidel(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'Gauss-Seidel'}
GaussSeidel = DD_GaussSeidel

class DD_SymmGaussSeidel(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'symmetric Gauss-Seidel'}
SymmGaussSeidel = DD_SymmGaussSeidel

class DD_BJacobi(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'Jacobi'}
BJacobi = DD_BJacobi

class DD_BGaussSeidel(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'Gauss-Seidel'}
BGaussSeidel = DD_BGaussSeidel

class DD_BSymmGaussSeidel(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'symmetric Gauss-Seidel'}
BSymmGaussSeidel = DD_BSymmGaussSeidel

class DD_ILU(IFPACK):
    """Incomplete LU factorization"""
    prectype = 'ILU'
ILU = DD_ILU

class DD_ILUT(IFPACK):
    """ILU with threshold"""
    prectype = 'ILUT'
ILUT = DD_ILUT

class DD_IC(IFPACK):
    """Incomplete Cholesky factorization"""
    prectype = 'IC'
IC = DD_IC

class DD_ICT(IFPACK):
    """IC with threshold"""
    prectype = 'ICT'
ICT = DD_ICT

class DD_Amesos(IFPACK):
    prectype = 'Amesos'
    def __init__(self, A, solver='Klu', **kwargs):
        self.params.update({'amesos: solver type': 'Amesos_'+solver})
        super(DD_Amesos, self).__init__(A, **kwargs)

del IFPACK
