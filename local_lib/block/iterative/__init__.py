"""A selection of iterative methods."""

from iterative import iterative

class ConjGrad(iterative):
    import conjgrad
    method = staticmethod(conjgrad.precondconjgrad)

class BiCGStab(iterative):
    import bicgstab
    method = staticmethod(bicgstab.precondBiCGStab)

class CGN(iterative):
    import cgn
    method = staticmethod(cgn.CGN_BABA)

class SymmLQ(iterative):
    import symmlq
    method = staticmethod(symmlq.symmlq)

class TFQMR(iterative):
    import tfqmr
    method = staticmethod(tfqmr.tfqmr)

class MinRes(iterative):
    import minres
    method = staticmethod(minres.minres)

class MinRes2(iterative):
    import minres2
    method = staticmethod(minres2.minres)



class LGMRES(iterative):
    import lgmres
    __doc__ = lgmres.lgmres.__doc__
    method = staticmethod(lgmres.lgmres)

class Richardson(iterative):
    import richardson
    method = staticmethod(richardson.richardson)
