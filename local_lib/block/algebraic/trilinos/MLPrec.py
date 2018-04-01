from __future__ import division

#####
# Original author: Kent Andre Mardal <kent-and@simula.no>
#####

from block.block_base import block_base

class ML(block_base):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        from PyTrilinos.ML import MultiLevelPreconditioner
        from dolfin import info
        from time import time
        T = time()
        # create the ML preconditioner
        MLList = {
            #"max levels"                : 30,
#             "ML print final list"       : -1,  
            #"ML output"                 : 10,
#            "coarse: type" : "Amesos-KLU",
            "smoother: type"            : "ML symmetric Gauss-Seidel" ,
            "smoother: sweeps"          : 2,
#            "smoother: damping factor"  : 0.8, 
            #"cycle applications"        : 2,
            "prec type"                 : "MGV",
#            "aggregation: type"         : "METIS" ,
#            "aggregation: damping factor": 1.6, 
#            "aggregation: smoothing sweeps" : 3,  

            "PDE equations"             : pdes,
#            "ML validate parameter list": True,
#            "repartition: enable": 1,
            }

        if parameters: MLList.update(parameters)
        self.A = A # Prevent matrix being deleted
        self.ml_prec = MultiLevelPreconditioner(A.down_cast().mat(), 0)
        if nullspace:
            # Convert to MultiVector if necessary
            from PyTrilinos import Epetra
            if not isinstance(nullspace, Epetra.MultiVector):
                n = len(nullspace)
                mv = Epetra.MultiVector(nullspace[0].down_cast().vec().Map(), n, False)
                for i in range(n):
                    mv[i] = Epetra.FEVector(nullspace[i].down_cast().vec())
                nullspace = mv
            self.ml_prec.SetParameterListAndNullSpace(MLList, nullspace)
        else:
            self.ml_prec.SetParameterList(MLList)
        self.ml_agg = self.ml_prec.GetML_Aggregate()
        err = self.ml_prec.ComputePreconditioner()
        if err:
            raise RuntimeError('ComputePreconditioner returned %d'%err)
        info('constructed ML preconditioner in %.2f s'%(time()-T))

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        # apply the ML preconditioner
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for ML matvec, %d != %d'%(len(x),len(b)))

        err = self.ml_prec.ApplyInverse(b.down_cast().vec(), x.down_cast().vec())
        if err:
            raise RuntimeError('ApplyInverse returned %d'%err)
        return x

    def down_cast(self):
        return self.ml_prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))
