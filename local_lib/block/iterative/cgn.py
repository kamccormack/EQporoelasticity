from __future__ import division
from common import *

def CGN_BABA(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, callback=None):

    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license). This code
    # relicensed under LGPL v2.1 or later, in agreement with the original
    # authors:
    # Kent Andre Mardal <kent-and@simula.no>
    # Ola Skavhaug <skavhaug@simula.no>
    # Gunnar Staff <gunnaran@simula.no>
    #####

    # Is this correct?? Should we not transpose the preconditoner somewhere??

    # jobh 02/2011: Changed residual from sqrt(rho) to sqrt(inner(r,r)) due to negative rho

    r     = b - A*x
    Br    = B*r
    ATBr  = transpmult(A, Br)
    BATBr = B*ATBr


    rho   = inner(BATBr,ATBr)
    if rho < 0:
        raise RuntimeError('CGN: Preconditioner not positive-definite')
    rho1  = rho
    p     = BATBr.copy()

    iter   = 0
    alphas = []
    betas  = []
    residuals = [sqrt(rho)]

    if relativeconv:
        tolerance *= residuals[0]

    while residuals[-1] > tolerance and iter <= maxiter:
        Ap     = A*p
        BAp    = B*Ap;
        ATBAp  = transpmult(A, BAp)
        alpha  = rho/inner(p,ATBAp)
        x      = x + alpha*p
        r      = b-A*x
        Br     = B*r
        ATBr   = transpmult(A, Br)
        BATBr  = B*ATBr

        rho    = inner(BATBr,ATBr)
        if rho < 0:
            raise RuntimeError('CGN: Preconditioner not positive-definite')
        beta   = rho/rho1
        rho1   = rho
        p      = BATBr+beta*p

        residual = sqrt(rho)

        # Call user provided callback with solution
        if callable(callback):
            callback(k=iter, x=x, r=residual)

        iter     += 1
        progress += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(residual)

    return x, residuals, alphas, betas
