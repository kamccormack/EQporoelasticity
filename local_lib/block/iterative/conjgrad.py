from __future__ import division
from common import *

def precondconjgrad(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, robustresidual=False, callback=None):
    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license). This code
    # relicensed under LGPL v2.1 or later, in agreement with the original
    # authors:
    # Kent Andre Mardal <kent-and@simula.no>
    # Ola Skavhaug <skavhaug@simula.no>
    # Gunnar Staff <gunnaran@simula.no>
    #####

    r = b - A*x
    z = B*r
    d = z
    rz = inner(r,z)
    if rz < 0:
        raise ValueError('Matrix is not positive')

    iter = 0
    alphas = []
    betas = []
    residuals = [sqrt(rz)]

    if relativeconv:
        tolerance *= residuals[0]

    while residuals[-1] > tolerance and iter < maxiter:
        z = A*d
        dz = inner(d,z)
        if dz == 0:
            print 'ConjGrad breakdown'
            break
        alpha = rz/dz
        x += alpha*d
        if robustresidual:
            r = b - A*x
        else:
            r -= alpha*z
        z = B*r
        rz_prev = rz
        rz = inner(r,z)
        if rz < 0:
            print 'ConjGrad breakdown'
            # Restore pre-breakdown state. Don't know if it helps any, but it's
            # consistent with returned quasi-residuals.
            x -= alpha*d
            break
        beta = rz/rz_prev
        d = z + beta*d

        residual = sqrt(rz)

        # Call user provided callback with solution
        if callable(callback):
            callback(k=iter, x=x, r=residual)

        iter += 1
        progress += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(residual)

    return x, residuals, alphas, betas
