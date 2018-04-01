from __future__ import division
from common import *

def symmlq(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, shift=0, callback=None):
    #####
    # Adapted from PyKrylov (https://github.com/dpo/pykrylov; LGPL license)
    #####

    msg={
        -1:' beta2 = 0.  If M = I, b and x are eigenvectors',
         0:' beta1 = 0.  The exact solution is  x = 0',
         1:' Requested accuracy achieved, as determined by tolerance',
         2:' Reasonable accuracy achieved, given eps',
         3:' x has converged to an eigenvector',
         4:' acond has exceeded 0.1/eps',
         5:' The iteration limit was reached',
         6:' aprod  does not define a symmetric matrix',
         7:' msolve does not define a symmetric matrix',
         8:' msolve does not define a pos-def preconditioner'}

    istop  = 0
    w      = 0*b

    # Set up y for the first Lanczos vector v1.
    # y is really beta1 * P * v1  where  P = C^(-1).
    # y and beta1 will be zero if b = 0.

    r1 = 1*b
    y  = B*r1

    beta1 = inner(r1, y)

    # Test for an indefinite preconditioner.
    # If b = 0 exactly, stop with x = 0.

    if beta1 < 0:
        raise ValueError('B does not define a pos-def preconditioner')
    if beta1 == 0:
        x *= 0
        return x, [0], [], []

    beta1 = sqrt(beta1)
    s     = 1.0 / beta1
    v     = s * y

    y = A*v

    # Set up y for the second Lanczos vector.
    # Again, y is beta * P * v2  where  P = C^(-1).
    # y and beta will be zero or very small if Abar = I or constant * I.

    if shift:
        y -= shift * v
    alfa = inner(v, y)
    y -= (alfa / beta1) * r1

    # Make sure  r2  will be orthogonal to the first  v.

    z  = inner(v, y)
    s  = inner(v, v)
    y -= (z / s) * v
    r2 = y
    y  = B*y

    oldb   = beta1
    beta   = inner(r2, y)
    if beta < 0:
        raise ValueError('B does not define a pos-def preconditioner')

    #  Cause termination (later) if beta is essentially zero.

    beta = sqrt(beta)
    if beta <= eps:
        istop = -1

    #  Initialize other quantities.
    rhs2   = 0
    tnorm  = alfa**2 + beta**2
    gbar   = alfa
    dbar   = beta

    bstep  = 0
    ynorm2 = 0
    snprod = 1

    gmin   = gmax   = abs(alfa) + eps
    rhs1   = beta1

    # ------------------------------------------------------------------
    # Main iteration loop.
    # ------------------------------------------------------------------
    # Estimate various norms and test for convergence.

    alphas = []
    betas = []
    residuals = [beta1/sqrt(tnorm)]
    itn = 0

    while True:
        itn   += 1
        progress += 1
        anorm  = sqrt(tnorm)
        ynorm  = sqrt(ynorm2)
        epsa   = anorm * eps
        epsx   = anorm * ynorm * eps
        epsr   = anorm * ynorm * tolerance
        diag   = gbar

        if diag == 0: diag = epsa

        lqnorm = sqrt(rhs1**2 + rhs2**2)
        qrnorm = snprod * beta1
        cgnorm = qrnorm * beta / abs(diag)

        # Estimate  Cond(A).
        # In this version we look at the diagonals of  L  in the
        # factorization of the tridiagonal matrix,  T = L*Q.
        # Sometimes, T(k) can be misleadingly ill-conditioned when
        # T(k+1) is not, so we must be careful not to overestimate acond

        if lqnorm < cgnorm:
            acond  = gmax / gmin
        else:
            acond  = gmax / min(gmin, abs(diag))

        zbar = rhs1 / diag
        z    = (snprod * zbar + bstep) / beta1

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above
        # (Abar = const * I).

        if istop == 0:
            if itn     >  maxiter    : istop = 5
            if acond   >= 0.1/eps    : istop = 4
            if epsx    >= beta1      : istop = 3
            if cgnorm  <= epsx       : istop = 2
            if cgnorm  <= epsr       : istop = 1

        residuals.append(cgnorm / anorm / (ynorm or 1))

        if istop !=0:
            break

        # Obtain the current Lanczos vector  v = (1 / beta)*y
        # and set up  y  for the next iteration.

        s = 1/beta
        v = s * y
        y = A*v
        if shift:
            y -= shift * v
        y -= (beta / oldb) * r1
        alfa = inner(v, y)
        y -= (alfa / beta) * r2
        r1 = r2.copy()
        r2 = y
        y = B*y
        oldb = beta
        beta = inner(r2, y)

        alphas.append(alfa)
        betas.append(beta)

        if beta < 0:
            raise ValueError('A does not define a symmetric matrix')

        beta  = sqrt(beta);
        tnorm = tnorm  +  alfa**2  +  oldb**2  +  beta**2;

        # Compute the next plane rotation for Q.

        gamma  = sqrt(gbar**2 + oldb**2)
        cs     = gbar / gamma
        sn     = oldb / gamma
        delta  = cs * dbar  +  sn * alfa
        gbar   = sn * dbar  -  cs * alfa
        epsln  = sn * beta
        dbar   =            -  cs * beta

        # Update  X.

        z = rhs1 / gamma
        s = z*cs
        t = z*sn
        x += s*w + t*v
        w *= sn
        w -= cs*v

        # Accumulate the step along the direction b, and go round again.

        bstep  = snprod * cs * z  +  bstep
        snprod = snprod * sn
        gmax   = max(gmax, gamma)
        gmin   = min(gmin, gamma)
        ynorm2 = z**2  +  ynorm2
        rhs1   = rhs2  -  delta * z
        rhs2   =       -  epsln * z

    # ------------------------------------------------------------------
    # End of main iteration loop.
    # ------------------------------------------------------------------

    # Move to the CG point if it seems better.
    # In this version of SYMMLQ, the convergence tests involve
    # only cgnorm, so we're unlikely to stop at an LQ point,
    # EXCEPT if the iteration limit interferes.

    if cgnorm < lqnorm:
        zbar   = rhs1 / diag
        bstep  = snprod * zbar + bstep
        ynorm  = sqrt(ynorm2 + zbar**2)
        x     += zbar * w

    # Add the step along b.

    bstep  = bstep / beta1
    y = B*b
    x += bstep * y

    if istop != 1:
        print 'SymmLQ:',msg[istop]

    return x, residuals, [], []
