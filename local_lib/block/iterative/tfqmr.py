from __future__ import division
from common import *

def tfqmr(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, callback=None):
    #####
    # Adapted from PyKrylov (https://github.com/dpo/pykrylov; LGPL license)
    #####

    r0 = b - A*x

    rho = inner(r0,r0)
    alphas = []
    betas = []
    residuals = [sqrt(rho)]

    if relativeconv:
        tolerance *= residuals[0]

    if residuals[-1] < tolerance:
        return x, residuals, [], []

    y = r0.copy()   # Initial residual vector
    w = r0.copy()
    d = 0*b
    theta = 0.0
    eta = 0.0
    k = 0

    z = B*y
    u = A*z
    v = u.copy()

    while k < maxiter:

        k += 1
        progress += 1
        sigma = inner(r0,v)
        alpha = rho/sigma

        # First pass
        w -= alpha * u
        d *= theta * theta * eta / alpha
        d += z

        residNorm = residuals[-1]
        theta = norm(w)/residNorm
        c = 1.0/sqrt(1 + theta*theta)
        residNorm *= theta * c
        eta = c * c * alpha
        x += eta * d
        m = 2.0 * k - 1.0
        if residNorm * sqrt(m+1) < tolerance:
            break

        # Second pass
        m += 1
        y -= alpha * v
        z = B*y

        u = A*z
        w -= alpha * u
        d *= theta * theta * eta / alpha
        d += z
        theta = norm(w)/residNorm
        c = 1.0/sqrt(1 + theta*theta)
        residNorm *= theta * c
        eta = c * c * alpha
        x += eta * d

        residual = residNorm * sqrt(m+1)

        # Call user provided callback with solution
        if callable(callback):
            callback(k=k, x=x, r=residual)

        residuals.append(residual)
        if residual < tolerance or k >= maxiter:
            break

        # Final updates
        rho_next = inner(r0,w)
        beta = rho_next/rho
        rho = rho_next

        alphas.append(alpha)
        betas.append(beta)

        # Update y
        y *= beta
        y += w

        # Partial update of v with current u
        v *= beta
        v += u
        v *= beta

        # Update u
        z = B*y
        u = A*z

        # Complete update of v
        v += u

    return x, residuals, alphas, betas
