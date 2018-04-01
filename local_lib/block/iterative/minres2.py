
"""
@license: Copyright (C) 2006
Author: Gunnar Staff

Simula Research Laboratory AS

This file is part of PyCC.

PyCC  free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

PyCC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyFDM; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


MinRes: A generic implementation of the Minimum Residual method.
"""


MinResError = "Error in MinRes"

import numpy as _n #/numarray
#from Utils import _n,inner
from math import sqrt,fabs

def inner(u,v):
    """Compute innerproduct of u and v.
       It is not computed here, we just check type of operands and
       call a suitable innerproduct method"""

    if isinstance(u,_n.ndarray):
        # assume both are numarrays:
        return _n.dot(u,v)
    else:
        # assume that left operand implements inner
        return u.inner(v)


def minres(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, rit_=1, callback=None):
    """
    precondMinRes(B,A,x,b): Solve Ax = b with the preconditioned minimum
    residual method.

    @param B: Preconditioner supporting the __mul__ operator for operating on
    x. B must be symmetric positive definite

    @param A: Matrix or discrete operator. Must support A*x to operate on x.
    A must be symmetric, but may be indefinite.

    @param x: Anything that A can operate on, as well as support inner product
    and multiplication/addition/subtraction with scalar and something of the
    same type

    @param b: Right-hand side, probably the same type as x.

    @param tolerance: Convergence criterion
    @type tolerance: float

    @param relativeconv: Control relative vs. absolute convergence criterion.
    That is, if relativeconv is True, ||r||_2/||r_init||_2 is used as
    convergence criterion, else just ||r||_2 is used.  Actually ||r||_2^2 is
    used, since that save a few cycles.

    @type relativeconv: bool

    @return:  the solution x.

    DESCRIPTION:

    The method implements a left preconditioned Minimum residual method
    for symmetric indefinite linear systems.  The preconditioning
    operator has to be symmetric and positive definite.

    The minimum residual method solves systems of the form BAx=Bb,
    where A is symmetric indefinite and B is symmetric positive
    definite.  The linear system is symmetric with respect to the inner
    product $ \((\cdot,\cdot)_B = (B^{-1}\cdot,\cdot)\).  The iterate
    x_k is determined by minimization of the norm of the residual $ \[
    \|B(b - A y)\|_B \] over the Krylov space $ \(span\{Bb, BABb,
    \ldots, (BA)^{k-1}Bb\}\).  $ Here the norm is defined by the inner
    product \((\cdot,\cdot)_B\).

    The default convergence monitor is $ \[ \rho_k = \|B(b - A x_k)\|_B
    = (B(b - A x_k), b - A x_k).\] $ The residual \(b - A x_k\) is not
    computed during the iteration, hence a direct computation of this
    quantity reqire an additional matrix vector product.  In the
    algorithm it is computed recursively.  Unfortunately this
    computations accumulates error and it may be necessary to compute
    the exact residual every update frequency iteration.



    """

    callback_converged = False

    residuals = []

    rit = rit_
    d = A*x
    r = b - d
    s = B*r
    rho = inner(r,s)


    po = s.copy()
    qo = A*po

    p = r.copy()
    p *= 0.0
    q = p.copy()
    u = p.copy()

    if relativeconv:
        tolerance *= sqrt(rho)

    residuals.append(sqrt(rho))
    iter = 0
    #print "tolerance ", tolerance
    # Alloc w
    #while sqrt(inner(r,r)) > tolerance:# and iter<maxiter:
    while (sqrt(rho) > tolerance and not callback_converged) and iter < maxiter:
        #print "iter, sqrt(inner(r,r))", iter, sqrt(inner(r,r))
        uo    = B*qo
        gamma = sqrt(inner(qo,uo))
        gammai= 1.0/gamma

        # Do a swap
        tmp = po
        po  = p
        p   = tmp
        p  *= gammai

        tmp = qo
        qo  = q
        q   = tmp
        q  *= gammai

        tmp = uo
        uo  = u
        u   = tmp
        u  *= gammai

        alpha = inner(s,q)
        x    += alpha*p
        s    -= alpha*u

        if iter%rit==0:
            r = b - A*x
            rho = inner(r,s)
        else:
            rho -= alpha*alpha
        rho = fabs(rho)

        t     = A*u
        alpha = inner(t,u)
        beta  = inner(t,uo)

        po *= -beta
        po -= alpha*p
        po += u

        # update qo
        #po-=beta*po
        qo *= -beta
        qo -= alpha*q
        qo += t


        residuals.append(sqrt(rho))
        #print "sqrt(rho) ", sqrt(rho)

        # Call user provided callback with solution
        if callable(callback):
            callback_converged = callback(k=iter, x=x, r=sqrt(rho)) #r=r)

        iter += 1
        #print "r",iter,"=",sqrt(inner(r,r))
    #print "precondMinRes finished, iter: %d, ||e||=%e" % (iter,sqrt(inner(r,r)))
    return x,residuals, [], []

