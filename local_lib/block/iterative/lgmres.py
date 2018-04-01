from __future__ import division
from common import *
import numpy

#####
# Adapted from SciPy (BSD license)
# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
#####

def lgmres(B, A, x, b, tolerance, maxiter, progress, relativeconv=False,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True, callback=None):
    """
    Solve a matrix equation using the LGMRES algorithm.

    The LGMRES algorithm [BJM]_ [BPh]_ is designed to avoid some problems
    in the convergence in restarted GMRES, and often converges in fewer
    iterations.

    Additional parameters
    ---------------------
    inner_m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    outer_k : int, optional
        Number of vectors to carry between inner GMRES iterations.
        According to [BJM]_, good values are in the range of 1...3.
        However, note that if you want to use the additional vectors to
        accelerate solving multiple similar problems, larger values may
        be beneficial.
    outer_v : list of tuples, optional
        List containing tuples ``(v, Av)`` of vectors and corresponding
        matrix-vector products, used to augment the Krylov subspace, and
        carried between inner GMRES iterations. The element ``Av`` can
        be `None` if the matrix-vector product should be re-evaluated.
        This parameter is modified in-place by `lgmres`, and can be used
        to pass "guess" vectors in and out of the algorithm when solving
        similar problems.
    store_outer_Av : bool, optional
        Whether LGMRES should store also A*v in addition to vectors `v`
        in the `outer_v` list. Default is True.

    Notes
    -----
    The LGMRES algorithm [BJM]_ [BPh]_ is designed to avoid the
    slowing of convergence in restarted GMRES, due to alternating
    residual vectors. Typically, it often outperforms GMRES(m) of
    comparable memory requirements by some measure, or at least is not
    much worse.

    Another advantage in this algorithm is that you can supply it with
    'guess' vectors in the `outer_v` argument that augment the Krylov
    subspace. If the solution lies close to the span of these vectors,
    the algorithm converges faster. This can be useful if several very
    similar matrices need to be inverted one after another, such as in
    Newton-Krylov iteration where the Jacobian matrix often changes
    little in the nonlinear steps.

    References
    ----------
    .. [BJM] A.H. Baker and E.R. Jessup and T. Manteuffel,
             SIAM J. Matrix Anal. Appl. 26, 962 (2005).
    .. [BPh] A.H. Baker, PhD thesis, University of Colorado (2003).
             http://amath.colorado.edu/activities/thesis/allisonb/Thesis.ps

    """
    import sys
    from scipy.linalg.basic import lstsq

    if outer_v is None:
        outer_v = []

    r_outer = A*x - b
    r_norm = norm(r_outer)

    if relativeconv:
        tolerance *= r_norm
    residuals = [r_norm]

    for k_outer in xrange(maxiter):
        progress += 1

        # -- check stopping condition
        if r_norm < tolerance:
            break

        # -- inner LGMRES iteration
        vs0 = -B*r_outer
        inner_res_0 = norm(vs0)

        if inner_res_0 == 0:
            rnorm = norm(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        vs0 *= 1.0/inner_res_0
        hs = []
        vs = [vs0]
        ws = []
        y = None

        for j in xrange(1, 1 + inner_m + len(outer_v)):
            # -- Arnoldi process:
            #
            #    Build an orthonormal basis V and matrices W and H such that
            #        A W = V H
            #    Columns of W, V, and H are stored in `ws`, `vs` and `hs`.
            #
            #    The first column of V is always the residual vector, `vs0`;
            #    V has *one more column* than the other of the three matrices.
            #
            #    The other columns in V are built by feeding in, one
            #    by one, some vectors `z` and orthonormalizing them
            #    against the basis so far. The trick here is to
            #    feed in first some augmentation vectors, before
            #    starting to construct the Krylov basis on `v0`.
            #
            #    It was shown in [BJM]_ that a good choice (the LGMRES choice)
            #    for these augmentation vectors are the `dx` vectors obtained
            #    from a couple of the previous restart cycles.
            #
            #    Note especially that while `vs0` is always the first
            #    column in V, there is no reason why it should also be
            #    the first column in W. (In fact, below `vs0` comes in
            #    W only after the augmentation vectors.)
            #
            #    The rest of the algorithm then goes as in GMRES, one
            #    solves a minimization problem in the smaller subspace
            #    spanned by W (range) and V (image).
            #
            #    XXX: Below, I'm lazy and use `lstsq` to solve the
            #    small least squares problem. Performance-wise, this
            #    is in practice acceptable, but it could be nice to do
            #    it on the fly with Givens etc.
            #

            #     ++ evaluate
            v_new = None
            if j < len(outer_v) + 1:
                z, v_new = outer_v[j-1]
            elif j == len(outer_v) + 1:
                z = vs0
            else:
                z = vs[-1]

            if v_new is None:
                v_new = B*A*z
            else:
                # Note: v_new is modified in-place below. Must make a
                # copy to ensure that the outer_v vectors are not
                # clobbered.
                v_new = v_new.copy()

            #     ++ orthogonalize
            hcur = []
            for v in vs:
                alpha = inner(v, v_new)
                hcur.append(alpha)
                v_new -= alpha*v
            hcur.append(norm(v_new))

            if hcur[-1] == 0:
                # Exact solution found; bail out.
                # Zero basis vector (v_new) in the least-squares problem
                # does no harm, so we can just use the same code as usually;
                # it will give zero (inner) residual as a result.
                bailout = True
            else:
                bailout = False
                v_new *= 1.0/hcur[-1]

            vs.append(v_new)
            hs.append(hcur)
            ws.append(z)

            # XXX: Ugly: should implement the GMRES iteration properly,
            #      with Givens rotations and not using lstsq. Instead, we
            #      spare some work by solving the LSQ problem only every 5
            #      iterations.
            if not bailout and j % 5 != 1 and j < inner_m + len(outer_v) - 1:
                continue

            # -- GMRES optimization problem
            hess  = numpy.zeros((j+1, j))
            e1    = numpy.zeros((j+1,))
            e1[0] = inner_res_0
            for q in xrange(j):
                hess[:(q+2),q] = hs[q]

            y, resids, rank, s = lstsq(hess, e1)
            inner_res = numpy.dot(hess, y) - e1
            inner_res = sqrt(numpy.inner(inner_res, inner_res))

            # -- check for termination
            if inner_res < tolerance * inner_res_0:
                break

        # -- GMRES terminated: eval solution
        dx = ws[0]*y[0]
        for w, yc in zip(ws[1:], y[1:]):
            dx += w*yc

        # -- Store LGMRES augmentation vectors
        nxi = 1/norm(dx)
        if store_outer_Av:
            q = numpy.dot(hess, y)
            ax = vs[0]*q[0]
            for v, qc in zip(vs[1:], q[1:]):
                ax += v*qc
            outer_v.append((nxi*dx, nxi*ax))
        else:
            outer_v.append((nxi*dx, None))

        # -- Retain only a finite number of augmentation vectors
        outer_v = outer_v[-outer_k:]

        # -- Apply step
        x += dx
        r_outer = A*x - b
        r_norm = norm(r_outer)

        residuals.append(r_norm)

        # Call user provided callback with solution
        if callable(callback):
            callback(k=k_outer, x=x, r=r_norm)

    return x, residuals, [], []
