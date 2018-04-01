from ufl.corealg.map_dag import MultiFunction
import collections
SplitForm = collections.namedtuple("SplitForm", ["indices", "form"])

class FormSplitter(MultiFunction):

    """Split a form in a list of subtrees for each component of the
    mixed space it is built on.  See :meth:`split` for a usage
    description."""

    def split(self, form):
        """Split the form.

        :arg form: the form to split.

        This is a no-op if none of the arguments in the form are
        defined on :class:`~.MixedFunctionSpace`\s.

        The return-value is a tuple for which each entry is.

        .. code-block:: python

           (argument_indices, form)

        Where ``argument_indices`` is a tuple indicating which part of
        the mixed space the form belongs to, it has length equal to
        the number of arguments in the form.  Hence functionals have
        a 0-tuple, 1-forms have a 1-tuple and 2-forms a 2-tuple
        of indices.

        For example, consider the following code:

        .. code-block:: python

            V = FunctionSpace(m, 'CG', 1)
            W = V*V*V
            u, v, w = TrialFunctions(W)
            p, q, r = TestFunctions(W)
            a = q*u*dx + p*w*dx

        Then splitting the form returns a tuple of two forms.

        .. code-block:: python

           ((0, 2), w*p*dx),
            (1, 0), q*u*dx))

        """
        from ufl.algorithms.map_integrands import map_integrand_dags
        from numpy import ndindex
        args = form.arguments()
        if all(a.function_space().num_sub_spaces() == 0 for a in args):
            # No mixed spaces, just return the form directly.
            idx = tuple([0]*len(form.arguments()))
            return (SplitForm(indices=idx, form=form), )
        forms = []
        # How many subspaces do we have for each argument?
        shape = tuple(max(1, a.function_space().num_sub_spaces()) for a in args)
        # Walk over all the indices of the spaces
        for idx in ndindex(shape):
            # Which subspace are we currently interested in?
            self.idx = dict(enumerate(idx))
            # Cache for the arguments we construct
            self._args = {}
            # Visit the form
            f = map_integrand_dags(self, form)
            # Zero-simplification may result in an empty form, only
            # collect those that are non-zero.
            if len(f.integrals()) > 0:
                forms.append(SplitForm(indices=idx, form=f))
        return tuple(forms)

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
       return o

    def argument(self, o):
        from ufl import as_vector
        from ufl.constantvalue import Zero
        from dolfin import Argument
        from numpy import ndindex
        V = o.function_space()
        if V.num_sub_spaces() == 0:
            # Not on a mixed space, just return ourselves.
            return o
        # Already seen this argument, return the cached version.
        if o in self._args:
            return self._args[o]
        args = []
        for i, V_i_sub in enumerate(V.split()):
            # Walk over the subspaces and build a vector that is zero
            # where the argument does not match the one we're looking
            # for and is just the non-mixed argument when we do want
            # it.
            V_i = V_i_sub.collapse()
            a = Argument(V_i, o.number(), part=o.part())
            indices = ndindex(a.ufl_shape)
            if self.idx[o.number()] == i:
                args += [a[j] for j in indices]
            else:
                args += [Zero() for j in indices]
        self._args[o] = as_vector(args)
        return self._args[o]

def split_form(form):
    # Takes a mixed space form and returns a numpy array of forms for use with cbc.block
    # Array entries are the forms on each collapsed subspace of the mixed space.
    from numpy import zeros

    args = form.arguments()
    assert len(args) <= 2

    shape = tuple(max(1, a.function_space().num_sub_spaces()) for a in args)

    forms = zeros(shape, dtype = object)

    for ij, form_ij in FormSplitter().split(form):
        forms[ij] = form_ij

    return forms


def _collapse_bc(bc):
    from dolfin import DirichletBC
    sub_space = bc.function_space()

    assert len(sub_space.component()) == 1

    sub_id = sub_space.component()[0]
    sub_bc = DirichletBC(sub_space.collapse(), bc.value(), *bc.domain_args)
    return (int(sub_id), sub_bc)    


def split_bcs(bcs, m):
    # return a list of lists of DirichletBC for use with cbc.block
    # we need the number of blocks m to ensure correct length of output list
    collapsed_bcs = [[] for i in xrange(m)]
    
    for i, bc_i in map(_collapse_bc, bcs):
        collapsed_bcs[i].append(bc_i)

    return collapsed_bcs

