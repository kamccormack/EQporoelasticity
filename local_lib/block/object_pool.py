import sys

class object_pool(object):
    """Manage a free-list of objects. The objects are automatically made
    available as soon as they are deleted by the caller. The assumption is that
    any operation is repeated a number of times (think iterative solvers), so
    that if N objects are needed simultaneously then soon N objects are needed
    again. Thus, objects managed by this pool are not deleted until the owning
    object (typically a Matrix) is deleted.
    """
    def __init__(self):
        self.all = set()
        self.free = []

    def add(self, obj):
        self.all.add(obj)

    def get(self):
        self.collect()
        return self.free.pop()

    def collect(self):
        for obj in self.all:
            if sys.getrefcount(obj) == 3:
                # 3 references: self.all, obj, getrefcount() parameter
                self.free.append(obj)

def shared_vec_pool(func):
    """Decorator for create_vec, which creates a per-object pool of (memoized)
    returned vectors, shared for all dimensions. To be used only on objects
    where it is known that the row and columns are distributed equally.
    """
    def pooled_create_vec(self, dim=1):
        if not hasattr(self, '_vec_pool'):
            self._vec_pool = object_pool()
        try:
            vec = self._vec_pool.get()
        except IndexError:
            vec = func(self, dim)
            self._vec_pool.add(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec

def vec_pool(func):
    """Decorator for create_vec, which creates a per-object pool of (memoized)
    returned vectors per dimension.
    """
    from collections import defaultdict
    def pooled_create_vec(self, dim=1):
        if not hasattr(self, '_vec_pool'):
            self._vec_pool = defaultdict(object_pool)
        try:
            vec = self._vec_pool[dim].get()
        except IndexError:
            vec = func(self, dim)
            self._vec_pool[dim].add(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec

def store_args_ref(func):
    """Decorator for any function, which stores a reference to the arguments
    on the object. Used to force a Python-side reference, when the native-side
    reference isn't sufficient (or present)."""
    def store_args_and_pass(self, *args, **kwargs):
        self._vec_pool_args = (args, kwargs)
        return func(self, *args, **kwargs)
    store_args_and_pass.__doc__ = func.__doc__
    return store_args_and_pass
