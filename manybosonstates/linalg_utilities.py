"""A numbe of utility functions to easy some linear algebra tasks."""
import functools
import operator
import numpy as np
import scipy
import scipy.misc


class DimensionalityError(Exception):
    pass


def prod(iterable):
    """Return product of inputs."""
    return functools.reduce(operator.mul, iterable, 1)


def abs2(arr):
    """Compute squared modulus of numpy arrays elementwise."""
    return np.abs(arr) ** 2


def fact(n):
    """Return factorial of `n`, as an integer."""
    return scipy.misc.factorial(n, exact=True)


def binom(m, n):
    """Return binomial factor between `m` and `n`, as an integer."""
    return scipy.special.binom(m, n).astype(int)


def take_largest_elements(arr, n):
    """Return the `n` largest elements of `arr`.

    This function assumes that `arr` is a real array.

    Examples
    --------
    >>> take_largest_elements(np.arange(10), 2)
    array([8, 9])
    """
    if n <= 0:
        raise ValueError('`n` must be a positive integer '
                         '(it currently is {})'.format(str(n)))
    max_indices = arr.argpartition(-n)[-n:]
    return arr[np.sort(max_indices)]


def random_normalized_array(dim, complex=True):
    """Generate a numpy random normalized array."""
    out = np.zeros(dim, dtype=np.complex128)
    out = np.random.randn(dim) + 1j * np.random.randn(dim)
    return out / np.linalg.norm(out)


def normalize(vector):
    """Return normalized vector."""
    return vector / np.linalg.norm(vector)


def orthogonalize(vec1, vec2):
    """Make `vec2` orthogonal to `vec1` and return it."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    normalized_vec1 = normalize(vec1)
    overlap_factor = np.vdot(normalized_vec1, vec2) * normalized_vec1
    return (vec1, vec2 - overlap_factor)


def orthonormalize(vec1, vec2):
    """Make `vec2` orthogonal to `vec1`, and normalize both."""
    v1, v2 = orthogonalize(vec1, vec2)
    return normalize(v1), normalize(v2)


def effective_dim(input_array):
    """Return the product of the two dimensions of the array."""
    if input_array.ndim != 2 and input_array.ndim != 1:
        raise DimensionalityError('The array must have 2 dimensions.')
    dims = input_array.shape
    if len(dims) == 2:
        return dims[0] * dims[1]
    else:
        return dims[0]


def safe_dot(vec1, vec2):
    """Compute complex dot product between dense or sparse vectors."""
    if vec1.ndim == 1 and vec2.ndim == 1:
        out = np.vdot(vec1, vec2)
    elif vec1.ndim == 2 and vec2.ndim == 2:
        try:
            out = np.vdot(vec1, vec2)[0, 0]
        except ValueError:
            # if vdot didn't work, it may be because the vectors are both nx1
            # matrices. Let's then try to transpose the first one
            out = np.vdot(vec1.T, vec2)[0, 0]
    else:
        raise ValueError('Not sure how to multiply these vectors together.')
    return out


def sparse_array(length,
                 nonzero_elements,
                 nonzero_values=None,
                 array_type=None):
    """Return a 1d numpy array, equal to zero except for some indices."""
    if array_type is None:
        array_type = 'list'
    nonzero_elements = np.asarray(nonzero_elements)
    if nonzero_values is None:
        nonzero_values = np.ones(len(nonzero_elements))

    if array_type == 'list':
        out = np.zeros(length)
        out[nonzero_elements] = nonzero_values
        return out
    else:
        raise ValueError('`array_type` must have value \'list\'.')
