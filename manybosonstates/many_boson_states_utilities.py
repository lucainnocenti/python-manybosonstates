import itertools
import numpy as np
import scipy
import scipy.sparse
import scipy.special

from .linalg_utilities import fact, binom, prod
from .misc_utilities import uneven_chunker


def list_cf_mols(n_modes, n_bosons):
    """Return list of collision-free mode occupation lists.

    To compute this we previously used
    ```
    base_list = [1] * n_bosons + [0] * (n_modes - n_bosons)
    return list(set(itertools.permutations(base_list)))
    ```
    The current method is however order of magnitudes more efficient
    (as it doesn't require to first compute a huge amount of permutations
    that will later be removed by `set`).
    """
    return [
        _mal_to_mol(mal, n_modes) for mal in list_cf_mals(n_modes, n_bosons)
    ]


def list_cf_mals(n_modes, n_bosons):
    """Return as an iterator all collision-free mode assignment lists.

    Examples
    --------
    >>> list(list_cf_mals(3, 2))
    [(0, 1), (0, 2), (1, 2)]
    """
    return itertools.combinations(range(n_modes), n_bosons)


def list_bunched_mals(modes, bosons):
    """Return mode assignment lists corresponding to bunched states."""
    if bosons != 2:
        raise ValueError('Only 2 bosons are currently supported.')
    return list(zip(range(modes), range(modes)))


def list_all_mals(n_modes, n_bosons, order=None):
    """Return all mode assignment lists.

    This is still pretty inefficient for many modes and more then two
    bosons. Handling the processing of the big lists of tuples in numpy
    may help significantly.
    """
    all_mals = itertools.combinations_with_replacement(
        range(n_modes), n_bosons)
    if order is None:
        return all_mals
    elif order == 'bclass':
        return sorted(all_mals, key=_get_bclass)
    elif order == 'bclass_dict':
        # all mals in a single list, sorted by bunching class (the
        # collision-free states come first, following by the rest and
        # closed by the totally bunched ones)
        sorted_mals = sorted(all_mals, key=_get_bclass)
        # compute all possible bunching classes and the number of states
        # in each one
        bclasses = _list_bunching_classes(n_bosons)
        bclasses_sizes = [
            _number_states_in_pclass(n_modes, bclass) for bclass in bclasses
        ]
        # convert the `sorted_mals` list into a list of lists, in which
        # every list contains all the states associated with a single
        # bunching class
        chunked_mals_list = [
            list(chunk)
            for chunk in uneven_chunker(sorted_mals, bclasses_sizes)
        ]
        # make the above list of lists into a nice dictionary
        return {
            bclass: chunk
            for bclass, chunk in zip(bclasses, chunked_mals_list)
        }


def _list_all_mals_np(n_modes, n_bosons, order=None):
    """Return all mode assignment lists.
    
    ---- WIP ----
    """
    all_mals = itertools.combinations_with_replacement(
        range(n_modes), n_bosons)
    all_mals = np.asarray(all_mals)
    if order is None:
        return all_mals
    elif order == 'bclass':
        return sorted(all_mals, key=_get_bclass)
    elif order == 'bclass_dict':
        # all mals in a single list, sorted by bunching class (the
        # collision-free states come first, following by the rest and
        # closed by the totally bunched ones)
        sorted_mals = sorted(all_mals, key=_get_bclass)
        # compute all possible bunching classes and the number of states
        # in each one
        bclasses = _list_bunching_classes(n_bosons)
        bclasses_sizes = [
            _number_states_in_pclass(n_modes, bclass) for bclass in bclasses
        ]
        # convert the `sorted_mals` list into a list of lists, in which
        # every list contains all the states associated with a single
        # bunching class
        chunked_mals_list = [
            list(chunk)
            for chunk in uneven_chunker(sorted_mals, bclasses_sizes)
        ]
        # make the above list of lists into a nice dictionary
        return {
            bclass: chunk
            for bclass, chunk in zip(bclasses, chunked_mals_list)
        }


def get_index_from_mal(target_mal, n_modes, n_bosons, order=None):
    """Return a specific index in the output of list_all_mals.

    Returns the index in the output of `list_all_mals(n_modes, n_bosons, order)`
    that correspond to the given `mal`.

    At the moment this function recreates the whole list by calling
    `list_all_mals` with the same parameters. This may not be very efficient
    for higher number of modes and bosons.

    Parameters
    ----------
    target_mal : tuple or dict
        If a `tuple`, the function returns the (supposedly only) index
        corresponding to that target_mal.
        If a `dict`, it must contain a 'with' key, denoting the set of
        modes of the requested mals.
        If the 'mode' key is equal to 'all', the returned indices correspond to
        the mals containing *all* of the requested modes. 
        If the 'mode' key is equal to 'any', the returned indices correspond
        to the mals containing *at least one* of the requested modes.
        If `target_mal` does not contain the 'mode' key, it defaults to 'all.
    """
    all_mals = list_all_mals(n_modes, n_bosons, order=order)
    # if `target_mal` is a tuple, return the (supposedly only) index
    # corresponding to a value equal to `mal`
    if isinstance(target_mal, tuple):
        if order is None or order == 'bclass':
            return next(idx for idx, mal in enumerate(all_mals)
                        if mal == target_mal)
        elif order == 'bclass_dict':
            bclass = _get_bclass(target_mal)
            return next(idx for idx, mal in enumerate(all_mals[bclass])
                        if mal == target_mal)
    # if `target_mal` is a dictionary, we value of the entry 'with' says
    # to look for all the mals containing some specified modes
    elif isinstance(target_mal, dict):
        # check that the 'with' key has been given
        try:
            requested_modes = target_mal['with']
        except KeyError:
            raise ValueError('target_mal must contain the key \'with\'.')
        # convert `requested_modes` into an iterable if it's not already
        try:
            iter(requested_modes)
        except TypeError:
            requested_modes = [requested_modes]
        # read the 'mode' key if given, otherwise default to 'all'
        try:
            search_mode = target_mal['mode']
        except KeyError:
            search_mode = 'all'
        # define helper function to use in list comprehension
        def satisfies_requests(mal):
            if search_mode == 'all':
                return all(req_mode in mal for req_mode in requested_modes)
            elif search_mode == 'any':
                return any(req_mode in mal for req_mode in requested_modes)
            else:
                raise ValueError('Incorrect value of \'search_mode\'.')

        # the way the indices are recovered depends on the value of the
        # `order` parameter.
        if order is None or order == 'bclass':
            return [
                idx for idx, mal in enumerate(all_mals)
                if satisfies_requests(mal)
            ]
        elif order == 'bclass_dict':
            bclass = _get_bclass(target_mal)
            return [
                idx for idx, mal in enumerate(all_mals[bclass])
                if satisfies_requests(mal)
            ]


def list_mols_in_bclass(n_modes, bunching_class):
    """Return the mols corresponding to the given bunching class.

    Bunching classes are the kinds of outputs given by
    `_list_bunching_classes`.
    """
    list_to_permute = list(bunching_class) + [0] * (
        n_modes - len(bunching_class))
    out_mols = list(set(itertools.permutations(list_to_permute)))
    return sorted(out_mols, reverse=True)


def list_mals_in_bclass(n_modes, bunching_class):
    """Return mals for a given bunching class.
    
    **NOTE**: This function still computes the *whole* set of mals, using
             `list_all_mals`, and only afterwards filters the requested
             ones. If all all the mals are needed, call that function
             instead, using `order='bclass_dict'` to request the mals
             grouped together by bunching class.
    """
    n_bosons = sum(bunching_class)
    all_mals_dict = list_all_mals(n_modes, n_bosons, order='bclass_dict')
    return all_mals_dict[bunching_class]


def _compute_mu_factor2(*input_mols):
    """Return product of factorials of the mols.

    Every element of `input_mols` is expected to be a tuples with the
    mode occupation list of a state.

    Examples
    --------
    >>> _compute_mu_factor((1, 1))
    1.0
    >>> _compute_mu_factor((1, 1, 0), (2, 0, 0))
    1.4142135623730951
    """
    mu_factor = 1
    for mol in input_mols:
        mu_factor *= np.prod(fact(mol))
    return mu_factor


def _compute_mu_factor(*input_mols):
    """Return square root of product of factorials of the mols."""
    return np.sqrt(_compute_mu_factor2(*input_mols))


def _compute_input_normalization(*amps):
    """Computes the normalization factor associated to bunched inputs."""
    if len(amps) < 2:
        raise ValueError('At least 2 amplitudes must be provided.')
    n_bosons = len(amps)
    left_range = range(n_bosons)
    right_ranges = list(itertools.permutations(left_range))
    total = 0.
    for right_range in right_ranges:
        i_prod = 1.
        for idx1, idx2 in zip(left_range, right_range):
            # if `idx1` and `idx2` are equal the contribution is given
            # by the inner product of an amplitude with itself. Given
            # that we are assuming the amplitudes to be normalized,
            # the result is always 1 and we can just skip it
            if idx1 == idx2:
                pass
            # otherwise we update the partial product computing the
            # inner product of the two relevant amplitudes (states)
            i_prod *= np.vdot(amps[idx1], amps[idx2])
        total += i_prod
    return np.sqrt(total)


def _mol_to_mal(mol):
    """Convert a mode occupation list to a mode assignment list.

    Examples
    --------
    >>> _mol_to_mal([2, 1, 1])
    array([0, 0, 1, 2])
    """
    mal = []
    for mode_number, mode_occupation in enumerate(mol):
        mal += [mode_number] * mode_occupation
    return tuple(sorted(mal))


def _mal_to_mol(mal, n_modes=None):
    """Convert a mode assignment list to a mode occupation list.

    Examples
    --------
    >>> _mal_to_mol((1, 2, 0, 0))
    array([2, 1, 1])
    """
    if n_modes is None:
        # the +1 is because the mode indices start from 0
        n_modes = np.max(mal) + 1
    mol = np.zeros(n_modes, dtype=int)
    for occupied_mode in mal:
        mol[occupied_mode] += 1
    return mol


def _list_bunching_classes(n_bosons, max_bosons=None):
    """Return the different \"bunching classes\".

    A *bunching class* is here meant as a descriptor of a set of states
    sharing the same bunching properties. It is effectively a sorted
    mode occupation list belonging to that given "bunching class".

    Examples
    --------
    >>> _list_bunching_classes(2)
    [(1, 1), (2,)]
    >>> _list_bunching_classes(4)
    [(1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)]
    """
    # `max_bosons` will be None usually when the function is invoked
    # from the outside (that is, not from the function itself through
    # iteration)
    if max_bosons is None:
        max_bosons = n_bosons
    if n_bosons == 0:
        return [()]
    if n_bosons == 1:
        return [(1, )]
    out_partitions = []
    for n0 in range(1, max_bosons + 1):
        partitions = _list_bunching_classes(n_bosons - n0, n0)
        out_partitions += [(n0, ) + partition for partition in partitions]

    return out_partitions


def _get_bclass(mal):
    """Return the bunching class to which the input `mal` belongs."""
    gen = (len(list(group)) for _, group in itertools.groupby(mal))
    return tuple(sorted(gen, reverse=True))


def _lengths_groupings(mol):
    """Return the list of lengths of equal occupation numbers in a mol.

    Examples
    --------
    >>> _lengths_groupings((1, 1, 1, 1))
    [4]
    >>> _lengths_groupings((2, 1, 1))
    [1, 2]
    >>> _lengths_groupings((2, 2))
    [2]
    """
    return [len(list(group)) for _, group in itertools.groupby(mol)]


def _number_states_in_pclass(n_modes, pclass):
    """Return the number of states in a given partition class.

    The input `pclass` should be one of the outputs of
    `_list_bunching_classes`.

    Examples
    --------
    >>> _number_states_in_pclass(4, (1, 1))
    6
    >>> _number_states_in_pclass(4, (2,)) 
    4
    """
    out = fact(n_modes) / fact(n_modes - len(pclass))
    out /= _compute_mu_factor2(_lengths_groupings(pclass))
    return out.astype(int)


def _partial_permanent(amps, mal):
    """Compute symmetrized product of given indices of the input amps.

    This is basically the same operation done to compute the permanent
    of a matrix, except it does not use a matrix as input, but rather
    a sequence of amplitudes.
    The second input `mal` specifies which elements of these amplitudes
    we are interested in, and the culprit of `_partial_permanent` is
    then to compute the symmetrized product of these elements.
    """
    partial_sum = 0.
    for mal_perm in itertools.permutations(mal):
        partial_sum += prod(amp[index] for amp, index in zip(amps, mal_perm))
    return partial_sum


def two_boson_amplitude(matrix, input_mal, output_mal):
    """Given two tuples, compute the permanent of the corresponding 2x2 submatrix.

    Parameters
    ----------
    input_state, output_state : tuple
        List of integers representing the mol of input and output states.

    Returns
    -------
    float
        The permanent of the 2x2 submatrix of `matrix` corresponding to the
        positions specified with `input_state` and `output_state`.

    Examples
    --------
    >>> import numpy as np
    >>> two_boson_amplitude(np.random.randn(3, 3), (1, 2), (2, 3))
    1.2  # or whatever other output number

    """
    sub = matrix[np.ix_(input_mal, output_mal)]
    out = sub[0, 0] * sub[1, 1] + sub[0, 1] * sub[1, 0]
    mu_factor = _compute_mu_factor(
        _mal_to_mol(input_mal), _mal_to_mol(output_mal))
    return out / mu_factor


def two_boson_matrix(matrix):
    """Compute the 2-boson evolution matrix corresponding to a given matrix."""
    # `dim` is the number of modes in the input matrix
    dim = len(matrix)
    # `bigm_dim` is the dimension of the 2-boson matrix
    bigm_dim = binom(dim + 1, 2)
    # initialize output 2-boson matrix
    out_matrix = np.zeros(shape=(bigm_dim, bigm_dim))
    # compute the list of mols of all possible 2-boson states (bunched and cf)
    all_states = list(list_all_mals(dim, 2))
    # iterate through all pairs of states to compute the actual 2-boson matrix
    for input_idx, input_pair in enumerate(all_states):
        for output_idx, output_pair in enumerate(all_states):
            out_matrix[output_idx, input_idx] = two_boson_amplitude(
                matrix, input_pair, output_pair)
    return out_matrix


def symmetrise_arrays(*amps):
    """Compute the symmetrized products of the subsets of the amplitudes.

    Each input amplitude should be a 1d vector.
    This function computes the probability amplitude for all output
    states, bunched and cf, when the state is written as product of a
    number of not necessarily orthogonal states.

    The order of the elements in the output arrays is determined by
    the function `list_all_mals`.

    Returns
    -------
    A dictionary whose keys are the possible bunching classes (that is,
    collision-free states, bunched states, and everything in between),
    and the value the corresponding amplitudes stored in numpy arrays.

    Examples
    --------
    >>> ket1 = np.array([1, 1]) / np.sqrt(2)
    >>> ket2 = np.array([1, -1]) / np.sqrt(2)
    >>> symmetrise_arrays(ket1, ket2)
    {(1, 1): array([ 0.+0.j]), (2,): array([ 0.70710678+0.j, -0.70710678+0.j])}
    """
    # some input checking
    if len(amps) < 2:
        raise ValueError('At least 2 amplitudes must be given.')
    amps = np.asarray(amps)
    for amp in amps:
        if amp.ndim != 1:
            raise ValueError('amps1 and amps2 must be 1d arrays (vectors).')
    # utility variables for the number of elements of input and output arrays
    n_modes = amps[0].shape[0]
    n_bosons = len(amps)
    # list all possible bunching classes
    bclasses = _list_bunching_classes(n_bosons)
    # compute a dict containing all the mals divided by bunching class
    all_mals_dict = list_all_mals(n_modes, n_bosons, order='bclass_dict')
    # initialize output set of amplitudes. To store both collision-free
    # and bunched amplitudes we use a dictionary, with each key
    # containing one class of amplitudes.
    new_amps = {}
    # compute sets of amplitudes for each bunching class
    for bclass in bclasses:
        # initialize the set of amplitudes for the current bunching class
        new_amps[bclass] = np.zeros(
            _number_states_in_pclass(n_modes, bclass), dtype=np.complex128)
        # extract all mals corresponding to the current bunching class
        mals = all_mals_dict[bclass]
        # for each such mal we compute the correspondig probability
        # amplitude of ending up in the corresponding state, when the
        # input state is the one encoded by `amps`. This require to
        # compute a sort of "partial permanent", which is handled by
        # `_partial_permanent`.
        for idx, mal in enumerate(mals):
            new_amps[bclass][idx] = _partial_permanent(amps, mal)
        # depending on the bunching class of the output encoded with
        # `mal`, we apply a normalization factor accordingly
        new_amps[bclass] /= _compute_mu_factor(
            _mal_to_mol(mals[0], n_modes=n_modes))
        # we also have to apply a normalization factor that depends
        # only on the input state (and in particular, on its bunched
        # nature)
        new_amps[bclass] /= _compute_input_normalization(*amps)
    return new_amps
