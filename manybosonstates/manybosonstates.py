"""Define the ManyBosonFockState class."""
import copy
import itertools
import numbers
import warnings
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse

from .linalg_utilities import DimensionalityError, binom, fact, safe_dot
from .many_boson_states_utilities import (_mal_to_mol, _mol_to_mal,
                                          symmetrise_arrays)


def _num_states_partition_class(n_modes, partition_class):
    """Compute the number of states in a partition class.

    Takes an element of the output of `list_partition_classes` and
    returns the number of states corresponding to it.
    """
    # extract the number of bosons corresponding to the partition class
    n_bosons = sum(partition_class)
    # count the classes of occupation numbers
    counts = [len(list(g)) for _, g in itertools.groupby(partition_class)]
    # compute the product of the factorial of the elements of `counts`,
    # the multiply with the factorial of the number of remaining modes,
    # and return the factorial of the total number of modes divided by
    # the above number. This is a generalized binomial coefficient.
    mu_factor = np.prod(fact(counts)).astype(np.int64)
    mu_factor *= fact(n_modes - n_bosons)
    return fact(n_modes) // mu_factor


class _ManyBosonAmplitudes:
    def __init__(self, n_modes=None, n_bosons=None,
                 list_of_single_boson_amps=None, many_boson_amps=None):
        """
        If provided with `list_of_single_boson_amps`, store each single-boson
        amplitude separately. Do not compute the many-boson amplitudes unless
        explicitly asked to.

        Parameters
        ----------
        n_modes : int, optional
        n_bosons : int, optional
        list_of_single_boson_amps : 2D array
            If given, specifies many-boson amplitudes for product states,
            through the lists of single-boson amplitudes.
        many_boson_amps : array or dict
            Full list of many-boson amplitudes.
        """
        logging.debug('Instantiating _ManyBosonAmplitudes object')
        # preflight checks
        if (list_of_single_boson_amps is not None
                and many_boson_amps is not None):
            warnings.warn('Both `list_of_single_boson_amps` and `many_boson_'
                          'amps` have been given. The latter is going to over'
                          'ride the former.')
        # initialize class attributes
        self.n_bosons = 0
        self.n_modes = None
        self._list_of_amps = None
        self._many_boson_amps_dict = None
        self._many_boson_amps_joined = None
        self._sorting_method = 'cf_first'
        self.is_product_state = None
        self.coefficient = 1.
        # decide how to proceed based on the kind of input arguments
        if many_boson_amps is not None:
            if isinstance(many_boson_amps, dict):
                bunching_groups = many_boson_amps.keys()
                full_bunched_group = min(bunching_groups, key=len)
                self.n_bosons = full_bunched_group[0]
                self.n_modes = len(many_boson_amps[full_bunched_group])
                self._many_boson_amps_dict = many_boson_amps
            else:
                if n_modes is None or n_bosons is None:
                    raise ValueError('Cannot retrieve number of bosons or '
                                     'number of modes from a joined string of'
                                     ' amplitudes.')
                self.n_bosons = n_bosons
                self.n_modes = n_modes
                self._many_boson_amps_joined = many_boson_amps
        elif list_of_single_boson_amps is not None:
            list_1bamps = np.asarray(list_of_single_boson_amps)
            _n_modes = list_1bamps.shape[1]
            _n_bosons = list_1bamps.shape[0]
            # check consistency with other arguments
            if n_modes is not None and n_modes != _n_modes:
                    raise ValueError('Inconsistent number of modes.')
            if n_bosons is not None and n_bosons != _n_bosons:
                raise ValueError('Inconsistent number of bosons.')
            # everything seems to be alright, proeed
            self.n_modes = _n_modes
            self.n_bosons = _n_bosons
            self._list_of_amps = list_1bamps
            logging.debug('Added amplitudes: {}'.format(self._list_of_amps))

    def __add__(self, other):
        """Implement coherent addition of different sets of many-boson amps.
        """
        self_amps = self._retrieve_mbamps_dict()
        other_amps = other._retrieve_mbamps_dict()
        final_amps = {}
        for key in self_amps.keys():
            final_amps[key] = (self.coefficient * self_amps[key] +
                               other.coefficient * other_amps[key])
        return _ManyBosonAmplitudes(
            n_modes=self.n_modes, n_bosons=self.n_bosons,
            many_boson_amps=final_amps
        )

    def __mul__(self, other):
        """Implement multiplication with scalar."""
        if not isinstance(other, numbers.Number):
            raise ValueError('Only multiplication with numbers is supported.')
        self.coefficient *= other

    def _compute_mbamps_dict(self):
        # We shouldn't need to consider the scalar coefficient at this point,
        # or risk to mess the normalization.
        if self._list_of_amps is None:
            return
        self._many_boson_amps_dict = symmetrise_arrays(*self._list_of_amps)
        # if self.coefficient != 1:
        #     for key in self._many_boson_amps_dict.keys():
        #         self._many_boson_amps_dict[key] *= self.coefficient

    def _retrieve_mbamps_dict(self, update=True):
        if update or self._many_boson_amps_dict is None:
            self._compute_mbamps_dict()
        return self._many_boson_amps_dict

    def _compute_mbamps_joined(self):
        # the big vector of many-boson amplitudes is computed from the
        # dictionary-based one. Note that this will also recompute the amps
        # in dictionary form
        self._compute_mbamps_dict()
        mbamps_dict = self._retrieve_mbamps_dict()
        # convert dict to joined
        if self._sorting_method == 'cf_first':
            self._many_boson_amps_joined = np.concatenate(
                tuple(amps for _, amps in mbamps_dict.items())
            )
        else:
            raise ValueError('TBD')

    def _retrieve_mbamps_joined(self, update=True):
        if update or self._many_boson_amps_joined is None:
            self._compute_mbamps_joined()
        return self._many_boson_amps_joined

    def add_amplitude(self, new_amplitude):
        logging.debug('Adding amplitude to _ManyBosonAmplitudes')
        self.n_bosons += 1
        new_amplitude = np.asarray(new_amplitude)[None, :]
        if self._list_of_amps is None:
            logging.debug('')
            self._list_of_amps = new_amplitude
        else:
            self._list_of_amps = np.concatenate(
                (self._list_of_amps, new_amplitude),
                axis=0
            )

    def get_many_boson_amplitudes(self, which='all', joined=False,
                                  update=True):
        if isinstance(which, str) and which == 'all':
            if joined:
                return self._retrieve_mbamps_joined(update=update)
            else:
                return self._retrieve_mbamps_dict(update=update)

class ManyBosonState(ABC):
    """Abstract class for many boson states."""
    def __mul__(self, coefficient):
        """Multiply with a scalar.

        Note that this will copy the object with all of its content, modify the
        copy, and return it. The original state will NOT be changed inplace.
        """
        if isinstance(coefficient, ManyBosonState):
            raise ValueError('To be done')
        else:
            new_state = copy.deepcopy(self)
            new_state.coefficient *= coefficient
            return new_state
    
    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1
    
    def __pos__(self):
        return self

    @abstractmethod
    def get_many_boson_amplitudes(self):
        pass

    def get_mbas(self, *args, **kwargs):
        return self.get_many_boson_amplitudes(*args, **kwargs)


class ManyBosonFockState(ManyBosonState):
    r"""A single many-boson Fock state (like a_1^\dagger a_2^\dagger)."""

    def __init__(self, n_modes=None, mal=None, mol=None):
        """Initialise necessary properties."""
        logging.debug('Instantiating ManyBosonFockState object')
        # initialize class attributes
        self.mol = None
        self.mal = None
        self.n_modes = None
        self.n_bosons = None
        self.coefficient = 1. # scalar added to the state
        # only one between `mol` and `mal` must be given
        if mal is not None and mol is not None:
            raise ValueError('Only one between `mol` and `mal` must'
                             'be used.')
        # if the state is defined using a mode occupation list, the value of
        # `n_modes` is ignored
        if mol is not None:
            # check that at least some boson has been specified
            if all(v == 0 for v in mol):
                raise ValueError('A MOL with all zeros is not a valid state.')
            if len(mol) != n_modes:
                raise ValueError('The length of `mol` must coincide with the '
                                 'number of modes in the state.')
            self.mol = mol
            self.n_modes = len(mol)
            self.n_bosons = sum(mol)
        # if instead a mode assignment list is given through `mal`, the value
        # of `n_modes`, if not None, is used to determine the total number of
        # modes in the state
        if mal is not None:
            if n_modes is not None and n_modes < max(mal):
                raise ValueError('The number of modes must be larger than the'
                                 ' mode numbers given with `mal`.')
            # the value of `self.n_modes` is taken from `n_modes` if given, or
            # assumed to be the higher mode in the given `mal` otherwise
            if n_modes is not None:
                self.n_modes = n_modes
            else:
                self.n_modes = max(mal)
            self.mal = tuple(sorted(mal))
            self.n_bosons = len(mal)

    def __add__(self, other):
        logging.debug('Trying to add {} to {}'.format(self, other))
        if isinstance(other, ManyBosonFockState):
            return ManyBosonStatesSuperposition._from_sum_of_Focks(self, other)
        else:
            raise ValueError('Not implemented yet.')

    def _get_mol_string(self):
        mol = self.get_mol()
        string = '|'
        for n in mol:
            string += str(n) + ','
        string = string[:-1] + ' >'
        return string

    def __repr__(self):
        string = 'Fock state with {} modes:\n  '.format(self.n_modes)
        if self.coefficient != 1:
            string += str(self.coefficient) + ' * '
        string += self._get_mol_string()
        return string


    def get_mol(self):
        """Get mode occupation list representing the state."""
        if self.mol is None:
            # if no mol was set, we obtain it's value from the saved mal
            self.mol = _mal_to_mol(self.mal, n_modes=self.n_modes)
        return self.mol

    def get_mal(self):
        """Get mode assignment list representing the state."""
        if self.mal is None:
            # generate the mal if not already done
            self.mal = _mol_to_mal(self.mol)
        return self.mal

    def get_mbs_amplitude(self):
        """Get the corresponding amplitude in the many-boson space."""
        amp = scipy.sparse.dok_matrix(
            (binom(self.n_modes, 2), 1), dtype=np.complex)
        amp[self._get_index_in_mbs()] = 1
        return amp

    def _get_index_in_mbs(self, order='mixed_order'):
        """Get index in the many-boson state corresponding to this mol.

        Returns the index corresponding to the current many-boson state in the
        full many-boson space with same number of bosons and modes.
        The order of the states in this space is assumed to be the same as that
        generated by `set(itertools.permutations(...)`.

        WARNING: the convention used here is different than the one used in
                 `ManyBosonProductState`, `symmetrise_arrays` etc.
        """
        if order == 'mixed_order':
            # With this ordering the many boson states are ordered according
            # to the following listing of mals:
            # (0, 0, 0), (0, 0, 1), (0, 0, 2), ..., (0, 0, m - 1),
            # (0, 1, 0), (0, 1, 1), (0, 1, 2), ..., (0, 1, m - 1), ...
            # (m - 1, m - 1, 0), (m - 1, m - 1, 1), ..., (m - 1, m - 1, m - 1),
            # where `m` is the number of modes, and the number of bosons is
            # here assumed to be 3.
            mal = self.get_mal()
            mbs_index = np.int64(0)
            prev_mode = 0
            for boson_index, mode_idx in enumerate(mal):
                for virtual_mode_idx in range(prev_mode, mode_idx):
                    mbs_index += scipy.special.binom(
                        (self.n_modes - virtual_mode_idx + self.n_bosons -
                         boson_index - 2), self.n_bosons - boson_index - 1)
                prev_mode = mode_idx
            return mbs_index.astype(int)
        elif order == 'cf_first':
            # With this ordering the many boson states are ordered as
            # dictated by the output of `_list_partition_classes`,
            # that is, all collision-free states are listed first, and
            # then progressively the more bunched states follow.

            # p_classes is the list of partition classes describing the
            # possible types of bunching
            p_classes = _list_partition_classes(self.n_bosons)
            # convert state mol to a partition class-like format
            mol = tuple(self.get_mol())
            stripped_mol = tuple(
                sorted([r for r in mol if r != 0], reverse=True))
            # see to which of the possible bunching class does the mol
            # correspond to
            actual_pclass = p_classes.index(stripped_mol)
            # set the counter to start count from the correct point
            counter = 0
            for prev_class_idx in range(actual_pclass):
                counter += _num_states_partition_class(
                    self.n_modes, p_classes[prev_class_idx])
            # now find the index of the mol inside the list of states
            # in the correct bunching class
            # states_in_pclass = sorted(list(set(itertools.permutations(mol))),
            #                           reverse=True)
            # counter += states_in_pclass.index(mol)
            if self.n_bosons != 2:
                raise NotImplementedError('Only implemented for 2 photons')
            mal_from_1 = tuple(1 + r for r in self.get_mal())
            counter = (mal_from_1[0] - 1) * (self.n_modes - mal_from_1[0] / 2)
            counter += mal_from_1[1] - mal_from_1[0] - 1
            counter = int(counter)

            return counter

    def get_many_boson_amplitudes(self):
        return self.get_mol()

    def evolve(self, matrix):
        """Evolve the state with the given evolution matrix.

        The evolution of the states is handled by the ManyBosonProductState
        class. This function thus converts the `ManyBosonFockState` into a
        `ManyBosonProductState` and calls the `evolve` method from there.
        """
        logging.debug('Evolving ManyBosonFock state')
        if matrix.shape[0] != self.n_modes:
            raise DimensionalityError('Modes mismatch, {} != {}.'.format(
                matrix.shape[0], self.n_modes))
        # mal contains the modes to load into the ManyBosonProductState object
        mal = self.get_mal()
        # take every boson specified in `mal` and inject it separately
        # as a new excitation for a `ManyBosonProductState` object
        mbps = ManyBosonProductState(self.n_modes)
        mbps._coefficient = self.coefficient
        # TODO: we could be more efficient here by computing only once the
        #       output amplitudes per input populated mode (that is, evoid to
        #       recompute output amplitudes for modes with more than one boson)
        logging.debug('Loading states into newly created ManyBosonProductState')
        for boson_idx in mal:
            amplitudes = _mal_to_mol((boson_idx,), n_modes=self.n_modes)
            logging.debug('Loading amplitude no {}'.format(boson_idx))
            mbps.add_excitation(amplitudes)
        # actually evolve the state using `matrix`
        logging.debug('Evolving ManyBosonProductState with given matrix')
        mbps.evolve(matrix, inplace=True)
        return mbps


class ManyBosonProductState(ManyBosonState):
    """Many boson product state object.

    This object is used to represent many-boson states that can be written as
    product of a number of creation operators (of given states).
    This representation can be more efficient for large number of modes, as
    it doesn't store the whole (exponentially large) many-boson state vector,
    but only the 1-body states of the single excitations.
    """

    def __init__(self, n_modes):
        """Initialize class variables."""
        logging.debug('Instantiating ManyBosonProductState object')
        self.n_modes = n_modes
        self.n_bosons = 0
        self.amplitudes = None  # list of amplitude vectors, 1 element per boson
        self._coefficient = 1
        self.many_boson_amplitudes = None

    def __repr__(self):
        string = ("Product state. Number of modes: {}. Number of photons: {}."
                  "\n".format(self.n_modes, len(self.amplitudes)))
        if self.many_boson_amplitudes.coefficient != 1:
            string += 'Coefficient: '
            string += str(self.many_boson_amplitudes.coefficient)
        string += '\nAmplitudes:\n' + str(self.amplitudes)
        return string
    
    def _repr_html_(self):
        string = '<div><p>'
        string += ("Product state. Number of modes: {}. Number of photons: {}."
                   .format(self.n_modes, len(self.amplitudes)))
        if self.many_boson_amplitudes.coefficient != 1:
            string += '<br><b>Coefficient:</b> {}'.format(
                self.many_boson_amplitudes.coefficient)
        string += '<br><b>Amplitudes:</b></p></div>' 
        pd.options.display.float_format = '{:,.2f}'.format
        df = pd.DataFrame(self.amplitudes)
        df.index = ['State ' + str(idx) for idx in range(df.shape[0])]
        return string + df._repr_html_()

    @property
    def coefficient(self):
        return self.many_boson_amplitudes.coefficient

    @coefficient.setter
    def coefficient(self, value):
        self.many_boson_amplitudes.coefficient = value

    def add_excitation(self, new_amplitude):
        """Add an excitation to the many boson state. Acts inplace.

        The new amplitude is renormalized and then stored into
        `self.amplitudes`, and the `self.n_bosons` counter is updated.
        Nothing else is done by this function (renormalizations due to
        bunching are applied later when the many boson amplitudes are
        requested).
        """
        logging.debug('Adding excitation to ManyBosonProductState')
        # check consistency of number of modes of added amplitude
        if new_amplitude.shape[0] != self.n_modes:
            raise ValueError('Incorrect number of modes: {} != {}.'.format(
                new_amplitude.shape[0], self.n_modes))
        # actually add new amplitude
        self.n_bosons += 1
        # normalize new amplitude
        new_amplitude = np.asarray(new_amplitude, dtype=np.complex)
        new_amplitude /= np.linalg.norm(new_amplitude)
        # update many-boson amplitudes state
        if self.many_boson_amplitudes is None:
            self.many_boson_amplitudes = _ManyBosonAmplitudes(
                list_of_single_boson_amps=new_amplitude[None, :])
            self.amplitudes = self.many_boson_amplitudes._list_of_amps
            self.many_boson_amplitudes.coefficient = self._coefficient
            del self._coefficient
        else:
            self.many_boson_amplitudes.add_amplitude(new_amplitude)
            self.amplitudes = self.many_boson_amplitudes._list_of_amps
        logging.debug('New amplitudes: {}'.format(self.amplitudes))
        # the following only makes sense for 2 bosons (not sure for more):
        # store normalization factor if the newly added amplitude is not
        # orthogonal
        # self.norm = np.abs(np.vdot(*self.amplitudes)) ^ 2
        # self.norm = np.sqrt(1 + self.norm)
        return None

    def remove_excitation(self, index_excitation):
        """Remove a specific interaction. Acts inplace."""
        if self.n_bosons == 0:
            raise ValueError('There are no excitations left to remove.')

        del self.amplitudes[index_excitation]
        self.n_bosons -= 1
        return None

    def get_many_boson_amplitudes(self, which='all', joined=False,
                                  update=True):
        """Compute and return many boson amplitudes.

        Parameters
        ----------
        joined : bool
            If True, the many-boson amplitudes are returned as a single
            1d numpy array. Otherwise, the output is a dictionary with
            keys the various bunching classes and values the amplitudes
            corresponding to that bunching class.
        update : bool
            If True, the many-boson amplitudes are recomputed.
        """
        if self.many_boson_amplitudes is None:
            raise ValueError('Amplitudes not computed, for some reason.')
        return self.many_boson_amplitudes.get_many_boson_amplitudes(
            which=which, joined=joined, update=update)

    def evolve(self, matrix, inplace=False):
        """Evolve the many-boson state using the given evolution matrix."""
        if inplace:
            self_ = self
        else:
            self_ = copy.deepcopy(self)
        # do the stuff
        for boson_index in range(self_.n_bosons):
            self_.amplitudes[boson_index] = matrix.dot(
                self_.amplitudes[boson_index])
        # return result
        if inplace:
            return None
        else:
            return self_


class ManyBosonStatesSuperposition(ManyBosonState):
    """Represent sum of product of many-boson states."""
    def __init__(self, data, repr=None):
        if repr is None:
            raise ValueError('Not implemented yet.')

        self.repr = repr
        self.data = data

        self.n_modes = None
        self.many_boson_amplitudes = None

        if self.repr == 'list of Focks':
            self.n_modes = self.data[0].n_modes
        
    @classmethod
    def _from_sum_of_Focks(cls, first_state, second_state):
        # check dimensions and stuff
        if first_state.n_modes != second_state.n_modes:
            raise ValueError('The number of modes must be the same.')
        # if all is right, instantiate
        return cls(
            repr='list of Focks',
            data=[first_state, second_state]
        )
    
    def __repr__(self):
        if self.repr == 'list of Focks':
            string = 'Superposition of Fock states.\n  '
            for fock_state in self.data:
                if fock_state.coefficient != 1:
                    string += str(fock_state.coefficient) + ' * '
                string += fock_state._get_mol_string()
                string += ' + '
            return string[:-2]
        elif self.repr == 'list of Products':
            string = 'Superposition of product states.'
            for idx, state in enumerate(self.data):
                string += '\n----------------\n'
                string += '## State {}:\n'.format(idx)
                string += '{}'.format(state.__repr__())
            string += '\n----------------\n'
            return string

    def normalize_coefficients(self):
        """Normalize coefficients inplace."""
        coefficients = np.array([s.coefficient for s in self.data])
        norm = scipy.linalg.norm(coefficients)
        for state in self.data:
            state.coefficient /= norm
        return None

    def get_coefficients(self):
        """Retrieve scalar coefficients of components."""
        return np.array([s.coefficient for s in self.data])

    def _evolve_list_of_Focks(self, matrix, inplace):
        """Evolution occurs not inplace by default."""
        self.repr = 'list of Products'
        newdata = []
        for fock_state in self.data:
            newdata.append(fock_state.evolve(matrix))
        # renormalize coefficients if needed
        coefficients = np.array([s.coefficient for s in newdata])
        norm = scipy.linalg.norm(coefficients)
        if not np.allclose(norm, 1):
            logging.debug('Renormalizing state.')
            for prod_state in newdata:
                prod_state.many_boson_amplitudes.coefficient /= norm
        if inplace:
            self.data = newdata
        else:
            newself = copy.deepcopy(self)
            newself.data = newdata
            return newself

    def evolve(self, matrix, inplace=False):
        """Evolve the superposition of many-boson states."""
        # consistency checks
        if matrix.shape[0] != self.n_modes:
            raise ValueError('Modes mismatch: {} != {}'.format(
                matrix.shape[0], self.n_modes
            ))
        # how the system is evolved depends on the internal representation
        if self.repr == 'list of Focks':
            out = self._evolve_list_of_Focks(matrix, inplace=inplace)
            if inplace:
                return None
            else:
                return out
    
    def get_many_boson_amplitudes(self, **kwargs):
        if self.repr == 'list of Products':
            final_amps = None  # many-boson amplitudes to be returned
            for state in self.data:  # state.__class__ is _ManyBosonAmplitude
                if final_amps is None:
                    final_amps = state.many_boson_amplitudes
                    continue
                final_amps = final_amps + state.many_boson_amplitudes
            self.many_boson_amplitudes = final_amps
        else:
            raise ValueError('TBD')
        return self.many_boson_amplitudes.get_many_boson_amplitudes(**kwargs)
