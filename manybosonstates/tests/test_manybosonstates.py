import unittest

import numpy as np
import qutip
import scipy

class TestManyBosonAmplitudes(unittest.TestCase):
    def test_many_boson_amplitudes_cf_fock_state(self):
        amps = mb._ManyBosonAmplitudes(
            list_of_single_boson_amps=[[0, 0, 1], [1, 0, 0]])
        self.assertEqual(amps.n_bosons, 2)
        self.assertEqual(amps.n_modes, 3)
        # check which='all', joined=False
        mb_amps = amps.get_many_boson_amplitudes(which='all', joined=False)
        self.assertEqual(set(mb_amps.keys()), set([(1, 1), (2,)]))
        np.testing.assert_allclose(
            mb_amps[(1, 1)], np.array([0, 1, 0]))
        np.testing.assert_allclose(
            mb_amps[(2,)], np.array([0, 0, 0]))
        # check which='all', joined=True
        mb_amps = amps.get_many_boson_amplitudes(which='all', joined=True)
        np.testing.assert_allclose(
            mb_amps, np.array([0, 1, 0, 0, 0, 0]))
    
    def test_many_boson_amplitudes_bunched_fock_state(self):
        amps = mb._ManyBosonAmplitudes(
            list_of_single_boson_amps=[[1, 0, 0], [1, 0, 0]])
        self.assertEqual(amps.n_bosons, 2)
        self.assertEqual(amps.n_modes, 3)
         # check which='all', joined=False
        mb_amps = amps.get_many_boson_amplitudes(which='all', joined=False)
        self.assertEqual(set(mb_amps.keys()), set([(1, 1), (2,)]))
        np.testing.assert_allclose(
            mb_amps[(1, 1)], np.array([0, 0, 0]))
        np.testing.assert_allclose(
            mb_amps[(2,)], np.array([1, 0, 0]))
        # check which='all', joined=True
        mb_amps = amps.get_many_boson_amplitudes(which='all', joined=True)
        np.testing.assert_allclose(
            mb_amps, np.array([0, 0, 0, 1, 0, 0]))


class TestManyBosonFockStates(unittest.TestCase):
    def test_create_and_evolve_from_bunched(self):
        mbfs = mb.ManyBosonFockState(2, mal=(0, 0))
        mbps = mbfs.evolve(qutip.hadamard_transform().full())
        amps = mbps.get_many_boson_amplitudes()
        self.assertEqual(list(amps.keys()), [(1, 1), (2,)])
        expected_output_11 = np.array([0.70710678 + 0.j])
        expected_output_2 = np.array([0.5 + 0.j, 0.5 + 0.j])
        np.testing.assert_allclose(amps[(1, 1)], expected_output_11)
        np.testing.assert_allclose(amps[(2,)], expected_output_2)

    def test_create_and_evolve_from_cf(self):
        mbfs = mb.ManyBosonFockState(2, mal=(0, 1))
        mbps = mbfs.evolve(qutip.hadamard_transform().full())
        amps = mbps.get_many_boson_amplitudes()
        self.assertEqual(list(amps.keys()), [(1, 1), (2,)])
        expected_output_11 = np.array([0])
        expected_output_2 = np.array([0.70710678 + 0.j, -0.70710678 + 0.j])
        np.testing.assert_allclose(amps[(1, 1)], expected_output_11)
        np.testing.assert_allclose(amps[(2,)], expected_output_2)

    def test_put_mol_get_mal(self):
        mbfs = mb.ManyBosonFockState(5, mal=(1, 2))
        mol = mbfs.get_mol()
        np.testing.assert_array_equal(mol, np.array([0, 1, 1, 0, 0]))
    
    def test_mol_mal_consistency(self):
        random_matrix = np.random.randn(4, 4)
        amps_mal = mb.ManyBosonFockState(4, mal=(0, 1)).evolve(
            random_matrix).get_many_boson_amplitudes(joined=True)
        amps_mol = mb.ManyBosonFockState(4, mol=(1, 1, 0, 0)).evolve(
            random_matrix).get_many_boson_amplitudes(joined=True)
        np.testing.assert_allclose(amps_mal, amps_mol)


class TestManyBosonStatesSuperposition(unittest.TestCase):
    def test_from_sum_of_Focks(self):
        mbfs = mb.ManyBosonFockState(2, mal=(0, 0))
        other_mbfs = mb.ManyBosonFockState(2, mal=(0, 1))
        state = 2 * mbfs + other_mbfs
        state.normalize_coefficients()
        # check normalization
        np.testing.assert_allclose(
            state.get_coefficients(),
            np.array([0.89442719, 0.4472136])
        )

    def test_normalize_Fock_superpositons(self):
        matrix = qutip.hadamard_transform(N=1).full()
        state = (mb.ManyBosonFockState(2, mal=(0, 0)) -
                 mb.ManyBosonFockState(2, mal=(1, 1)))
        state.evolve(matrix)
        state.normalize_coefficients()
        np.testing.assert_allclose(
            state.get_coefficients(),
            [0.7071067811865475, -0.7071067811865475]
        )


if __name__ == '__main__':
    import manybosonstates.manybosonstates as mb
    unittest.main(failfast=True)
