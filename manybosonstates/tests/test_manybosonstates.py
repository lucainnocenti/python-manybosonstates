import unittest

import numpy as np
import qutip
import scipy

class TestManyBosonFockStates(unittest.TestCase):
    def test_create_and_evolve(self):
        mbfs = mb.ManyBosonFockState(2, mal=(0, 0))
        mbps = mbfs.evolve(qutip.hadamard_transform().full())
        amps = mbps.get_many_boson_amplitudes()
        self.assertEqual(list(amps.keys()), [(1, 1), (2,)])
        expected_output = {
            (1, 1): np.array([0. + 0.j]),
            (2,): np.array([ 0.70710678 + 0.j, -0.70710678 + 0.j])
        }
        self.assertDictEqual(amps, expected_output)

if __name__ == '__main__':
    import manybosonstates as mb
    unittest.main(failfast=True)
