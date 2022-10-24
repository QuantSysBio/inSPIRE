""" Test suite for the inSPIRE mz_match utilities.
"""
import unittest

import numpy as np

from inspire.constants import C_TERMINUS, ION_OFFSET, N_TERMINUS, PROTON
from inspire.mz_match import (
    compute_potential_mws,
    get_ion_masses,
    match_mz,
)

TEST_PEPTIDE = 'ACDEFGHIKMN'
TEST_PTM_SEQ = '0.02000000010.0'
TEST_PTM_WEIGHTS = {
    0: 0.0,
    1: 15.994915,
    2: 57.021464,
}
SUMMED_MASSES_FORWARD = [
    71.037114,
    231.067763,
    346.094706,
    475.137299,
    622.205713,
    679.227177,
    816.286089,
    929.370153,
    1057.465116,
    1204.500516,
]
SUMMED_MASSES_REVERSE = [
    114.042927,
    261.078327,
    389.17329,
    502.257354,
    639.316266,
    696.33773,
    843.406144,
    972.448737,
    1087.47568,
    1247.506329,
]
TOTAL_MASS = 1318.543443
EXPECTED_B_IONS = [x + ION_OFFSET['b'] for x in SUMMED_MASSES_FORWARD]
EXPECTED_Y_IONS = [x + ION_OFFSET['y'] for x in SUMMED_MASSES_REVERSE]
EXPECTED_PRECURSOR_MASS = TOTAL_MASS + C_TERMINUS + N_TERMINUS
TEST_OBSERVED_MZS = [
    114.22,
    261.33,
    389.42,
]

class TestMzMatch(unittest.TestCase):
    """ Testing suite for the inSPIRE mz_match input utilities.
    """
    def test_compute_potential_mws_forward(self):
        """ Function to test the compute_potential_mws function with reverse=False.
        """
        forward_mws, forward_total_mw = compute_potential_mws(
            TEST_PEPTIDE,
            TEST_PTM_SEQ,
            False,
            TEST_PTM_WEIGHTS,
        )
        self.assertAlmostEqual(forward_total_mw, TOTAL_MASS)
        for returned_mw, expected_mw in zip(forward_mws.tolist(), SUMMED_MASSES_FORWARD):
            self.assertAlmostEqual(returned_mw, expected_mw)

    def test_compute_potential_mws_reverse(self):
        """ Function to test the compute_potential_mws function with reverse=True.
        """
        forward_mws, forward_total_mw = compute_potential_mws(
            TEST_PEPTIDE,
            TEST_PTM_SEQ,
            True,
            TEST_PTM_WEIGHTS,
        )
        self.assertAlmostEqual(forward_total_mw, TOTAL_MASS)
        for returned_mw, expected_mw in zip(forward_mws.tolist(), SUMMED_MASSES_REVERSE):
            self.assertAlmostEqual(returned_mw, expected_mw)

    def test_get_ion_mases(self):
        """ Function to test the get_ion_masses function.
        """
        masses, total_mass = get_ion_masses(
            TEST_PEPTIDE,
            TEST_PTM_WEIGHTS,
            TEST_PTM_SEQ,
        )
        for returned_mw, expected_mw in zip(masses['b'].tolist(), EXPECTED_B_IONS):
            self.assertAlmostEqual(returned_mw, expected_mw)

        for returned_mw, expected_mw in zip(masses['y'].tolist(), EXPECTED_Y_IONS):
            self.assertAlmostEqual(returned_mw, expected_mw)

        self.assertAlmostEqual(total_mass, EXPECTED_PRECURSOR_MASS)

    def test_match_mz(self):
        """ Function to test the match_mz function.
        """
        mz_error, mz_index = match_mz(
            261.32 - PROTON,
            1,
            np.array(TEST_OBSERVED_MZS),
        )
        self.assertAlmostEqual(mz_error, 0.01)
        self.assertEqual(mz_index, 1)
