""" Test suite for the inSPIRE prosit_delta utilities.
"""
import unittest

import numpy as np

from inspire.prosit_delta import (
    calculate_sa_from_dict,
    check_oxidation,
    compute_single_mz,
    convert_ptm_seq,
    get_intes_at_loc,
    get_mass_diff,
)

TEST_PEPTIDE = 'ACDEFGHIKMN'
TEST_PTM_SEQ = '0.02000000010.0'
TEST_PTM_WEIGHTS = {
    0: 0.0,
    1: 15.994915,
    2: 57.021464,
}
PROSIT_IONS = [
    'b1',
    'y2',
    'b1^3',
]
MATCHED_INTES = [
    1,
    2,
    4,
]
EXPERIMENT_ION_DICT = {
    'y4': 1,
    'b1^3': 1,
}
PROSIT_ION_DICT = {
    'b1': 1,
    'b1^3': 1,
}
PRED_INTES = [0, 1, 1]

class TestPrositDelta(unittest.TestCase):
    """ Testing suite for the inSPIRE prosit_delta utilities.
    """
    def test_check_oxidation(self):
        """ Function to test the check_oxidation function.
        """
        oxidation_true = check_oxidation(11, TEST_PTM_SEQ, 9, '1')
        self.assertEqual(oxidation_true, 1)

        oxidation_false = check_oxidation(11, TEST_PTM_SEQ, 10, '1')
        self.assertEqual(oxidation_false, 0)

    def test_get_mass_diff(self):
        """ Function to test the get_mass_diff function.
        """
        mass_diff = get_mass_diff(TEST_PEPTIDE, TEST_PTM_SEQ, 1, '1')
        self.assertAlmostEqual(mass_diff, 88.993535)

    def test_get_intes_at_loc(self):
        """ Function to test the get_intes_at_loc function.
        """
        summed_inte = get_intes_at_loc(
            11,
            np.array(MATCHED_INTES),
            np.array(PROSIT_IONS),
            1,
            'b'
        )
        self.assertEqual(summed_inte, 5)

    def test_compute_single_mz(self):
        """ Function to test the compute_single_mz function.
        """
        single_mz = compute_single_mz(TEST_PEPTIDE, TEST_PTM_SEQ, 'y', 6, TEST_PTM_WEIGHTS)
        self.assertAlmostEqual(single_mz, 714.3482947)

    def test_calculate_sa_from_dict(self):
        """ Function to test the calculate_sa_from_dict function.
        """
        spectral_angle = calculate_sa_from_dict(PROSIT_ION_DICT, EXPERIMENT_ION_DICT)
        self.assertAlmostEqual(spectral_angle, 1/3)

    def test_convert_ptm_seq(self):
        """ Function to test the convert_ptm_seq function.
        """
        ptm_seq = convert_ptm_seq(TEST_PTM_SEQ, 2)
        self.assertEqual(ptm_seq, '0.00200000010.0')
