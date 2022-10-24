""" Test suite for the inSPIRE spectral_features utilities.
"""
import unittest


import numpy as np

from inspire.spectral_features import (
    calculate_spectral_angle,
    check_for_precursor_peak,
    get_coverage,
    get_mz_error_stats,
)

TEST_PEPTIDE = 'ACDEFGHIKM(ox)N'
TEST_CHARGE = 1
EXPERIMENT_MZS = [110.22, 220.33, 330.44]
EXPERIMENT_INTES = [1, 0, 1]
PRED_INTES = [0, 1, 1]
MZ_ERRORS = [0.01, 0.0, -0.01]
ION_LIST = ['y1', 'b1', 'b3']

class TestSpectralFeatures(unittest.TestCase):
    """ Testing suite for the inSPIRE spectral features utilities
    """
    def test_calculate_spectral_angle(self):
        """ Function to test the calculate_spectral_angle function.
        """
        exp_spec = np.array(EXPERIMENT_INTES)
        pred_spec = np.array(PRED_INTES)
        spec_angle = calculate_spectral_angle(exp_spec, pred_spec)
        self.assertAlmostEqual(spec_angle, 1/3)

    def test_check_for_precursor_peak_absent(self):
        """ Function to test the check_for_precursor_peak function.
        """
        matched_inte, assigned_inds = check_for_precursor_peak(
            np.array(EXPERIMENT_MZS),
            np.array(EXPERIMENT_INTES),
            227.3,
            0.02,
            [1, 2],
        )
        self.assertEqual(matched_inte, 0)
        self.assertEqual(assigned_inds, [1, 2])

    def test_check_for_precursor_peak_present(self):
        """ Function to test the check_for_precursor_peak function.
        """
        matched_inte, assigned_inds = check_for_precursor_peak(
            np.array(EXPERIMENT_MZS),
            np.array(EXPERIMENT_INTES),
            330.441,
            0.02,
            [0,],
        )
        self.assertEqual(matched_inte, 1)
        self.assertEqual(assigned_inds, [0, 2])

    def test_get_mz_error_stats(self):
        """ Function to test the get_mz_error_stats function.
        """
        med_err, err_var = get_mz_error_stats(MZ_ERRORS, 0.02)
        self.assertAlmostEqual(med_err, 2/300)
        self.assertAlmostEqual(err_var, 0.0001)

    def test_get_coverage(self):
        """ Function to test the get_coverage function.
        """
        total_cov, b_cov, y_cov = get_coverage(
            4, np.array(ION_LIST), per_letter=True
        )
        self.assertAlmostEqual(total_cov, 2/3)
        self.assertAlmostEqual(b_cov, 2/3)
        self.assertAlmostEqual(y_cov, 1/3)
