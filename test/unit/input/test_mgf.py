""" Test suite for the inSPIRE mgf input utilities.
"""
import unittest

from inspire.constants import CHARGE_KEY, RT_KEY
from inspire.input.mgf import process_mgf_file

EXPECTED_OUTPUT_COLUMNS = [
    'source', 'scan', 'intensities', 'mzs'
]
SCANS_TO_SELECT = {
    1779, 2358, 2603, 2604, 217, 335,
}

class TestMgf(unittest.TestCase):
    """ Testing suite for the inSPIRE mgf input utilities.
    """
    def test_process_mgf_file_default(self):
        """ Test process mgf file.
        """
        mgf_df = process_mgf_file('test/resources/test.mgf', None, None, None)
        self.assertEqual(mgf_df.shape[0], 931)
        self.assertEqual(list(mgf_df.columns), EXPECTED_OUTPUT_COLUMNS)

    def test_process_mgf_file_filter_scans(self):
        """ Test process mgf file.
        """
        mgf_df = process_mgf_file('test/resources/test.mgf', SCANS_TO_SELECT, None, None)
        self.assertEqual(mgf_df.shape[0], 6)
        self.assertEqual(list(mgf_df.columns), EXPECTED_OUTPUT_COLUMNS)

    def test_process_mgf_file_filter_scans_with_charge(self):
        """ Test process mgf file including charge column.
        """
        mgf_df = process_mgf_file(
            'test/resources/test.mgf',
            SCANS_TO_SELECT,
            None,
            None,
            with_charge=True,
        )
        self.assertEqual(mgf_df.shape[0], 6)
        self.assertEqual(list(mgf_df.columns), EXPECTED_OUTPUT_COLUMNS + [CHARGE_KEY])

    def test_process_mgf_file_filter_scans_with_rt(self):
        """ Test process mgf file including rt column.
        """
        mgf_df = process_mgf_file(
            'test/resources/test.mgf', SCANS_TO_SELECT, None, None, with_retention_time=True
        )
        self.assertEqual(mgf_df.shape[0], 6)
        self.assertEqual(list(mgf_df.columns), EXPECTED_OUTPUT_COLUMNS + [RT_KEY])
