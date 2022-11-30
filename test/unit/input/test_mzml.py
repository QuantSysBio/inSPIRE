""" Test suite for the inSPIRE mzML input utilities.
"""
import unittest

from inspire.constants import CHARGE_KEY, RT_KEY
from inspire.input.mzml import process_mzml_file

EXPECTED_OUTPUT_COLUMNS = [
    'source', 'scan', 'intensities', 'mzs'
]
SCANS_TO_SELECT = {
    1779, 746, 923, 2417, 2482, 3300,
}

class TestMzML(unittest.TestCase):
    """ Testing suite for the inSPIRE mzML input utilities.
    """
    def test_process_mzml_file_default(self):
        """ Test process mzML file.
        """
        mzml_df = process_mzml_file('test/resources/test.mzML', None)
        self.assertEqual(mzml_df.shape[0], 1263)
        self.assertEqual(list(mzml_df.columns), EXPECTED_OUTPUT_COLUMNS)

    def test_process_mzml_file_filter_scans(self):
        """ Test process mzML file.
        """
        mzml_df = process_mzml_file('test/resources/test.mzML', SCANS_TO_SELECT)
        self.assertEqual(mzml_df.shape[0], 6)
        self.assertEqual(list(mzml_df.columns), EXPECTED_OUTPUT_COLUMNS)

    def test_process_mzml_file_filter_scans_with_charge(self):
        """ Test process mzML file including charge column.
        """
        mzml_df = process_mzml_file('test/resources/test.mzML', SCANS_TO_SELECT, with_charge=True)
        self.assertEqual(mzml_df.shape[0], 6)
        self.assertEqual(list(mzml_df.columns), EXPECTED_OUTPUT_COLUMNS + [CHARGE_KEY])

    def test_process_mzml_file_filter_scans_with_rt(self):
        """ Test process mzML file including rt column.
        """
        mzml_df = process_mzml_file(
            'test/resources/test.mzML', SCANS_TO_SELECT, with_retention_time=True
        )
        self.assertEqual(mzml_df.shape[0], 6)
        self.assertEqual(list(mzml_df.columns), EXPECTED_OUTPUT_COLUMNS + [RT_KEY])
