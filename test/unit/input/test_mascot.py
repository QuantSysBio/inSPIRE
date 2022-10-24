""" Test suite for the inSPIRE mascot input utilities.
"""
import os
import unittest

from inspire.input.mascot import MASCOT_PEP_QUERY_KEY, read_mascot_data


EXPECTED_COLUMNS = {
    'source',
    'scan',
    'peptide',
    'Label',
    'proteins',
    'ptm_seq',
    'pep_query',
    'sequenceLength',
    'missedCleavages',
    'charge',
    'massDiff',
    'retentionTime',
    'engineScore',
    'deltaScore',
    'ms1Intensity',
    'fromChimera',
    'avgResidueMass',
}



class TestMascot(unittest.TestCase):
    """ Testing suite for the inSPIRE mascot input utilities.
    """
    def setUp(self):
        cwd = os.getcwd()
        self.target_file_path = f'{cwd}/test/resources/mascot_search.csv'
        self.decoy_file_path = f'{cwd}/test/resources/mascot_decoy_search.csv'

    def test_read_mascot_data_reduce(self):
        """ Function to test the read_mascot_data function with reduce=True.
        """
        search_df, mods_df = read_mascot_data(
            [self.target_file_path, self.decoy_file_path],
            None,
            None,
            True
        )
        self.assertEqual(search_df.shape[0], 123)
        self.assertEqual(search_df.shape[0], search_df[MASCOT_PEP_QUERY_KEY].nunique())
        self.assertEqual(mods_df['Name'].tolist(), ['Oxidation (M)', 'Carbamidomethyl (C)'])
        self.assertEqual(mods_df['Identifier'].tolist(), [1, 2])

    def test_read_mascot_data_no_reduce(self):
        """ Function to test the read_mascot_data function with reduce=False.
        """
        search_df, mods_df = read_mascot_data(
            [self.target_file_path, self.decoy_file_path],
            None,
            None,
            False,
        )
        self.assertEqual(search_df.shape[0], 239)
        self.assertGreater(search_df.shape[0], search_df[MASCOT_PEP_QUERY_KEY].nunique())
        self.assertEqual(mods_df['Name'].tolist(), ['Oxidation (M)', 'Carbamidomethyl (C)'])
        self.assertEqual(mods_df['Identifier'].tolist(), [1, 2])
        self.assertEqual(EXPECTED_COLUMNS, set(search_df.columns))

if __name__ == '__main__':
    unittest.main()
