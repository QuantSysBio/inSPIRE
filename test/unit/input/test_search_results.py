""" Test suite for the inSPIRE search_results input utilities.
"""
import os
import unittest

from inspire.config import Config
from inspire.input.search_results import generic_read_df

EXPECTED_COLUMNS = {
    'source',
    'scan',
    'peptide',
    'Label',
    'proteins',
    'ptm_seq',
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

class TestSearchResults(unittest.TestCase):
    """ Testing suite for the inSPIRE search_results input utilities.
    """
    def setUp(self):
        cwd = os.getcwd()
        self.search_file_path = f'{cwd}/test/resources/max_quant_search.txt'
        self.config = Config('test/resources/config.yml')

    def test_generic_read_df(self):
        """ Function to test the generic_read_df function.
        """
        self.config.search_engine = 'maxquant'
        self.config.search_results = [self.search_file_path]
        self.config.fixed_modifications = ['Carbamidomethylation']
        search_df, mods_df = generic_read_df(
            self.config,
            save_dfs=False
        )
        self.assertEqual(mods_df['Name'].tolist(), ['Oxidation (M)', 'Carbamidomethylation'])
        self.assertEqual(mods_df['Identifier'].tolist(), [1, 2])
        self.assertEqual(search_df.shape[0], 409)
        self.assertEqual(EXPECTED_COLUMNS, set(search_df.columns))
        self.assertEqual(search_df['ptm_seq'].iloc[408], '0.2222222.0')

if __name__ == '__main__':
    unittest.main()
