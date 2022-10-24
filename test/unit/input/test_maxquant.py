""" Test suite for the inSPIRE maxquant input utilities.
"""
import os
import unittest


from inspire.input.maxquant import read_mq_data

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

class TestMaxQuant(unittest.TestCase):
    """ Testing suite for the inSPIRE maxquant input utilities.
    """
    def setUp(self):
        cwd = os.getcwd()
        self.search_file_path = f'{cwd}/test/resources/max_quant_search.txt'

    def test_read_mq_data(self):
        """ Function to test the read_mq_data function with reduce=True.
        """
        search_df, mods_df = read_mq_data(
            [self.search_file_path],
        )
        self.assertEqual(mods_df['Name'].tolist(), ['Oxidation (M)'])
        self.assertEqual(mods_df['Identifier'].tolist(), [1])
        self.assertEqual(search_df.shape[0], 409)
        self.assertEqual(EXPECTED_COLUMNS, set(search_df.columns))



if __name__ == '__main__':
    unittest.main()
