""" Test suite for the inSPIRE peaks input utilities.
"""
import os
import unittest

from inspire.input.peaks import read_peaks_data

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

class TeastPeaks(unittest.TestCase):
    """ Testing suite for the inSPIRE predict spectra utilities.
    """
    def setUp(self):
        cwd = os.getcwd()
        self.search_file_path = f'{cwd}/test/resources/peaks_search.csv'

    def test_read_peaks_data(self):
        """ Function to test the read_peaks_data function.
        """
        search_df, mods_df = read_peaks_data(
            [self.search_file_path]
        )
        self.assertEqual(mods_df['Name'].tolist(), ['Carbamidomethylation', 'Oxidation (M)'])
        self.assertEqual(mods_df['Identifier'].tolist(), [1, 2])
        self.assertEqual(search_df.shape[0], 426)
        self.assertEqual(EXPECTED_COLUMNS, set(search_df.columns))


if __name__ == '__main__':
    unittest.main()
