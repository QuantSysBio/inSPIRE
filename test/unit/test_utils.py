""" Test suite for the inSPIRE utils.
"""
import unittest

import pandas as pd
from inspire.constants import PEPTIDE_KEY, PTM_SEQ_KEY

from inspire.utils import (
    add_fixed_modifications,
    get_mokapot_weights,
    get_ox_flag,
    modify_sequence_for_skyline,
    permute_ptms,
    permute_seq,
)


TEST_PEPTIDE = 'ACDEFGHIKMN'
TEST_PTM_SEQ = '0.02000000010.0'
TEST_PTM_WEIGHTS = {
    0: 0.0,
    1: 15.994915,
    2: 57.021464,
}
EXPECTED_PERMUTED_SEQS = [
    'CADEFGHIKMN',
	'ADCEFGHIKMN',
	'ACEDFGHIKMN',
	'ACDFEGHIKMN',
	'ACDEGFHIKMN',
	'ACDEFHGIKMN',
	'ACDEFGIHKMN',
	'ACDEFGHKIMN',
	'ACDEFGHIMKN',
	'ACDEFGHIKNM',
]
EXPECTED_PERMUTED_PTMS = [
    '0.20000000010.0',
	'0.00200000010.0',
	'0.02000000010.0',
	'0.02000000010.0',
	'0.02000000010.0',
	'0.02000000010.0',
	'0.02000000010.0',
	'0.02000000010.0',
	'0.02000000100.0',
	'0.02000000001.0',
]
EXPECTED_MODIFIED_SEQUENCE = 'AC[+57.0]DEFGHIKM[+16.0]N'
class TestUtils(unittest.TestCase):
    """ Testing suite for the inSPIRE utils.
    """
    def setUp(self):
        self.search_df = pd.read_csv('test/resources/formatted_search.csv')
        self.mods_df = pd.read_csv('test/resources/mods_df.csv')

    def test_permute_ptms(self):
        """ Function to test the permute_ptms function.
        """
        permuted_ptms = permute_ptms(
            TEST_PEPTIDE,
            TEST_PTM_SEQ,
        )
        self.assertEqual(permuted_ptms, EXPECTED_PERMUTED_PTMS)

        permute_ptms_uniform_length = permute_ptms(
            TEST_PEPTIDE,
            TEST_PTM_SEQ,
            True
        )
        self.assertEqual(permute_ptms_uniform_length, EXPECTED_PERMUTED_PTMS + [None]*19)

    def test_permute_seqs(self):
        """ Function to test the permute_seq function.
        """
        permuted_seqs = permute_seq(
            TEST_PEPTIDE,
        )
        self.assertEqual(permuted_seqs, EXPECTED_PERMUTED_SEQS)

        permute_seqs_uniform_length = permute_seq(
            TEST_PEPTIDE,
            True
        )
        self.assertEqual(permute_seqs_uniform_length, EXPECTED_PERMUTED_SEQS + [None]*19)

    def test_get_ox_flag(self):
        """ Function to test the get_ox_flag function.
        """
        ox_flag = get_ox_flag(self.mods_df)
        self.assertEqual(ox_flag, 1)

    def test_get_mokapot_weights(self):
        """ Function to test the get_mokapot_weights function.
        """
        mokapot_weights_df = get_mokapot_weights(
            'test/resources/output',
            'final',
        )
        self.assertEqual(mokapot_weights_df.shape, (3, 39))

    def test_modify_sequence_for_skyline(self):
        """ Function to test the modify_sequence_for_skyline function.
        """
        modified_sequence = modify_sequence_for_skyline(
            {
                PEPTIDE_KEY: TEST_PEPTIDE,
                PTM_SEQ_KEY: TEST_PTM_SEQ
            },
            TEST_PTM_WEIGHTS,
        )
        self.assertEqual(modified_sequence, EXPECTED_MODIFIED_SEQUENCE)

    def test_add_fixed_modification(self):
        """ Function to test the get_mokapot_weights function.
        """
        search_df, mods_df = add_fixed_modifications(self.search_df, self.mods_df, ['Carbamidomethyl (C)'])
        self.assertEqual(search_df['ptm_seq'].iloc[408], '0.2222222.0')
        self.assertEqual(mods_df['Name'].iloc[1], 'Carbamidomethyl (C)')
        self.assertEqual(mods_df['isVar'].iloc[1], False)
