""" Test suite for the inSPIRE prosit utilities.
"""
import unittest


import numpy as np
from inspire.constants import MAX_CHARGE, MAX_SEQ_LEN

from inspire.prosit import (
    generate_mod_strings,
    generate_mods_string_tuples,
    get_precursor_charge_onehot,
    get_sequence_integer,
    peptide_parser,
    process_csv_file
)

TEST_PEPTIDE = 'ACDEFGHIKM(ox)N'
TEST_CHARGE = 1
EXPECTED_SEQ_INTEGER = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 12, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
EXPECTED_CHARGE_ONE_HOT = [1, 0, 0, 0, 0, 0]
EXPECTED_MOD_STRING_TUPLES = [
    (2, 'C', 'Carbamidomethyl'),
    (10, 'M', 'Oxidation'),
]
EXPECTED_MOD_STRINGS = (
    '2/3,C,Carbamidomethyl/11,M,Oxidation',
    'Carbamidomethyl@C3; Oxidation@M11',
)

class TestProsit(unittest.TestCase):
    """ Testing suite for the inSPIRE prosit utilities.
    """
    def test_get_precursor_charge_onehot(self):
        """ Function to test the get_precursor_charge_onehot function.
        """
        one_hot_charge = get_precursor_charge_onehot([TEST_CHARGE])
        self.assertEqual(one_hot_charge.tolist(), [EXPECTED_CHARGE_ONE_HOT])

    def test_peptide_parser(self):
        """ Function to test the peptide_parser function.
        """
        for res_idx, residue in enumerate(peptide_parser(TEST_PEPTIDE)):
            self.assertEqual(residue, EXPECTED_SEQ_INTEGER[res_idx])

    def test_get_sequence_integer(self):
        """ Function to test the get_sequence_integer function.
        """
        seq_integer = get_sequence_integer([TEST_PEPTIDE])
        self.assertEqual(seq_integer.tolist(), [EXPECTED_SEQ_INTEGER])

    def test_process_csv_file(self):
        """ Function to test the process_csv_file function.
        """
        input_df, prosit_input = process_csv_file('test/resources/output/prositInput.csv')
        self.assertEqual(
            prosit_input['collision_energy_aligned_normed'].shape, (input_df.shape[0], 1)
        )
        self.assertEqual(
            prosit_input['sequence_integer'].shape, (input_df.shape[0], MAX_SEQ_LEN)
        )
        self.assertEqual(
            prosit_input['precursor_charge_onehot'].shape, (input_df.shape[0], MAX_CHARGE)
        )

    def test_generate_mods_string_tuples(self):
        """ Function to get generation of mod string tuples
        """
        mod_tuples = generate_mods_string_tuples(np.array(EXPECTED_SEQ_INTEGER))
        self.assertEqual(mod_tuples, EXPECTED_MOD_STRING_TUPLES)

    def test_generate_mod_strings(self):
        """ Function to get generation of mod string tuples
        """
        mod_strings = generate_mod_strings(np.array(EXPECTED_SEQ_INTEGER))
        self.assertEqual(mod_strings, EXPECTED_MOD_STRINGS)
