""" Test suite for the inSPIRE accession utilities.
"""
import unittest


import numpy as np

from inspire.config import Config
from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    PEPTIDE_KEY,
    LABEL_KEY,
)
from inspire.accession import (
    validate_accession_stratum
)

TEST_PROTEOME = [
    ('test', 'ACDEFGHLKMNPQR')
]
TEST_PROTEOME_REVERSED = [
    ('reversed_test', 'RQPNMKLHGFEDCA')
]
TEST_PEPTIDE_CANONICAL = 'FGHIK'
TEST_PEPTIDE_SPLICED = 'DEFGKMNP'
TEST_REVERSED_PEPTIDE_CANONICAL = 'GFED'

class TestAccession(unittest.TestCase):
    """ Testing suite for the inSPIRE accession utilities
    """
    def setUp(self):
        self.config = Config('test/resources/config.yml')

    def test_validate_accession_stratum_canonical(self):
        """ Function to test the validate_accession_stratum function.
        """
        updated_df_row = validate_accession_stratum(
            {
                PEPTIDE_KEY: TEST_PEPTIDE_CANONICAL,
                ACCESSION_STRATUM_KEY: 2,
                LABEL_KEY: 1,
            },
            TEST_PROTEOME,
            TEST_PROTEOME_REVERSED,
            self.config
        )
        self.assertEqual(updated_df_row[ACCESSION_STRATUM_KEY], 0)

    def test_validate_accession_stratum_canonical_reversed(self):
        """ Function to test the validate_accession_stratum function.
        """
        updated_df_row = validate_accession_stratum(
            {
                PEPTIDE_KEY: TEST_REVERSED_PEPTIDE_CANONICAL,
                ACCESSION_STRATUM_KEY: 2,
                LABEL_KEY: -1,
            },
            TEST_PROTEOME,
            TEST_PROTEOME_REVERSED,
            self.config
        )
        self.assertEqual(updated_df_row[ACCESSION_STRATUM_KEY], 0)

    def test_validate_accession_stratum_spliced(self):
        """ Function to test the validate_accession_stratum function.
        """
        updated_df_row = validate_accession_stratum(
            {
                PEPTIDE_KEY: TEST_PEPTIDE_SPLICED,
                ACCESSION_STRATUM_KEY: 1,
                LABEL_KEY: 1,
            },
            TEST_PROTEOME,
            TEST_PROTEOME_REVERSED,
            self.config
        )
        self.assertEqual(updated_df_row[ACCESSION_STRATUM_KEY], 1)
