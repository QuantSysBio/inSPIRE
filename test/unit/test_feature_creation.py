""" Test suite for the inSPIRE feature_creation utilities.
"""
import unittest

import pandas as pd

from inspire.config import Config
from inspire.feature_creation import process_unknown_modifications

class TestFeatureCreation(unittest.TestCase):
    """ Testing suite for the inSPIRE feature_creation input utilities.
    """
    def setUp(self):
        self.config = Config('test/resources/config.yml')
        self.target_df = pd.read_csv('test/resources/formatted_search.csv')
        self.mods_df = pd.read_csv('test/resources/mods_df.csv')

    def test_process_unknown_modifications_default_without_unknown(self):
        """ Function to test the process_unknown_modifications function with default settings.
        """
        processed_df = process_unknown_modifications(
            self.target_df,
            self.mods_df,
            self.config
        )
        self.assertEqual(processed_df.shape[0], 408)

    def test_process_unknown_modifications_default_with_unknown(self):
        """ Function to test the process_unknown_modifications function
            with an unknown modification.
        """
        self.mods_df['Name'] = 'blah'
        processed_df = process_unknown_modifications(
            self.target_df,
            self.mods_df,
            self.config
        )
        self.assertEqual(processed_df.shape[0], 399)

    def test_process_unknown_modifications_keep(self):
        """ Function to test the process_unknown_modifications with all results kept.
        """
        self.config.filter_c = False
        self.config.drop_unknown_mods = False
        self.mods_df['Name'] = 'blah'
        processed_df = process_unknown_modifications(
            self.target_df,
            self.mods_df,
            self.config
        )
        self.assertEqual(processed_df.shape[0], 409)

    


