""" Test suite for the inSPIRE predict_spectra.
"""
import os
import unittest

from math import acos, pi

import numpy as np
import pandas as pd

from inspire.config import Config
from inspire.predict_spectra import predict_spectra
from inspire.input.msp import msp_to_df

def calculate_spectral_angle(df_row):
    """ Helper function to get spectral angle between predicted spectra.
    """
    keys_to_check = set(list(df_row['expectedIons'].keys()) + list(df_row['predictedIons'].keys()))
    expected_l2_norm = np.linalg.norm(np.array(list(df_row['expectedIons'].values())), ord=2)
    predicted_l2_norm = np.linalg.norm(np.array(list(df_row['predictedIons'].values())), ord=2)

    product = 0.0
    for ion in keys_to_check:
        product += (df_row['expectedIons'].get(ion, 0.0) * df_row['predictedIons'].get(ion, 0.0))

    l2_norm_product = expected_l2_norm * predicted_l2_norm
    product /= l2_norm_product
    product = max(min(product, 1.0), 0.0)

    return 1.0 - 2*acos(product)/pi


class TestPredictSpectra(unittest.TestCase):
    """ Testing suite for the inSPIRE predict spectra utilities
    """
    def setUp(self):
        self.config = Config('test/resources/config.yml')
        self.mods_df = pd.read_csv('test/resources/mods_df.csv')

    @classmethod
    def tearDownClass(cls):
        os.remove('test/resources/output/prositPredictions.msp')
        os.remove('test/resources/output/ms2pipInput_Immuno-HCD_predictions.msp')

    def test_predict_spectra_prosit(self):
        """ Function to test the inSPIRE predict spectra function for Prosit input.
        """
        self.config.spectral_predictor = 'prosit'
        predict_spectra(self.config, pipeline='core')
        prediction_df = msp_to_df('test/resources/output/prositPredictions.msp', 'prosit', None)
        expected_df = msp_to_df('test/resources/prositPredictions.msp', 'prosit', None)
        expected_df = expected_df.rename(
            columns={'prositIons': 'expectedIons', 'iRT': 'expected_iRT'}
        )
        prediction_df = prediction_df.rename(
            columns={'prositIons': 'predictedIons', 'iRT': 'predicted_iRT'}
        )

        combined_df = pd.merge(
            expected_df,
            prediction_df,
            how='inner',
            on=['modified_sequence', 'charge', 'collisionEnergy'],
        )
        combined_df['iRT_diff'] = np.abs(combined_df['expected_iRT'] - combined_df['predicted_iRT'])
        combined_df['spectralAngle'] = combined_df.apply(calculate_spectral_angle, axis=1)
        self.assertEqual(combined_df.shape[0], expected_df.shape[0])
        self.assertAlmostEqual(combined_df['iRT_diff'].mean(), 0.0)
        self.assertAlmostEqual(combined_df['spectralAngle'].mean(), 1.0)

    def test_predict_spectra_ms2pip(self):
        """ Function to test the inSPIRE predict spectra function for Prosit input.
        """
        self.config.spectral_predictor = 'ms2pip'
        self.config.ms2pip_model = 'Immuno-HCD'
        predict_spectra(self.config, pipeline='core')
        expected_df = msp_to_df(
            'test/resources/ms2pipInput_Immuno-HCD_predictions.msp', 'ms2pip', self.mods_df
        )
        prediction_df = msp_to_df(
            'test/resources/output/ms2pipInput_Immuno-HCD_predictions.msp', 'ms2pip', self.mods_df
        )
        expected_df = expected_df.rename(
            columns={'prositIons': 'expectedIons', 'iRT': 'expected_iRT'}
        )
        prediction_df = prediction_df.rename(
            columns={'prositIons': 'predictedIons', 'iRT': 'predicted_iRT'}
        )

        combined_df = pd.merge(
            expected_df,
            prediction_df,
            how='inner',
            on=['peptide', 'ptm_seq', 'charge'],
        )
        combined_df['spectralAngle'] = combined_df.apply(calculate_spectral_angle, axis=1)
        self.assertEqual(combined_df.shape[0], expected_df.shape[0])
        self.assertAlmostEqual(combined_df['spectralAngle'].mean(), 1.0)
