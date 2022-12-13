""" Functions for simple pipeline to get spectral angle on of identified PSMs.
"""
import pandas as pd

from inspire.constants import(
    CHARGE_KEY,
    KNOWN_PTM_WEIGHTS,
    SCAN_KEY,
    SOURCE_KEY,
)
from inspire.input.msp import msp_to_df
from inspire.predict_spectra import predict_spectra
from inspire.retention_time import add_delta_irt
from inspire.spectral_features import calculate_spectral_features
from inspire.utils import convert_mod_seq_to_ptm_seq, fetch_scan_data

def get_spectral_angle(config):
    """ Function to generate pair plots of selected PSMs (experimental vs. Prosit
        predicted spectra.).

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.
    """
    for dataset in config.sa_query_dfs:
        input_df = pd.read_csv(dataset)
        output_loc = dataset.split('.csv')[0] + '_spectralAngle.csv'
        input_cols = list(input_df.columns)

        get_charge_from_scan_file = not CHARGE_KEY in input_df.columns
        scan_df = fetch_scan_data(input_df, config, get_charge_from_scan_file)

        input_df = pd.merge(
            input_df,
            scan_df,
            how='inner',
            on=[SOURCE_KEY, SCAN_KEY]
        )

        input_df['modified_sequence'] = input_df['modifiedSequence'].apply(
            lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
        )
        input_df['precursor_charge'] = input_df[CHARGE_KEY]
        if 'collisionEnergy' in input_df.columns:
            input_df = input_df.rename(columns={'collisionEnergy': 'collision_energy'})
        else:
            input_df['collision_energy'] = config.collision_energy

        input_df[['modified_sequence', 'precursor_charge', 'collision_energy']].to_csv(
            f'{config.output_folder}/saInput.csv', index=False,
        )

        predict_spectra(config, pipeline='spectralAngle')
        prosit_df = msp_to_df(f'{config.output_folder}/saPredictions.msp', 'prosit', None)
        prosit_df = prosit_df.drop_duplicates(subset=['modified_sequence', CHARGE_KEY])

        input_df = pd.merge(
            input_df,
            prosit_df,
            how='inner',
            on=['modified_sequence', CHARGE_KEY]
        )

        input_df = input_df.reset_index(drop=True)

        input_df['ptm_seq'] = input_df['modifiedSequence'].apply(
            convert_mod_seq_to_ptm_seq
        )
        input_df = input_df.apply(
            lambda x : calculate_spectral_features(
                x,
                {
                    0: 0.0,
                    1: KNOWN_PTM_WEIGHTS['Oxidation (M)'],
                    2: KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
                },
                config.mz_accuracy,
                config.mz_units,
                None,
                '1',
                config.delta_method,
                config.spectral_predictor,
                minimal_features=True,
            ),
            axis=1,
        )

        input_df = add_delta_irt(input_df, config, None)
        input_df[input_cols + ['deltaRT', 'spectralAngle']].to_csv(
            output_loc
        )
