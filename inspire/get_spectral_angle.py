""" Functions for simple pipeline to get spectral angle on of identified PSMs.
"""
import pandas as pd
import polars as pl

from inspire.constants import(
    CHARGE_KEY,
    KNOWN_PTM_WEIGHTS,
    SCAN_KEY,
    SOURCE_KEY,
    INTENSITIES_KEY,
    MZS_KEY,
    PEPTIDE_KEY,
    PROSIT_INTES_KEY,
    PROSIT_IONS_KEY,
    PROSIT_SEQ_KEY,
    PTM_SEQ_KEY,
    RT_KEY,
)
from inspire.input.msp import msp_to_df
from inspire.input.ssl import ssl_file_to_inspire_format
from inspire.predict_spectra import predict_spectra
from inspire.spectral_features import calculate_spectral_features
from inspire.utils import convert_mod_seq_to_ptm_seq, fetch_scan_data, filter_for_prosit

def get_spectral_angle(config):
    """ Function to generate pair plots of selected PSMs (experimental vs. Prosit
        predicted spectra.).

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.
    """
    for dataset in config.sa_query_dfs:
        if dataset.endswith('.ssl'):
            input_df = ssl_file_to_inspire_format(dataset)
        else:
            input_df = pl.read_csv(dataset)

        output_loc = dataset.split('.csv')[0] + '_spectralAngle.csv'
        input_cols = list(input_df.columns)

        get_charge_from_scan_file = not CHARGE_KEY in input_df.columns
        get_rt = not RT_KEY in input_df.columns
        scan_df = fetch_scan_data(input_df, config, get_charge_from_scan_file, with_rt=get_rt)

        input_df = input_df.join(
            scan_df,
            how='inner',
            on=[SOURCE_KEY, SCAN_KEY]
        )

        input_df = filter_for_prosit(input_df)

        input_df = input_df.with_columns(
            pl.col('modifiedSequence').apply(
                lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
            ).alias('modified_sequence'),
            pl.col(CHARGE_KEY).alias('precursor_charge'),
        )

        if 'collisionEnergy' in input_df.columns:
            input_df = input_df.rename({'collisionEnergy': 'collision_energy'})
        else:
            input_df = input_df.with_columns(
                pl.lit(config.collision_energy).alias('collision_energy')
            )

        input_df[['modified_sequence', 'precursor_charge', 'collision_energy']].write_csv(
            f'{config.output_folder}/saInput.csv'
        )

        predict_spectra(config, pipeline='spectralAngle')
        prosit_df = msp_to_df(f'{config.output_folder}/saPredictions.msp', 'prosit', None)
        prosit_df = prosit_df.unique(subset=['modified_sequence', CHARGE_KEY])

        input_df = input_df.join(
            prosit_df,
            how='inner',
            on=['modified_sequence', CHARGE_KEY]
        )

        input_df = input_df.with_columns(
            pl.col('modifiedSequence').apply(
                convert_mod_seq_to_ptm_seq
            ).alias('ptm_seq'),
        )

        input_df = input_df.with_columns(
            pl.struct([
                CHARGE_KEY,
                'collisionEnergy',
                INTENSITIES_KEY,
                MZS_KEY,
                PEPTIDE_KEY,
                PROSIT_INTES_KEY,
                PROSIT_IONS_KEY,
                PROSIT_SEQ_KEY,
                PTM_SEQ_KEY,
            ]).apply(lambda df_row : calculate_spectral_features(
                df_row,
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
                minimal_features=True,
            )).alias('spectralResults')
        )
        input_df = input_df.unnest('spectralResults')
        output_columns = ['spectralAngle', 'spearmanR']

        try:
            sources = input_df['source'].unique().tolist()
            for source in sources:
                source_df = input_df[input_df['source'] == source]
                source_rt_df = pd.read_csv(f'{config.output_folder}/rt_fit_{source}.csv')
                intercept = source_rt_df['intercepts'].mean()
                coefficient = source_rt_df['coefficents'].mean()
                source_df['deltaRT'] = source_df[['source', 'retentionTime', 'iRT']].apply(
                    lambda df_row, coef=coefficient, interc=intercept : abs(
                        (df_row['retentionTime'] - (df_row['iRT']*coef)+interc)/coef
                    ),
                    axis=1,
                )
                output_columns.append('deltaRT')
        except Exception as e:
            print(f'Retention Time Prediction Comparison failed with error {e}')

        input_df.select(input_cols + output_columns).write_csv(
            output_loc
        )
