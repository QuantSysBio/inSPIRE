""" Functions for simple pipeline to get spectral angle on of identified PSMs.
"""
import os

import pandas as pd
import polars as pl

from inspire.constants import(
    CHARGE_KEY,
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
    STANDARD_PTM_DICT,
)
from inspire.input.msp import msp_to_df
from inspire.input.mhcpan import read_mhcpan_output
from inspire.input.psms import read_single_psms
from inspire.input.ssl import ssl_file_to_inspire_format
from inspire.predict_binding import predict_binding
from inspire.predict_spectra import predict_spectra
from inspire.retention_time import add_delta_irt
from inspire.spectral_features import calculate_spectral_features, get_spectral_return_dtype
from inspire.utils import fetch_scan_data, filter_for_prosit, get_nuggets_pred_df

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
            input_df = read_single_psms(dataset)

        input_df = input_df.with_columns(
            pl.col('source').cast(pl.String)
        )
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
            pl.col('modifiedSequence').str.replace_many(
                [
                    '[+16.0]', '[+57.0]', '[+1.0]', '[+42.0]',
                    '[+119.0]', '[+43.0]', '[+26.0]', '[-17.0]',
                ],
                [
                    '(ox)', '', '', '', '', '', '', '',
                ],
            ).alias('modified_sequence'),
            pl.col(CHARGE_KEY).alias('precursor_charge'),
        )

        if 'collisionEnergy' in input_df.columns:
            input_df = input_df.rename({'collisionEnergy': 'collision_energy'})
        else:
            if isinstance(config.collision_energy, dict):
                input_df = input_df.with_columns(
                    pl.col('source').replace(config.collision_energy).cast(
                        pl.Int64
                    ).alias('collision_energy')
                )
            else:
                input_df = input_df.with_columns(
                    pl.lit(config.collision_energy).alias('collision_energy')
                )

        input_df.select(
            ['modified_sequence', 'precursor_charge', 'collision_energy']
        ).unique(maintain_order=True).write_csv(
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
            ]).map_elements(lambda df_row : calculate_spectral_features(
                df_row,
                STANDARD_PTM_DICT,
                config.mz_accuracy,
                config.mz_units,
                None,
                '1',
                'ignore',
                minimal_features=True,
            ), return_dtype=get_spectral_return_dtype(
                True, 'ignore',
            ),).alias('spectralResults')
        )
        input_df = input_df.unnest('spectralResults')

        output_columns = [
            'spectralAngle', 'spearmanR', 'medianFragmentMzError',
            'fragmentMzErrorVariance', 'iRT',
        ]
        if get_rt:
            output_columns = ['retentionTime'] + output_columns


        if config.use_binding_affinity is not None:
            if not os.path.exists(f'{config.output_folder}/mhcpan'):
                os.mkdir(f'{config.output_folder}/mhcpan')

            mhcpan_files = [
                f for f in os.listdir(f'{config.output_folder}/mhcpan') if f.endswith(".bak")
            ]
            for pan_f in mhcpan_files:
                os.remove(os.path.join(f'{config.output_folder}/mhcpan', pan_f))

            input_df = input_df.with_columns(
                pl.col('peptide').str.len_chars().alias('sequenceLength')
            )
            input_df.select(['peptide', 'sequenceLength']).write_csv(
                f'{config.output_folder}/formated_df.csv'
            )
            for pep_len in input_df['sequenceLength'].unique().to_list():
                input_df.filter(pl.col('sequenceLength').eq(pep_len)).select(
                    ['peptide']
                ).write_csv(
                    f'{config.output_folder}/mhcpan/inputLen{pep_len}_0.txt', include_header=False,
                )

            predict_binding(config)
            mhc_pan_df = read_mhcpan_output(f'{config.output_folder}/mhcpan', per_allele=True)
            ba_cols = [str(x) for x in mhc_pan_df.columns if x.endswith('_BindingAffinity')]
            mhc_pan_df = mhc_pan_df.select(['Peptide'] + ba_cols)

            mhc_pan_df = mhc_pan_df.with_columns(
                pl.struct(ba_cols).map_elements(
                    lambda x : min([x[ba_c] for ba_c in ba_cols]), return_dtype=pl.Float64
                ).alias('mhcpanPrediction')
            ).select(['Peptide'] + ba_cols + ['mhcpanPrediction'])

            mhc_pan_df = mhc_pan_df.rename({'Peptide': 'peptide'})
            mhc_pan_df = mhc_pan_df.select(
                ['peptide'] + ba_cols + ['mhcpanPrediction']
            )

            input_df = input_df.join(
                mhc_pan_df, how='left', on=['peptide'],
            )
            nugget_pred_df = get_nuggets_pred_df(config)
            input_df = input_df.join(
                nugget_pred_df.select('peptide', 'nuggetsPrediction'), how='left', on=['peptide'],
            )
            output_columns.extend(['mhcpanPrediction', 'nuggetsPrediction'] + ba_cols)


        dfs = []
        sources = input_df['source'].unique().to_list()
        for source in sources:
            source_df = input_df.filter(
                input_df['source'] == source
            )

            if config.rt_fit_loc is not None:
                source_rt_df = pd.read_csv(f'{config.rt_fit_loc}/rt_fit_{source}.csv')
            elif os.path.exists(f'{config.output_folder}/rt_fit_{source}.csv'):
                source_rt_df = pd.read_csv(f'{config.output_folder}/rt_fit_{source}.csv')
            else:
                source_df = add_delta_irt(source_df, config, source)
                dfs.append(source_df)
                continue

            intercept = source_rt_df['intercepts'].mean()
            coefficient = source_rt_df['coefficents'].mean()

            source_df = source_df.with_columns(
                pl.struct(['retentionTime', 'iRT']).map_elements(
                    lambda df_row, coef=coefficient, interc=intercept : abs(
                        (df_row['retentionTime'] - ((df_row['iRT']*coef)+interc))/coef
                    )
                ).alias('deltaRT')
            )
            dfs.append(source_df)

        input_df = pl.concat(dfs)
        output_columns.append('deltaRT')
        input_df.select(input_cols + output_columns).write_csv(
            output_loc
        )
