""" Functions for calibrating the optimal collision energy setting
    for your experiment.
"""
import pandas as pd

from inspire.constants import (
    BOLD_TEXT,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    LABEL_KEY,
    OKBLUE_TEXT,
    OKCYAN_TEXT,
    PTM_SEQ_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
    UNDERLINE_TEXT,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.input.search_results import generic_read_df
from inspire.feature_creation import combine_spectral_data
from inspire.predict_spectra import predict_spectra
from inspire.prepare import write_prosit_input_df
from inspire.spectral_features import calculate_spectral_features
from inspire.utils import get_ox_flag, remove_source_suffixes

COLLISION_ENERGY_RANGE = [20 + i for i in range(21)]

def _get_top_hits(config):
    """ Function to extract the top scoring hits for collision energy calibration.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    target_df, mods_df = generic_read_df(config)

    target_df = target_df[target_df[PTM_SEQ_KEY].isna()]
    top_5_pct_cut = target_df[ENGINE_SCORE_KEY].quantile(0.95)

    target_df = target_df[
        (target_df[ENGINE_SCORE_KEY] > top_5_pct_cut) &
        (target_df[LABEL_KEY] == 1)
    ]

    if target_df.shape[0] > 1000:
        target_df = target_df.sort_values(by=ENGINE_SCORE_KEY)
        target_df = target_df.reset_index(drop=True)
        target_df = target_df[target_df.index < 1000]

    return target_df, mods_df

def prepare_calibration(config):
    """ Function to generate Prosit input for collision energy calibration.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    target_df, mods_df = _get_top_hits(config)

    for idx, collision_energy in enumerate(COLLISION_ENERGY_RANGE):
        write_prosit_input_df(
            target_df,
            mods_df,
            config,
            collision_energy,
            'calibrationInput',
            overwrite=idx==0,
        )

def calibrate(config):
    """ Function to calibrate the optimal collision energy for Prosit input.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    print(
        OKCYAN_TEXT +
        '\tSelecting top hits...' +
        ENDC_TEXT
    )
    prepare_calibration(config)
    predict_spectra(config, 'calibrate')

    target_df, mods_df = _get_top_hits(config)
    prosit_df = msp_to_df(
        f'{config.output_folder}/calibrationPredictions.msp', 'prosit', None,
    )

    if config.combined_scans_file is not None:
        scan_files = [remove_source_suffixes(config.combined_scans_file)]
    else:
        scan_files = target_df[SOURCE_KEY].unique().tolist()

    ox_flag = get_ox_flag(mods_df)

    scan_dfs = []
    for scan_file in scan_files:
        if config.combined_scans_file is not None:
            filtered_search_df = target_df
        else:
            filtered_search_df = target_df[target_df[SOURCE_KEY] == scan_file]

        scans = filtered_search_df[SCAN_KEY].unique()
        if config.scans_format == 'mzML':
            scan_df = process_mzml_file(
                f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                set(scans.tolist()),
            )
        else:
            scan_df = process_mgf_file(
                f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                set(scans.tolist()),
                config.scan_title_format,
                config.source_files
            )
        scan_dfs.append(scan_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY]))
    combined_scan_df = pd.concat(scan_dfs)
    print(
        OKCYAN_TEXT +
        '\t\tCombining all spectral data...' +
        ENDC_TEXT
    )
    combined_df = combine_spectral_data(
        filtered_search_df,
        combined_scan_df,
        prosit_df,
        ox_flag,
        'prosit',
    )
    print(
        OKCYAN_TEXT +
        '\t\tCalculating spectral angles...' +
        ENDC_TEXT
    )
    combined_df = combined_df.apply(
        lambda x : calculate_spectral_features(
            x,
            {0: 0.0},
            config.mz_accuracy,
            config.mz_units,
            None,
            '1',
            config.delta_method,
            config.spectral_predictor,
            spectral_angle_only=True,
        ),
        axis=1
    )
    results_df = combined_df.groupby('collisionEnergy', as_index=False)[
        SPECTRAL_ANGLE_KEY
    ].mean()
    results_df.columns = ['collisionEnergy', SPECTRAL_ANGLE_KEY]

    optimal_collision_energy = results_df['collisionEnergy'].iloc[
        results_df[SPECTRAL_ANGLE_KEY].idxmax()
    ]
    print(
        OKBLUE_TEXT + BOLD_TEXT + UNDERLINE_TEXT +
        f'\n\t---> Optimal Collision Energy Setting: {optimal_collision_energy} <---\n' +
        ENDC_TEXT
    )
    results_df.to_csv(
        f'{config.output_folder}/collisionEnergyStats.csv',
        index=False
    )
