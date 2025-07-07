""" Functions for calibrating the optimal collision energy setting
    for your experiment.
"""
import polars as pl

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    KNOWN_PTM_WEIGHTS,
    LABEL_KEY,
    OKCYAN_TEXT,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    SCAN_KEY,
    SEQ_LEN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
    CHARGE_KEY,
    INTENSITIES_KEY,
    MZS_KEY,
    PEPTIDE_KEY,
    PROSIT_INTES_KEY,
    PROSIT_IONS_KEY,
    PROSIT_SEQ_KEY,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.input.search_results import generic_read_df
from inspire.feature_creation import combine_spectral_data
from inspire.predict_spectra import predict_spectra
from inspire.prepare import write_prosit_input_df
from inspire.spectral_features import calculate_spectral_features, get_spectral_return_dtype
from inspire.utils import (
    check_bad_mods,
    get_ox_flag,
    get_cam_flag,
)

COLLISION_ENERGY_RANGE = [20 + i for i in range(21)]

def _get_top_hits(config):
    """ Function to extract the top scoring hits for collision energy calibration.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    target_df, mods_df = generic_read_df(config, save_dfs=False, for_calibration=True)

    unknown_mods = mods_df[
        (mods_df[PTM_NAME_KEY] != 'Oxidation (M)') &
        ((mods_df[PTM_NAME_KEY] != 'Carbamidomethyl (C)'))
    ][PTM_ID_KEY].tolist()
    unknown_mods = {str(x) for x in unknown_mods}

    target_df = target_df.with_columns(
        pl.col(PTM_SEQ_KEY).map_elements(
            lambda x : check_bad_mods(x, unknown_mods),
            skip_nulls=False, return_dtype=pl.Boolean,
        ).alias('unknownModifications')
    )
    target_df = target_df.filter(
        pl.col('unknownModifications').not_()
    ).drop(
        ['unknownModifications'],
    )

    if ACCESSION_STRATUM_KEY in target_df.columns:
        target_df = target_df.filter(pl.col(ACCESSION_STRATUM_KEY).eq(0))

    target_df = target_df.filter(
        target_df[LABEL_KEY] == 1
    )

    if config.calibrate_per_file:
        target_dfs = []
        for source_file in sorted(target_df[SOURCE_KEY].unique().to_list()):
            filt_targ_df = target_df.filter(pl.col('source').eq(source_file))
            filt_targ_df = filter_target_df(filt_targ_df, config.rescore_method)
            target_dfs.append(filt_targ_df)
        target_df = pl.concat(target_dfs)
    else:
        target_df = filter_target_df(target_df, config.rescore_method)

    return target_df, mods_df

def filter_target_df(target_df, rescore_method):
    """ Function to filter target df to get the top quality hits.
    """
    if rescore_method == 'percolatorSeparate':
        score_cut_off = target_df[ENGINE_SCORE_KEY].quantile(0.98)
    else:
        score_cut_off = target_df[ENGINE_SCORE_KEY].quantile(0.9)

    target_df = target_df.filter(
        (target_df[ENGINE_SCORE_KEY] > score_cut_off)
    )

    if target_df.shape[0] > 600:
        target_df = target_df.top_k(
            600,
            by=[ENGINE_SCORE_KEY, SEQ_LEN_KEY, SCAN_KEY, PEPTIDE_KEY],
            descending=True,
        )

    return target_df

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

    return target_df, mods_df

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
    target_df, mods_df = prepare_calibration(config)
    predict_spectra(config, 'calibrate')

    prosit_df = msp_to_df(
        f'{config.output_folder}/calibrationPredictions.msp', 'prosit', None
    )

    scan_files = sorted(target_df[SOURCE_KEY].unique().to_list())

    ox_flag = get_ox_flag(mods_df)
    cam_flag = get_cam_flag(mods_df)

    mods_dict = {
        0: 0.0,
        int(cam_flag): KNOWN_PTM_WEIGHTS['Carbamidomethyl (C)'],
        int(ox_flag): KNOWN_PTM_WEIGHTS['Oxidation (M)']
    }

    results_dfs = []
    scan_dfs = []
    for scan_file in scan_files:
        filtered_search_df = target_df.filter(
            target_df[SOURCE_KEY].eq(scan_file)
        )

        scans = filtered_search_df[SCAN_KEY].unique()

        if config.scans_format in ('mzML', 'mzML_rt'):
            scan_df = process_mzml_file(
                f'{config.scans_folder}/{scan_file}.mzML',
                set(scans.to_list()),
            )
        else:
            scan_df = process_mgf_file(
                f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                set(scans.to_list()),
                config.scan_title_format,
                config.source_files,
                combined_source_file=config.combined_scans_file is not None,
            )

        if config.calibrate_per_file:
            results_df, optimal_collision_energy = get_ce_stats(
                filtered_search_df, scan_df, prosit_df, ox_flag, mods_dict, config,
            )
            results_df = results_df.with_columns(pl.lit(scan_file).alias('source'))
            results_df = results_df.filter(pl.col('collisionEnergy').eq(optimal_collision_energy))
            results_dfs.append(results_df)
        else:
            scan_dfs.append(scan_df.unique(subset=[SOURCE_KEY, SCAN_KEY]))

    if config.calibrate_per_file:
        results_df = pl.concat(results_dfs)
        results_df = results_df.sort(by=['source', 'collisionEnergy'])
    else:
        combined_scan_df = pl.concat(scan_dfs)
        results_df, optimal_collision_energy = get_ce_stats(
            target_df, combined_scan_df, prosit_df, ox_flag, mods_dict, config
        )
        print(f'\n\t---> Optimal Collision Energy Setting: {optimal_collision_energy} <---\n')

    results_df.write_csv(
        f'{config.output_folder}/collisionEnergyStats.csv'
    )

def get_ce_stats(target_df, scan_df, prosit_df, ox_flag, mods_dict, config):
    """ Function to get collision energy statist
    """
    print('\t\tCombining all spectral data...')
    combined_df = combine_spectral_data(
        target_df,
        scan_df,
        prosit_df,
        ox_flag,
        'prosit',
        config.output_folder,
        for_calibration=True,
        collision_energy=config.collision_energy,
    )
    print('\t\tCalculating spectral angles...')
    combined_df = combined_df.with_columns(
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
        ]).map_elements(
            lambda x : calculate_spectral_features(
                x,
                mods_dict,
                config.mz_accuracy,
                config.mz_units,
                None,
                '1',
                config.delta_method,
                minimal_features=True,
            ),
            skip_nulls=False,
            return_dtype=get_spectral_return_dtype(
                True, config.delta_method,
            )
        ).alias('spectralResults')
    ).unnest('spectralResults')

    results_df = combined_df.group_by('collisionEnergy', maintain_order=True).agg(
        pl.mean(SPECTRAL_ANGLE_KEY)
    )

    optimal_collision_energy = results_df.sort(
        by='spectralAngle', descending=True,
    )['collisionEnergy'].to_list()[0]

    results_df = results_df.sort(by='collisionEnergy')

    return results_df, optimal_collision_energy
