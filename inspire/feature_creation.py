""" Functions for writing percolator/mokapot input using Prosit and other features.
"""
from math import log10
import multiprocessing
import os

import pandas as pd

from inspire.accession import process_accession_groups
from inspire.basic_features import create_basic_features
from inspire.constants import(
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    BASIC_FEATURES,
    CHARGE_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    LABEL_KEY,
    OKCYAN_TEXT,
    IN_ACCESSION_KEY,
    PEPTIDE_KEY,
    PSM_ID_KEY,
    PERC_SCAN_ID,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    SCAN_KEY,
    SOURCE_INDEX_KEY,
    SOURCE_KEY,
    WARNING_TEXT,
)
from inspire.input.mascot import MASCOT_PEP_QUERY_KEY
from inspire.input.mgf import process_mgf_file
from inspire.input.mhcpan import read_mhcpan_output
from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.input.search_results import generic_read_df
from inspire.prepare import create_prosit_mod_seq
from inspire.retention_time import add_delta_irt
from inspire.spectral_features import SPECTRAL_FEATURES, DELTA_FEATURES, create_spectral_features
from inspire.utils import (
    get_ox_flag,
    modify_sequence_for_skyline,
    remove_source_suffixes,
    permute_seq,
    permute_ptms,
)

def combine_spectral_data(search_df, scan_df, prosit_df, ox_flag, spectral_predictor):
    """ Function to combine DataFrame of PEAKS results with prosit predicted
        spectrum and the true eperimental spectrum.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        The DataFrame of PEAKS search results.
    mzml_df : pd.DataFrame
        The DataFrame of experimental observed spectra.
    prosit_df : pd.DataFrame
        The DataFrame of prosit predicted spectra.

    Returns
    -------
    final_df : pd.DataFrame
        The DataFrame with all merged Data.
    """
    if spectral_predictor == 'prosit':
        search_df['modified_sequence'] = search_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
            lambda x : create_prosit_mod_seq(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], ox_flag),
            axis=1
        )
        prosit_df = prosit_df.drop_duplicates(
            subset=['modified_sequence', CHARGE_KEY, 'collisionEnergy']
        )


    print(
        OKCYAN_TEXT +
        f'\t\t\t{search_df.shape[0]} in original search results.' +
        ENDC_TEXT
    )
    if spectral_predictor == 'prosit':
        df_with_prosit = pd.merge(
            search_df,
            prosit_df,
            how='inner',
            on=['modified_sequence', CHARGE_KEY]
        )
    elif spectral_predictor == 'ms2pip':
        df_with_prosit = pd.merge(
            search_df,
            prosit_df,
            how='inner',
            on=[PEPTIDE_KEY, PTM_SEQ_KEY, CHARGE_KEY]
        )

    print(
        OKCYAN_TEXT +
        f'\t\t\t{df_with_prosit.shape[0]} after combination with predicted spectra.'
        + ENDC_TEXT
    )

    scan_df[SOURCE_KEY] = scan_df[SOURCE_KEY].apply(
        remove_source_suffixes
    )

    final_df = pd.merge(
        df_with_prosit,
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )
    print(
        OKCYAN_TEXT +
        f'\t\t\t{final_df.shape[0]} after combination with experimental spectra.' +
        ENDC_TEXT
    )

    return final_df

def filter_input_columns(combined_df, config, file_idx):
    """ Function to filter the final percolator/mokapot input DataFrame to the required
        features.

    Parameters
    ----------
    combined_df : pd.DataFrame
        The percolator/mokapot input DataFrame.
    config : inspire.config.Config
        The config object.

    Returns
    -------
    combined_df : pd.DataFrame
        The input DataFrame containing only the features required for percolator/mokapot.
    """
    psm_id_key = PSM_ID_KEY[config.rescore_method]
    use_cols = [
        psm_id_key,
        LABEL_KEY,
        PERC_SCAN_ID,
    ] + BASIC_FEATURES + SPECTRAL_FEATURES

    if config.delta_method != 'ignore':
        use_cols += DELTA_FEATURES
    use_cols += ['deltaRT']

    if config.use_binding_affinity == 'asFeature':
        use_cols += ['bindingAffinity']

    if config.combined_scans_file is not None:
        scan_files = [remove_source_suffixes(x) for x in config.source_files]
        combined_df[SOURCE_INDEX_KEY] = combined_df[SOURCE_KEY].apply(scan_files.index)
    else:
        combined_df[SOURCE_INDEX_KEY] = file_idx

    use_cols += [SOURCE_INDEX_KEY]

    use_cols += [PEPTIDE_KEY, ACCESSION_KEY]

    if config.use_accession_stratum:
        acc_cols = []
        if config.raw_file_groups is not None:
            for a_idx in range(len(config.accession_hierarchy)):
                stratum_name = config.accession_hierarchy[int(a_idx)]
                for file_group in config.raw_file_groups:
                    f_names = config.raw_file_groups[file_group]
                    combined_df[f'accession_{stratum_name}_{file_group}'] = combined_df.apply(
                        lambda x, a=a_idx, g=f_names: (
                            1 if x[ACCESSION_STRATUM_KEY] == a and x[SOURCE_KEY] in g else 0
                        ),
                        axis=1
                    )
                    acc_cols.append(f'accession_{stratum_name}_{file_group}')
        else:
            for a_idx in range(len(config.accession_hierarchy)):
                stratum_name = config.accession_hierarchy[int(a_idx)]
                combined_df[f'accession_{stratum_name}'] = combined_df[
                    ACCESSION_STRATUM_KEY
                ].apply(
                    lambda x, val=a_idx: 1 if x == val else 0
                )
                acc_cols.append(f'accession_{stratum_name}')

        use_cols += sorted(acc_cols)

    combined_df = combined_df[use_cols].rename(
        columns={ACCESSION_KEY: IN_ACCESSION_KEY[config.rescore_method]}
    )

    return combined_df

def write_with_spectral_features(
        search_df,
        mods_df,
        config,
    ):
    """ Function to write the percolator/mokapot input DataFrame with spectral features.

    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.
    """
    if config.spectral_predictor == 'prosit':
        prosit_predictions = f'{config.output_folder}/prositPredictions.msp'
        prosit_df = msp_to_df(prosit_predictions, 'prosit', None)
    elif config.spectral_predictor == 'ms2pip':
        model = config.ms2pip_model
        ms2pip_predictions = f'{config.output_folder}/ms2pipInput_{model}_predictions.msp'
        prosit_df = msp_to_df(ms2pip_predictions, 'ms2pip', mods_df)

    if config.combined_scans_file is not None:
        scan_files = [remove_source_suffixes(config.combined_scans_file)]
    else:
        scan_files = search_df[SOURCE_KEY].unique().tolist()


    max_scan = search_df[SCAN_KEY].max()
    if config.n_cores == 1 or len(scan_files) == 1:
        for file_idx, scan_file in enumerate(scan_files):
            process_single_file(
                search_df, mods_df, prosit_df, config, file_idx, scan_file, max_scan, False
            )
    else:
        n_cores = min(config.n_cores, multiprocessing.cpu_count())
        func_args = [
            (search_df, mods_df, prosit_df, config,
            file_idx, scan_file, max_scan, True)
            for file_idx, scan_file in enumerate(scan_files)
        ]
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(process_single_file, func_args)

        if os.path.exists(f'{config.output_folder}/input_all_features.tab'):
            os.remove(f'{config.output_folder}/input_all_features.tab')

        with open(
            f'{config.output_folder}/input_all_features.tab', 'w', encoding='UTF-8'
        ) as out_file:
            for tab_file in sorted(results):
                with open(tab_file, 'r', encoding='UTF-8') as in_file:
                    for line in in_file:
                        out_file.write(line)
                os.remove(tab_file)
    print(
        OKCYAN_TEXT + '\t\t\tFull input DataFrame written to csv.' + ENDC_TEXT
    )


def process_single_file(
        search_df, mods_df, prosit_df, config, file_idx, scan_file, max_scan, in_parallel
    ):
    """ Function to process all PSMs from a single mgf or mzML file.

    Parameters
    ----------
    search_df : pd.DataFrame
        A DataFrame of PSMs.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs present in the data.
    prosit_df : pd.DataFrame
        A DataFrame of the predicted spectra from Prosit.
    config : inspire.config.Config
        The config file for the experiment.
    file_idx : int
        The index of the file being processed.
    scan_file : str
        The name of the file being processed.
    max_scan : int
        The maximum scan number observed in the data.
    in_parallel : bool
        Whether the data is being processed sequentially or in parallel.
    """
    delta_df = None
    previous_ptm_name = None
    previous_ions_name = None
    previous_name = None
    if config.delta_method == 'bruteForce' and config.spectral_predictor == 'prosit':
        delta_predictions = f'{config.output_folder}/deltaPredictions.msp'
        delta_df = msp_to_df(delta_predictions, 'prosit', None)
        delta_df['modified_sequence'] = delta_df['modified_sequence'].apply(
            lambda x : x.replace('M(ox)', 'm')
        )
        delta_df = delta_df.rename(columns={'precursor_charge': CHARGE_KEY})
        previous_name = 'modified_sequence'
        previous_ions_name = 'prositIons'
        delta_df = delta_df[['modified_sequence', CHARGE_KEY, 'prositIons']]
        delta_df = delta_df.drop_duplicates(subset=['modified_sequence', CHARGE_KEY])
    if config.delta_method == 'bruteForce' and config.spectral_predictor == 'ms2pip':
        model = config.ms2pip_model
        delta_predictions = f'{config.output_folder}/deltaInput_{model}_predictions.msp'
        delta_df = msp_to_df(delta_predictions, 'ms2pip', mods_df)
        previous_ptm_name = PTM_SEQ_KEY
        previous_ions_name = 'prositIons'
        previous_name = 'peptide'

    ox_flag = get_ox_flag(mods_df)
    print(
        OKCYAN_TEXT +
        f'\t\tProcessing scan file {file_idx}.' +
        ENDC_TEXT
    )
    if config.combined_scans_file is not None:
        filtered_search_df = search_df
    else:
        filtered_search_df = search_df[search_df[SOURCE_KEY] == scan_file]

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
    scan_df = scan_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY])
    combined_df = combine_spectral_data(
        filtered_search_df,
        scan_df,
        prosit_df,
        ox_flag,
        config.spectral_predictor,
    )
    if config.delta_method == 'bruteForce':
        if config.spectral_predictor == 'prosit':
            combined_df['mSeq'] = combined_df['modified_sequence'].apply(
                lambda x : x.replace('M(ox)', 'm')
            )
            combined_df['permSeqs'] = combined_df['mSeq'].apply(
                lambda x : permute_seq(x, uniform_length=True)
            )
            combined_df[[f'flip{idx}' for idx in range(1, 30)]] = pd.DataFrame(
                combined_df.permSeqs.tolist(), index=combined_df.index
            )

            for i in range(1, 30):
                delta_df = delta_df.rename(columns={
                    previous_name: f'flip{i}',
                    previous_ions_name: f'flip{i}Ions',
                })
                combined_df = pd.merge(
                    combined_df,
                    delta_df,
                    how='left',
                    on=[f'flip{i}', CHARGE_KEY]
                )
                previous_name = f'flip{i}'
                previous_ions_name = f'flip{i}Ions'
        else:
            combined_df['permSeqs'] = combined_df['peptide'].apply(
                lambda x : permute_seq(x, uniform_length=True)
            )

            combined_df[[f'flip{idx}' for idx in range(1, 30)]] = pd.DataFrame(
                combined_df.permSeqs.tolist(), index=combined_df.index
            )
            combined_df['permPtms'] = combined_df[['peptide', PTM_SEQ_KEY]].apply(
                lambda x : permute_ptms(x['peptide'], x[PTM_SEQ_KEY], uniform_length=True),
                axis=1
            )
            combined_df[[f'flip{idx}Ptms' for idx in range(1, 30)]] = pd.DataFrame(
                combined_df.permPtms.tolist(), index=combined_df.index
            )

            for i in range(1, 30):
                delta_df = delta_df.rename(columns={
                    previous_name: f'flip{i}',
                    previous_ions_name: f'flip{i}Ions',
                    previous_ptm_name: f'flip{i}Ptms'
                })
                combined_df = pd.merge(
                    combined_df,
                    delta_df,
                    how='left',
                    on=[f'flip{i}', f'flip{i}Ptms', CHARGE_KEY]
                )
                previous_name = f'flip{i}'
                previous_ions_name = f'flip{i}Ions'
                previous_ptm_name = f'flip{i}Ptms'

    print(
        OKCYAN_TEXT + '\t\t\tCombined DB Search, Spectral, and Prosit Data.' + ENDC_TEXT
    )

    if not combined_df.shape[0]:
        print(
            WARNING_TEXT +
            f'Warning. No matched scans found for source file {scan_file}' +
            ENDC_TEXT
        )
        return None

    combined_df = create_spectral_features(combined_df, mods_df, config)
    combined_df = add_delta_irt(combined_df)

    print(
        OKCYAN_TEXT + '\t\t\tCreated Spectral and Delta RT Features.' + ENDC_TEXT
    )
    if config.search_engine != 'mascot':
        combined_df[PERC_SCAN_ID] = combined_df[SCAN_KEY].apply(
            lambda x, f_id=file_idx : f_id * max_scan  + x
        )
    else:
        combined_df[PERC_SCAN_ID] = combined_df[MASCOT_PEP_QUERY_KEY].apply(
            lambda x, f_id=file_idx : f_id * max_scan  + int(x.split('.mgf')[-1])
        )
    combined_df = filter_input_columns(combined_df, config, file_idx)
    combined_df = combined_df.sort_values(by=PERC_SCAN_ID)

    file_loc = _write_to_tab_file(combined_df, file_idx, config.output_folder, in_parallel)

    return file_loc

def _write_to_tab_file(combined_df, file_idx, output_folder, in_parallel):
    """ Function to write percolator input in tab format.

    Parameters
    ----------
    combined_df : pd.DataFrame
        A DataFrame of processed search results.
    file_idx : int
        The index of the scan file used.
    output_folder : str
        The folder where all inSPIRE output is written.
    """
    if in_parallel:
        filename = f'input_all_features{file_idx}.tab'
    else:
        filename = 'input_all_features.tab'
    if file_idx == 0:
        combined_df.to_csv(
            f'{output_folder}/{filename}',
            sep='\t',
            index=False,
        )
    else:
        if in_parallel:
            write_mode = 'w'
        else:
            write_mode = 'a'
        combined_df.to_csv(
            f'{output_folder}/{filename}',
            sep='\t',
            index=False,
            mode=write_mode,
            header=False,
        )
    return f'{output_folder}/{filename}'

def write_rescoring_features(
        search_df,
        mods_df,
        config,
    ):
    """ Function to write the percolator/mokapot input DataFrame.

    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.
    """
    feature_df = create_basic_features(search_df, mods_df)

    if config.use_binding_affinity == 'asFeature':
        mhc_pan_df = read_mhcpan_output(f'{config.output_folder}/mhcpan')
        mhc_pan_df = mhc_pan_df.rename(columns={
            'Peptide': PEPTIDE_KEY,
            'Aff(nM)': 'bindingAffinity'
        })
        mhc_pan_df['bindingAffinity'] = mhc_pan_df['bindingAffinity'].apply(log10)
        mhc_pan_df = mhc_pan_df[[PEPTIDE_KEY, 'bindingAffinity']]
        feature_df = pd.merge(
            feature_df,
            mhc_pan_df,
            on=PEPTIDE_KEY,
            how='inner'
        )

    psm_id_key = PSM_ID_KEY[config.rescore_method]
    if mods_df.empty:
        mod_weights = {}
    else:
        mod_weights = dict(zip(mods_df[PTM_ID_KEY].tolist(), mods_df[PTM_WEIGHT_KEY].tolist()))
    feature_df['modSeq'] = feature_df.apply(
        lambda x : modify_sequence_for_skyline(x, mod_weights),
        axis=1
    )

    feature_df[psm_id_key] = feature_df[[SOURCE_KEY, SCAN_KEY, 'modSeq']].apply(
        lambda x : str(x[SOURCE_KEY]) + '_' + str(x[SCAN_KEY]) + '_' + str(x['modSeq']),
        axis=1
    )
    print(
        OKCYAN_TEXT +
        '\tBasic Features added, adding Spectral Features.' +
        ENDC_TEXT
    )

    write_with_spectral_features(
        feature_df,
        mods_df,
        config,
    )

def get_mod_score(df_row):
    """ Helper function to compare different accession strata.
    """
    if df_row[ACCESSION_STRATUM_KEY] == 0:
        return df_row[ENGINE_SCORE_KEY]
    return 0.7*df_row[ENGINE_SCORE_KEY]

def check_bad_mods(ptm_str, bad_mods):
    """ Function to check for the presence of ptms other than oxidation of methionine
        or carbamidomehtylation of Cysteine.
    """
    if not isinstance(ptm_str, str):
        return False

    for mod_id in ptm_str:
        if mod_id in bad_mods:
            return True
    return False

def process_unknown_modifications(target_df, mods_df, config):
    """ Function to handle modifications which are unknown to the Prosit spectral predictor
        (as well as unmodified cysteines).

    Parameters
    ----------
    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.

    Returns
    -------
    target_df : pd.DataFrame
        The results from the original search DataFrame with unknown modifications filtered.
    """
    if config.drop_unknown_mods:
        unknown_mods = mods_df[
            (mods_df[PTM_NAME_KEY] != 'Oxidation (M)') &
            (mods_df[PTM_NAME_KEY] != 'Carbamidomethylation') &
            (mods_df[PTM_NAME_KEY] != 'Carbamidomethyl (C)')
        ][PTM_ID_KEY].tolist()
        unknown_mods = {str(x) for x in unknown_mods}
        target_df['unknownModifications'] = target_df[PTM_SEQ_KEY].apply(
            lambda x : check_bad_mods(x, unknown_mods)
        )
        count_before_drop = target_df.shape[0]
        if config.use_accession_stratum:
            target_df['modEngineScore'] = target_df[
                [ENGINE_SCORE_KEY, ACCESSION_STRATUM_KEY]
            ].apply(get_mod_score, axis=1)
            target_df['maxModScore'] = target_df.groupby(
                [SOURCE_KEY, SCAN_KEY]
            )['modEngineScore'].transform(max)

            unknown_df = target_df[
                (target_df['unknownModifications']) &
                (target_df['maxModScore'] == target_df['modEngineScore'])
            ]
            unknown_df = unknown_df[[SOURCE_KEY, SCAN_KEY]].drop_duplicates()
            unknown_df['drop'] = 'yes'
            target_df = target_df[
                ~target_df['unknownModifications']
            ].drop(['modEngineScore', 'maxModScore', 'unknownModifications'], axis=1)

            target_df = pd.merge(
                target_df,
                unknown_df,
                how='left',
                on=[SOURCE_KEY, SCAN_KEY]
            )
            target_df = target_df[target_df['drop'] != 'yes']
            target_df = target_df.drop('drop', axis=1)
        else:
            target_df = target_df[~target_df['unknownModifications']].drop(
                ['unknownModifications'], axis=1
            )
        count_after_drop = target_df.shape[0]
        filtered_psms = count_before_drop - count_after_drop
        print(
            OKCYAN_TEXT +
            f'\tFiltered {filtered_psms} PSMs due to modifications unknown to Prosit.' +
            ENDC_TEXT
        )

    if not mods_df[
            (mods_df[PTM_NAME_KEY] == 'Carbamidomethylation') |
            (mods_df[PTM_NAME_KEY] == 'Carbamidomethyl (C)')
        ].shape[0] and config.filter_c:
        count_before_drop = target_df.shape[0]
        target_df = target_df[target_df[PEPTIDE_KEY].apply(lambda x : 'C' not in x)]
        count_after_drop = target_df.shape[0]
        filtered_psms = count_before_drop - count_after_drop
        print(
            OKCYAN_TEXT +
            f'\tFiltered {filtered_psms} PSMs due to unmodified cysteines.' +
            ENDC_TEXT
        )

    return target_df

def create_features(config):
    """ Function to create features for percolator/mokapot input.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used throughout the pipeline.
    """
    target_df, mods_df = generic_read_df(config)

    if config.use_accession_stratum:
        target_df = process_accession_groups(target_df, config)

    target_df = process_unknown_modifications(target_df, mods_df, config)

    n_variable_mods = mods_df.shape[0]  # pylint: disable=no-member
    if  n_variable_mods > 9:
        raise ValueError(
            f'inSPIRE supports no more than 9 unique PTMs, found {n_variable_mods}'
        )

    print(
        OKCYAN_TEXT +
        '\tMS Search Results ready.' +
        ENDC_TEXT
    )
    write_rescoring_features(
        target_df,
        mods_df,
        config,
    )
