""" Functions for writing percolator/mokapot input using Prosit and other features.
"""
import re

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
    KNOWN_PTM_LOC,
    KNOWN_PTM_WEIGHTS,
    LABEL_KEY,
    OKCYAN_TEXT,
    PEPTIDE_KEY,
    PSM_ID_KEY,
    PERC_SCAN_ID,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    SCAN_KEY,
    SOURCE_INDEX_KEY,
    SOURCE_KEY,
    WARNING_TEXT,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.mhcpan import read_mhcpan_output
from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.input.search_results import generic_read_df
from inspire.prepare import create_prosit_mod_seq
from inspire.retention_time import add_delta_irt
from inspire.spectral_features import SPECTRAL_FEATURES, DELTA_FEATURES, create_spectral_features
from inspire.utils import get_ox_flag, modify_sequence_for_skyline, remove_source_suffixes

def combine_spectral_data(search_df, scan_df, prosit_df, ox_flag):
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
    search_df['modified_sequence'] = search_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
        lambda x : create_prosit_mod_seq(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], ox_flag),
        axis=1
    )
    print(
        OKCYAN_TEXT +
        f'\t\t\t{search_df.shape[0]} in original search results.' +
        ENDC_TEXT
    )
    prosit_df = prosit_df.drop_duplicates(
        subset=['modified_sequence', CHARGE_KEY, 'collisionEnergy']
    )
    df_with_prosit = pd.merge(
        search_df,
        prosit_df,
        how='inner',
        on=['modified_sequence', CHARGE_KEY]
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

    combined_df = combined_df[use_cols]

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
    prosit_predictions = f'{config.output_folder}/prositPredictions.msp'
    prosit_df = msp_to_df(prosit_predictions)
    if config.combined_scans_file is not None:
        scan_files = [remove_source_suffixes(config.combined_scans_file)]
    else:
        scan_files = search_df[SOURCE_KEY].unique().tolist()

    ox_flag = get_ox_flag(mods_df)

    max_scan = search_df[SCAN_KEY].max()

    for file_idx, scan_file in enumerate(scan_files):
        print(
            OKCYAN_TEXT +
            f'\t\tProcessing scan file {file_idx+1} of {len(scan_files)}' +
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
                scans.tolist()
            )
        else:
            scan_df = process_mgf_file(
                f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                scans.tolist(),
                config.scan_title_format,
                config.source_files
            )
        scan_df = scan_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY])
        combined_df = combine_spectral_data(
            filtered_search_df,
            scan_df,
            prosit_df,
            ox_flag
        )

        print(
            OKCYAN_TEXT + '\t\t\tCombined DB Search, Spectral, and Prosit Data.' + ENDC_TEXT
        )

        if not combined_df.shape[0]:
            print(
                WARNING_TEXT +
                f'Warning. No matched scans found for source file {scan_file}' +
                ENDC_TEXT
            )
            continue

        combined_df = create_spectral_features(combined_df, mods_df, config.mz_accuracy)
        combined_df = add_delta_irt(combined_df)

        print(
            OKCYAN_TEXT + '\t\t\tCreated Spectral and Delta RT Features.' + ENDC_TEXT
        )

        combined_df[PERC_SCAN_ID] = combined_df[SCAN_KEY].apply(
            lambda x, f_id=file_idx : f_id * max_scan  + x
        )

        combined_df = filter_input_columns(combined_df, config, file_idx)
        combined_df = combined_df.sort_values(by=PERC_SCAN_ID)

        _write_to_tab_file(combined_df, file_idx, config.output_folder)

        print(
            OKCYAN_TEXT + '\t\t\tFull input DataFrame written to csv.' + ENDC_TEXT
        )

def _write_to_tab_file(combined_df, file_idx, output_folder):
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
    if file_idx == 0:
        combined_df.to_csv(
            f'{output_folder}/input_all_features.tab',
            sep='\t',
            index=False,
        )
    else:
        combined_df.to_csv(
            f'{output_folder}/input_all_features.tab',
            sep='\t',
            index=False,
            mode='a',
            header=False,
        )


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
    feature_df = create_basic_features(search_df, mods_df, config.digest)

    if config.use_binding_affinity == 'asFeature':
        mhc_pan_df = read_mhcpan_output(f'{config.output_folder}/mhcpan')
        mhc_pan_df = mhc_pan_df.rename(columns={
            'Peptide': PEPTIDE_KEY,
            'Aff(nM)': 'bindingAffinity'
        })
        mhc_pan_df = mhc_pan_df[[PEPTIDE_KEY, 'bindingAffinity']]
        feature_df = pd.merge(
            feature_df,
            mhc_pan_df,
            on=PEPTIDE_KEY,
            how='left'
        )

    psm_id_key = PSM_ID_KEY[config.rescore_method]
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

def create_features(config):
    """ Function to create features for percolator/mokapot input.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used throughout the pipeline.
    """
    target_df, mods_df = generic_read_df(config)
    target_df = target_df[target_df[SOURCE_KEY].apply(lambda x : x not in config.exclude_raw_files)]

    if config.fixed_modifications is not None:
        target_df, mods_df = add_fixed_modifications(
            target_df,
            mods_df,
            config.fixed_modifications
        )

    if config.use_accession_stratum:
        target_df = process_accession_groups(target_df, config)

    if config.drop_unknown_mods:
        unknown_mods = mods_df[
            (mods_df[PTM_NAME_KEY] != 'Oxidation (M)') &
            ((mods_df[PTM_NAME_KEY] != 'Carbamidomethyl (C)'))
        ][PTM_ID_KEY].tolist()
        unknown_mods = [str(x) for x in unknown_mods]
        if config.use_accession_stratum:
            target_df['modEngineScore'] = target_df[
                [ENGINE_SCORE_KEY, ACCESSION_STRATUM_KEY]
            ].apply(get_mod_score, axis=1)
            target_df['maxModScore'] = target_df.groupby(
                [SOURCE_KEY, SCAN_KEY]
            )['modEngineScore'].transform(max)
            target_df['unknownModifications'] = target_df[PTM_SEQ_KEY].apply(
                lambda x : check_bad_mods(x, unknown_mods)
            )

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


    if config.filter_c:
        target_df = target_df[target_df[PEPTIDE_KEY].apply(lambda x : 'C' not in x)]

    if mods_df.shape[0] > 9:
        raise ValueError(f'inSPIRE supports no more than 9 unique PTMs, found {mods_df.shape[0]}')

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

def _modify_ptm_seq(pep_seq, ptm_seq, ptm_locs, ptm_id):
    """ Helper function to add a fixed PTM to the ptm_seq column of MS search
        result DataFrame.

    Parameters
    ----------
    pep_seq : str
        The peptide sequence.
    ptm_seq : str
        The ptm sequence.
    ptm_locs : str
        The location(s) where the ptm should be added.
    ptm_id : str
        The id of the ptm.

    Returns
    -------
    ptm_seq : str
        The modified ptm sequence.
    """
    if ptm_locs == 'N-term':
        if ptm_seq is None:
            return ptm_id + '.' + ''.join(['0']*len(pep_seq)) + '.0'
        return ptm_id + ptm_seq[1:]

    for loc in ptm_locs:
        edit_points = [m.start() for m in re.finditer(loc, pep_seq)]
        if edit_points and ptm_seq is None:
            ptm_seq = '0.' + ''.join(['0']*len(pep_seq)) + '.0'
        for e_point in edit_points:
            ptm_seq = ptm_seq[:e_point+2] + ptm_id + ptm_seq[e_point+3:]

    return ptm_seq

def add_fixed_modifications(search_df, mods_df, fixed_modifications):
    """ Function to add fixed modifications to the search data.

    Parameters
    ----------
    search_df : pd.DataFrame
        The DataFrame of MS search results from either MaxQuant or PEAKS.
    mods_df : pd.DataFrame
        The DataFrame of variable modifications in the search results.
    fixed_modifications : list
        A list of the fixed modifications found in the DataFrame.

    Returns
    -------
    mods_df : pd.DataFrame
        A DataFrame of all modifications found (both variable and fixed).
    """
    ptm_ids = []
    ptm_names = []
    ptm_weights = []
    for idx, modification in enumerate(fixed_modifications):
        if modification in mods_df[PTM_NAME_KEY]:
            continue

        ptm_weight = KNOWN_PTM_WEIGHTS.get(modification)
        ptm_idx = mods_df.shape[0]+idx+1
        if ptm_weight is None:
            raise ValueError(f'Unsupported fixed modification {modification}.')

        ptm_ids.append(ptm_idx)
        ptm_names.append(modification)
        ptm_weights.append(ptm_weight)
        str_ptm_idx = str(ptm_idx)
        search_df[PTM_SEQ_KEY] = search_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
            lambda x, mod=modification, ptm_id=str_ptm_idx : _modify_ptm_seq(
                x[PEPTIDE_KEY],
                x[PTM_SEQ_KEY],
                KNOWN_PTM_LOC[mod],
                ptm_id
            ),
            axis=1
        )

    fixed_ptm_df = pd.DataFrame({
        PTM_ID_KEY: ptm_ids,
        PTM_NAME_KEY: ptm_names,
        PTM_WEIGHT_KEY: ptm_weights,
        PTM_IS_VAR_KEY: [False]*len(fixed_modifications)
    })
    mods_df = pd.concat([
        mods_df[[PTM_ID_KEY, PTM_NAME_KEY, PTM_WEIGHT_KEY, PTM_IS_VAR_KEY]],
        fixed_ptm_df]
    )

    return search_df, mods_df
