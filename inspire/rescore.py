""" Functions for rescoring PSMs with an optimised feature set.
"""
import subprocess

import pandas as pd

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    CHARGE_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    FINAL_Q_VALUE_KEY,
    FINAL_SCORE_KEY,
    OKCYAN_TEXT,
    OUT_ACCESSION_KEY,
    OUT_Q_KEY,
    OUT_SCORE_KEY,
    PEPTIDE_KEY,
    PSM_ID_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)

def apply_rescoring(
        output_folder,
        input_filename,
        fdr,
        rescore_method,
        output_prefix,
    ):
    """ Function to apply percolator and return the PSMs matched.

    Parameters
    ----------
    perc_input_df : pd.DataFrame
        A DataFrame read for percolator input.
    output_folder : str
        The folder in which all output for the pipeline is written.
    fdr : float
        The False Positive Rate with which to train percolator (0.0-1.0)
    output_filename : str or None
        A specific filename for the output psms, defaults
    output_weights : bool
        Flag indicating whether or not to output the feature weights.

    Returns
    -------
    results : pd.DataFrame
        The predictions from Percolator.
    """
    psm_output_key = f'{output_folder}/{output_prefix}.{rescore_method}.psms.txt'
    pep_output_key = f'{output_folder}/{output_prefix}.{rescore_method}.peptides.txt'

    if rescore_method == 'mokapot':
        rescore_name = 'mokapot'
        clis = (
            f' --dest_dir {output_folder} --keep_decoys --verbosity 0 ' +
            f' --train_fdr {fdr} -v 0 ' +
            f' --test_fdr {fdr} --file_root {output_prefix} --save_models '
        )
        trailing_args = ''
    elif rescore_method == 'percolatorSeparate':
        rescore_name = 'percolator'
        percolator_decoy_key = f'{output_folder}/{output_prefix}.{rescore_method}.decoy.psms.txt'
        weights_path = f'{output_folder}/{output_prefix}.{rescore_method}.weights.csv'
        clis = (
            f' -U -F {fdr} -t {fdr} -i 10 -M {percolator_decoy_key} --post-processing-tdc  ' +
            f' -w {weights_path} --override ' +
            f' --results-psms {psm_output_key} --results-peptides {pep_output_key} '
        )
        trailing_args = ''
    else:
        rescore_name = 'percolator'
        percolator_decoy_key = f'{output_folder}/{output_prefix}.{rescore_method}.decoy.psms.txt'
        weights_path = f'{output_folder}/{output_prefix}.{rescore_method}.weights.csv'
        clis = (
            f' -F {fdr} -t {fdr} -i 10 -M {percolator_decoy_key} --post-processing-tdc ' +
            f' -I concatenated -w {weights_path} -v 0 ' +
            f' --results-psms {psm_output_key} --results-peptides {pep_output_key} --override '
        )
        trailing_args = ''

    bash_command = (
        f'{rescore_name} {clis} {output_folder}/{input_filename} {trailing_args}'
    )

    with open(f'{output_folder}/rescore.log', 'w', encoding='UTF-8') as log_file:
        subprocess.run(
            bash_command,
            check=True,
            shell=True,
            stdout=log_file,
        )

    results = pd.read_csv(psm_output_key, sep='\t')

    return results

def _split_psm_ids(df_row, psm_id_key):
    """ Function for splitting a PSM Id back into its source name, scan number
        and peptide sequence.

    Parameters
    ----------
    df_row : pd.Series
        A row of the DataFrame to which the function is being applied.

    Parameters
    ----------
    df_row : pd.Series
        The same row with source and scan added.
    """
    source_scan_list = df_row[psm_id_key].split('_')
    df_row['modifiedSequence'] = source_scan_list[-1]
    df_row[SCAN_KEY] = int(source_scan_list[-2])
    df_row[SOURCE_KEY] = '_'.join(source_scan_list[:-2])
    return df_row

def _regroup_accession(df_row, acc_cols):
    """ Helper function to remove one hot encoding from Accession Stratum.

    Parameters
    ----------
    df_row : pd.Series
        A row of the final results DataFrame.
    acc_cols : list of str
        All of the accession related columns.

    Returns
    -------
    acc_stratum : str
        The final accession stratum listed in the results.
    """
    for acc_col in acc_cols:
        if df_row[acc_col] == 1:
            return acc_col.split('_')[1]
    return 'unknown'

def _add_key_features(target_psms, config):
    """ Function to add spectral angle and engine score back to percolator
        output PSMs.

    Parameters
    ----------
    target_psms : pd.DataFrame
        A DataFrame of percolator output PSMs.
    config : inspire.config.Config
        The config object used throughout the pipeline.

    Returns
    -------
    output_df : pd.DataFrame
        The output PSMs labelled with original search engine score and
        spectral angle.
    """
    psm_id_key = PSM_ID_KEY[config.rescore_method]

    input_key = f'{config.output_folder}/input_all_features.tab'
    input_df = pd.read_csv(
        input_key,
        sep='\t',
    )
    input_df = input_df.drop_duplicates(subset=[psm_id_key, PEPTIDE_KEY])

    key_features = [SPECTRAL_ANGLE_KEY, 'matchedCoverage', 'deltaRT', ENGINE_SCORE_KEY, CHARGE_KEY]
    if config.use_accession_stratum:
        acc_cols = [x for x in input_df.columns if x.startswith('accession')]
        input_df[ACCESSION_STRATUM_KEY] = input_df.apply(
            lambda x : _regroup_accession(x, acc_cols), axis=1
        )
        key_features.append(ACCESSION_STRATUM_KEY)

    output_df = pd.merge(
        target_psms,
        input_df[[psm_id_key, PEPTIDE_KEY] + key_features],
        how='inner',
        on=[psm_id_key, PEPTIDE_KEY]
    )

    return output_df, key_features

def final_rescoring(config):
    """ Function to rescore PSMs using the final feature set.

    Parameters
    ----------
    config : inspire.config.Config
        The config object used throughout the pipeline.
    """
    out_score_key = OUT_SCORE_KEY[config.rescore_method]
    out_q_key = OUT_Q_KEY[config.rescore_method]
    psm_id_key = PSM_ID_KEY[config.rescore_method]
    out_accession_key = OUT_ACCESSION_KEY[config.rescore_method]

    in_path = 'final_input.tab'
    output_prefix = 'final'

    target_psms = apply_rescoring(
        config.output_folder,
        in_path,
        config.fdr,
        config.rescore_method,
        output_prefix,
    )

    if 'PSMId' in target_psms.columns:
        target_psms = target_psms.rename(  # pylint: disable=no-member
            columns={'PSMId': psm_id_key}
        )

    print(
        OKCYAN_TEXT + '\tRescoring complete.' + ENDC_TEXT
    )
    output_df, key_features = _add_key_features(target_psms, config)

    output_df = output_df.apply(
       lambda x: _split_psm_ids(x, psm_id_key),
       axis=1
    ).drop(psm_id_key, axis=1)

    output_df = output_df.rename(
        columns={
            out_score_key: FINAL_SCORE_KEY,
            out_q_key: FINAL_Q_VALUE_KEY,
            out_accession_key: ACCESSION_KEY,
        }
    )
    final_columns = (
        [
            SOURCE_KEY,
            SCAN_KEY,
            PEPTIDE_KEY,
            'modifiedSequence',
            FINAL_SCORE_KEY,
            FINAL_Q_VALUE_KEY,
        ] + key_features +
        [
            ACCESSION_KEY
        ]
    )

    output_df = output_df[final_columns]

    output_df = output_df.sort_values(by=FINAL_SCORE_KEY, ascending=False)

    out_path = f'{config.output_folder}/finalAssignments.csv'

    output_df.to_csv(out_path, index=False)
    print(
        OKCYAN_TEXT + '\tFinal assignments written to csv.' + ENDC_TEXT
    )
