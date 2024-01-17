""" Functions for rescoring PSMs with an optimised feature set.
"""
from pathlib import Path
import subprocess

import pandas as pd
import polars as pl

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    CHARGE_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    FINAL_POSTEP_KEY,
    FINAL_Q_VALUE_KEY,
    FINAL_SCORE_KEY,
    OKCYAN_TEXT,
    OUT_ACCESSION_KEY,
    OUT_POSTEP_KEY,
    OUT_Q_KEY,
    OUT_SCORE_KEY,
    PEPTIDE_KEY,
    PSM_ID_KEY,
    RT_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.input.mhcpan import read_mhcpan_output
from inspire.utils import fetch_proteome, parallel_remap

def apply_rescoring(
        output_folder,
        input_filename,
        fdr,
        rescore_method,
        output_prefix,
        rescore_command,
        proteome=None,
        decoy_prot_key='rev_',
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
    prot_out_key = f'{output_folder}/{output_prefix}.{rescore_method}.proteins.txt'

    if rescore_method == 'mokapot':
        clis = (
            f' --dest_dir {output_folder} --keep_decoys  ' +
            f' --train_fdr {fdr} ' +
            f' --test_fdr {fdr} --file_root {output_prefix} --save_models '
        )
        trailing_args = ''
    elif rescore_method == 'percolatorSeparate':
        percolator_decoy_key = f'{output_folder}/{output_prefix}.{rescore_method}.decoy.psms.txt'
        weights_path = f'{output_folder}/{output_prefix}.{rescore_method}.weights.csv'
        clis = (
            f' -F {fdr} -t {fdr} -i 10 -M {percolator_decoy_key} --post-processing-tdc  ' +
            f' -w {weights_path} ' +
            f' --results-psms {psm_output_key} --results-peptides {pep_output_key} '
        )
        trailing_args = ''
    else:
        percolator_decoy_key = f'{output_folder}/{output_prefix}.{rescore_method}.decoy.psms.txt'
        weights_path = f'{output_folder}/{output_prefix}.{rescore_method}.weights.csv'
        clis = (
            f' -F {fdr} -t {fdr} -i 10 -M {percolator_decoy_key} --post-processing-tdc ' +
            f' -I concatenated -w {weights_path} ' +
            f' --results-psms {psm_output_key} --results-peptides {pep_output_key} '
        )
        trailing_args = ''

    if proteome is not None:
        trailing_args += (
            f' -f {proteome} -z no_enzyme --results-proteins {prot_out_key} -P {decoy_prot_key}'
        )

    bash_command = (
        f'"{rescore_command}" {clis} {output_folder}/{input_filename} {trailing_args}'
    )

    with open(f'{output_folder}/rescore.log', 'w', encoding='UTF-8') as log_file:
        subprocess.run(
            bash_command,
            check=True,
            shell=True,
            stdout=log_file,
        )

    psm_results_df = pl.read_csv(psm_output_key, separator='\t')
    psm_results_df = psm_results_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(lambda x : x[2:-2])
    )

    peptide_results_df = pl.read_csv(pep_output_key, separator='\t')
    peptide_results_df = peptide_results_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(lambda x : x[2:-2])
    )

    return psm_results_df, peptide_results_df

def _split_psm_ids(psm_id):
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
    results = {}
    source_scan_list = psm_id.split('_')
    results['modifiedSequence'] = source_scan_list[-1]
    results[SCAN_KEY] = int(source_scan_list[-2])
    results[SOURCE_KEY] = '_'.join(source_scan_list[:-2])
    return results

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
    input_df = pl.from_pandas(pd.read_csv(
        input_key,
        sep='\t',
    ))
    input_df = input_df.unique(subset=[psm_id_key, PEPTIDE_KEY])

    key_features = [
        SPECTRAL_ANGLE_KEY,
        'spearmanR',
        'predCoverage',
        'matchedCoverage',
        RT_KEY,
        'deltaRT',
        ENGINE_SCORE_KEY,
        CHARGE_KEY,
    ]
    if config.use_accession_stratum:
        acc_cols = [
            x for x in input_df.columns if x.startswith('accession') and x != 'accessionGroup'
        ]
        input_df = input_df.with_columns(
            pl.struct(acc_cols).apply(
                lambda x : _regroup_accession(x, acc_cols)
            ).alias(ACCESSION_STRATUM_KEY)
        )

        key_features.append(ACCESSION_STRATUM_KEY)

    if config.use_binding_affinity is not None:
        mhc_pan_df = read_mhcpan_output(
            f'{config.output_folder}/mhcpan',
            per_allele=True,
        )
        mhc_pan_df = mhc_pan_df.rename({
            'Peptide': PEPTIDE_KEY,
        })
        mhc_pan_cols = (
            [x for x in mhc_pan_df.columns if (
                x.endswith('Affinity') or x.endswith('Level') or x.endswith('%Rank_BA')
            )]
        )

        target_psms = target_psms.join(
            mhc_pan_df.select([PEPTIDE_KEY] + mhc_pan_cols),
            on=PEPTIDE_KEY,
            how='left',
        )

    if isinstance(config.collision_energy, list):
        key_features.append('collisionEnergy')

    output_df = target_psms.join(
        input_df[[psm_id_key, PEPTIDE_KEY] + key_features],
        how='inner',
        on=[psm_id_key, PEPTIDE_KEY]
    )

    if config.use_binding_affinity is not None:
        for col in mhc_pan_cols:
            if col.endswith('Affinity') or col.endswith('%Rank_BA'):
                output_df = output_df.with_columns(
                    pl.col(col).fill_null(
                        pl.lit(-1),
                    ).alias(col),
                )
            else:
                output_df = output_df.with_columns(
                    pl.col(col).apply(
                        lambda x : 'Strong-Binder' if x == '<=SB' else (
                            'Weak-Binder' if x == '<=WB' else (
                                'Not predicted' if x is None else
                            'Non-Binder')
                        ),
                        skip_nulls=False,
                    )
                )
        key_features.extend(mhc_pan_cols)

    return output_df, key_features

def final_rescoring(config):
    """ Function to rescore PSMs using the final feature set.

    Parameters
    ----------
    config : inspire.config.Config
        The config object used throughout the pipeline.
    """
    in_path = 'final_input.tab'
    output_prefix = 'final'

    if config.infer_proteins:
        proteome = config.proteome
    else:
        proteome = None

    target_psms, target_peptides = apply_rescoring(
        config.output_folder,
        in_path,
        config.fdr,
        config.rescore_method,
        output_prefix,
        config.rescore_command,
        proteome,
        decoy_prot_key=config.decoy_protein_flag,
    )

    print(
        OKCYAN_TEXT + '\tRescoring complete.' + ENDC_TEXT
    )

    output_peptides_df = apply_post_processing(target_peptides, config)
    output_psm_df = apply_post_processing(target_psms, config)

    if config.accession_format == 'invitroSPI':
        output_peptides_df.write_csv(f'{config.output_folder}/pre_finalAssignments.csv')
    else:
        output_peptides_df.write_csv(f'{config.output_folder}/finalPeptideAssignments.csv')

    output_psm_df.write_csv(f'{config.output_folder}/finalPsmAssignments.csv')

    print(
        OKCYAN_TEXT + '\tFinal assignments written to csv.' + ENDC_TEXT
    )


def apply_post_processing(target_psms, config):
    """ Function to separate columns and re-add relevant information.
    """
    out_score_key = OUT_SCORE_KEY[config.rescore_method]
    out_q_key = OUT_Q_KEY[config.rescore_method]
    out_postep_key = OUT_POSTEP_KEY[config.rescore_method]
    psm_id_key = PSM_ID_KEY[config.rescore_method]
    out_accession_key = OUT_ACCESSION_KEY[config.rescore_method]

    if 'PSMId' in target_psms.columns:
        target_psms = target_psms.rename({'PSMId': psm_id_key})


    output_df, key_features = _add_key_features(target_psms, config)

    output_df = output_df.with_columns(
        pl.col(psm_id_key)
        .apply(_split_psm_ids, skip_nulls=False)
        .alias("results")
    ).unnest("results")

    output_df = output_df.rename({
        out_score_key: FINAL_SCORE_KEY,
        out_q_key: FINAL_Q_VALUE_KEY,
        out_postep_key: FINAL_POSTEP_KEY,
        out_accession_key: ACCESSION_KEY,
        out_postep_key: FINAL_POSTEP_KEY,
    })

    final_columns = (
        [
            SOURCE_KEY,
            SCAN_KEY,
            PEPTIDE_KEY,
            'modifiedSequence',
            FINAL_SCORE_KEY,
            FINAL_Q_VALUE_KEY,
            FINAL_POSTEP_KEY,
        ] + key_features
    )

    if config.map_contaminants is not None:
        if config.map_contaminants == 'standard':
            home = str(Path.home())
            contamns_path =f'{home}/inSPIRE_models/utilities/contaminants_20120713.fasta'
        else:
            contamns_path = config.map_contaminants


        if config.proteome is not None:
            target_proteome = fetch_proteome(config.proteome, with_desc=False)
            output_df = parallel_remap(
                output_df,
                config.n_cores,
                target_proteome,
                'mapsToTarget',
                trace_accession=False,
            )

            contams_proteome = fetch_proteome(contamns_path, with_desc=False)
            output_df = parallel_remap(
                output_df,
                config.n_cores,
                contams_proteome,
                'mapsToContaminant',
                trace_accession=False,
            )
        if config.proteome is not None:
            final_columns.extend(['mapsToTarget', 'mapsToContaminant'])


    final_columns.append(ACCESSION_KEY)
    output_df = output_df.select(final_columns)
    output_df = output_df.sort(by=[FINAL_SCORE_KEY, SOURCE_KEY, SCAN_KEY], descending=True)

    return output_df
