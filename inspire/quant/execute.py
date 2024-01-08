""" Functions for running Skyline quantification via a Docker image.
"""
import os
import shutil

import docker
import pandas as pd
import platform
import polars as pl

from inspire.constants import (
    ACCESSION_KEY,
    CHARGE_KEY,
    MOD_SEQ_KEY,
    PEPTIDE_KEY,
    FINAL_Q_VALUE_KEY,
    SOURCE_KEY,
)
from inspire.download import download_utils
from inspire.quant.utils import plot_correlations
from inspire.utils import is_control

SKYLINE_INTERMEDIATE_FILES = [
    'inspire_identifications.fasta',
    'inspire_identifications.ssl',
    'skyline_config.blib',
    'skyline_config.redundant.blib',
    'skyline_config.sky',
    'skyline_config.skyd',
    'skyline_config.slc',
    'skyline_report.csv',
    'skyline_report_template.skyr',
]
SKYLINE_AREA_KEY = 'TotalAreaMs1'
SKYLINE_BACKGROUND_KEY = 'TotalBackgroundMs1'

SKYLINE_INPUT_FILE_KEY = 'file'
SKYLINE_INPUT_SCAN_KEY = 'scan'
SKYLINE_INPUT_CHARGE_KEY = 'charge'
SKYLINE_INPUT_PEP_KEY = 'sequence'
SKYLINE_INPUT_SCORE_KEY = 'score'
SKYLINE_INPUT_KEYS = [
    SKYLINE_INPUT_FILE_KEY,
    SKYLINE_INPUT_SCAN_KEY,
    SKYLINE_INPUT_CHARGE_KEY,
    SKYLINE_INPUT_PEP_KEY,
    SKYLINE_INPUT_SCORE_KEY,
]

SKYLINE_OUTPUT_CHARGE_KEY = 'PrecursorCharge'
SKYLINE_OUTPUT_PEP_KEY = 'PeptideSequence'
SKYLINE_OUTPUT_FILE_KEY = 'FileName'
SKYLINE_OUTPUT_IDP_KEY = 'IsotopeDotProduct'


def quantify_identifications(config):
    """ Function to run Skyline quantification via the Docker image and format the output.

    Parameters
    ----------
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    # Download necessary config files if necessary
    download_utils()

    # Prepare inputs
    create_skyline_input(config)
    copy_skyline_files_to_scans_dir(config)

    # Execute quantification
    execute_skyline(config)

    # Format output, add metadata.
    format_skyline_output(config)

    # Plot correlations between raw results.
    plot_correlations('quantified_per_file', 'raw', config)

def create_skyline_input(config):
    """ Function to format the inSPIRE identifications for Skyline.

    Parameters
    ----------
    config : inspire.config.Config
        Config object which controls the experiment.
    """
    output_df = pl.read_csv(f'{config.output_folder}/finalPsmAssignments.csv')
    output_df = output_df.with_columns(
        pl.struct([ACCESSION_KEY, PEPTIDE_KEY]).apply(
            lambda df_row : (
                f'>{df_row[ACCESSION_KEY]}_{df_row[PEPTIDE_KEY]}\n{df_row[PEPTIDE_KEY]}\n'
            )
        ).alias('fastaEntry')
    )

    # Write all peptides to the fasta file
    with open(
        f'{config.output_folder}/quant/inspire_identifications.fasta',
        mode='w',
        encoding='UTF-8',
    ) as out_fasta:
        output_df['fastaEntry'].apply(
            out_fasta.write
        )

    # Filter identifications
    quant_df = output_df.filter(
        pl.col(FINAL_Q_VALUE_KEY).lt(config.quantification_cut_off)
    )

    # Format for Skyline
    quant_df = quant_df.rename({
        MOD_SEQ_KEY: SKYLINE_INPUT_PEP_KEY,
    })
    quant_df = quant_df.with_columns(
        (pl.col(SOURCE_KEY) + '.raw').alias(SKYLINE_INPUT_FILE_KEY),
        pl.lit(0.001).alias(SKYLINE_INPUT_SCORE_KEY),
    )

    # Write identifications for Skyline.
    quant_df.select(
        SKYLINE_INPUT_KEYS
    ).write_csv(
        f'{config.output_folder}/quant/inspire_identifications.ssl',
        separator='\t',
    )


def copy_skyline_files_to_scans_dir(config):
    """ Function to copy the required Skyline files to the scans folder.

    Parameters
    ----------
    config : inspire.config.Config
        Config object which controls the experiment.
    """
    shutil.copyfile(
        f'{config.output_folder}/quant/inspire_identifications.fasta',
        f'{config.scans_folder}/inspire_identifications.fasta',
    )
    shutil.copyfile(
        f'{config.output_folder}/quant/inspire_identifications.ssl',
        f'{config.scans_folder}/inspire_identifications.ssl',
    )
    shutil.copyfile(
        config.skyline_config_file,
        f'{config.scans_folder}/skyline_config.sky',
    )
    shutil.copyfile(
        config.skyline_report_template,
        f'{config.scans_folder}/skyline_report_template.skyr',
    )


def execute_skyline(config):
    """ Function to execute Skyline via the docker image.

    Parameters
    ----------
    config : inspire.config.Config
        Config object for the whole experiment.
    scans_folder_abs_path : str
        The absolute path to the folder containing scan data (where inSPIRE
        has copied the other data required by Skyline).
    """
    scans_folder_abs_path = os.path.abspath(config.scans_folder)

    # Execute docker
    if platform.system() != 'Windows':
        client = docker.from_env()
        client.containers.run(
            'proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses',
            ' wine SkylineCmd --timestamp --dir=/data --in=skyline_config.sky ' +
            ' --full-scan-rt-filter=ms2_ids --full-scan-rt-filter-tolerance=0.25 '
            ' --import-search-file=inspire_identifications.ssl ' +
            ' --import-fasta=inspire_identifications.fasta ' +
            ' --import-search-include-ambiguous ' +
            ' --report-conflict-resolution=overwrite ' +
            ' --report-add=skyline_report_template.skyr --report-name=myreport --report-invariant' +
            f' --report-file=skyline_report.csv > {config.output_folder}/quant/skyline_log.txt',
            tty=True,
            stdin_open=True,
            auto_remove=True,
            volumes={scans_folder_abs_path: {'bind': '/data', 'mode': 'rw'}},
        )
    else:
        os.system(
            f'"{config.skyline_runner}"' +
            f' --timestamp --dir={scans_folder_abs_path} --in=skyline_config.sky ' +
            ' --import-search-file=inspire_identifications.ssl ' +
            ' --import-fasta=inspire_identifications.fasta ' +
            ' --report-conflict-resolution=overwrite ' +
            ' --import-search-include-ambiguous ' +
            ' --report-add=skyline_report_template.skyr --report-name=myreport --report-invariant' +
            f' --report-file=skyline_report.csv > {config.output_folder}/quant/skyline_log.txt',
        )

    # Copy output back
    if os.path.exists(f'{config.scans_folder}/skyline_report.csv'):
        shutil.copyfile(
            f'{config.scans_folder}/skyline_report.csv',
            f'{config.output_folder}/quant/skyline_report.csv',
        )
    else:
        raise RuntimeError(f'Skyline failed. Check log at: {config.output_folder}/quant/skyline_log.txt')

    # Remove intermediate files.
    for skyline_file in SKYLINE_INTERMEDIATE_FILES:
        if os.path.exists(f'{config.scans_folder}/{skyline_file}'):
            os.remove(f'{config.scans_folder}/{skyline_file}')


def format_skyline_output(config):
    """ Function to format skyline outputs - per file total area less background
        and IDP per peptide per file.

    Parameters
    ----------
    config : inspire.config.Config
        Config object which controls the experiment.
    """
    # Read in report and rename some columns
    quant_df = pd.read_csv(f'{config.output_folder}/quant/skyline_report.csv')
    quant_df = quant_df.rename(columns={
        SKYLINE_OUTPUT_PEP_KEY: PEPTIDE_KEY,
        SKYLINE_OUTPUT_CHARGE_KEY: CHARGE_KEY,
        SKYLINE_OUTPUT_FILE_KEY: SOURCE_KEY,
    })

    # Avoid IDP of 0 if area is 0.
    quant_df['IsotopeDotProduct'] = quant_df[[SKYLINE_AREA_KEY, SKYLINE_OUTPUT_IDP_KEY]].apply(
        lambda df_row : round(
            df_row[SKYLINE_OUTPUT_IDP_KEY], 3
        ) if df_row[SKYLINE_AREA_KEY] > 0 else 1.0,
        axis=1,
    )
    quant_df = quant_df.sort_values(by=SKYLINE_AREA_KEY, ascending=False)
    quant_df = quant_df.drop_duplicates(subset=[
        SOURCE_KEY, PEPTIDE_KEY
    ])

    # Filter rows where source not defined, remove the ".raw" suffix, and collect unique sources.
    quant_df = quant_df[
        quant_df[SOURCE_KEY].apply(
            lambda x : isinstance(x, str)
        )
    ]
    quant_df[SOURCE_KEY] = quant_df[SOURCE_KEY].apply(lambda x : x[:-4])
    sources = quant_df[SOURCE_KEY].unique().tolist()

    # Calculate ratio of area to background and get area less background (min of 0).
    quant_df['AreaBackgroundRatio'] = quant_df[SKYLINE_AREA_KEY]/(
        quant_df[SKYLINE_BACKGROUND_KEY]+quant_df[SKYLINE_AREA_KEY]
    )
    quant_df['AreaLessBackground'] = (
        quant_df[SKYLINE_AREA_KEY] - quant_df[SKYLINE_BACKGROUND_KEY]
    )
    quant_df['AreaLessBackground'] = quant_df['AreaLessBackground'].apply(
        lambda x : x if x > 0 else 0
    )

    # Pivot peptide table and add IDP and add area/background ratio inforamtion.
    quant_area_df = pivot_quant_data(quant_df, 'AreaLessBackground', '_raw')
    quant_idp_df = pivot_quant_data(quant_df, 'IsotopeDotProduct', '_idp')
    quant_ratio_df = pivot_quant_data(quant_df, 'AreaBackgroundRatio', '_ratio')
    pivot_quant_df = pd.merge(
        quant_area_df,
        quant_idp_df,
        how='inner',
        on=PEPTIDE_KEY,
    )
    pivot_quant_df = pd.merge(
        pivot_quant_df,
        quant_ratio_df,
        how='inner',
        on=PEPTIDE_KEY,
    )

    # If not quantified fill with idp 0.
    for source in sources:
        pivot_quant_df[f'{source}_idp'] = pivot_quant_df[f'{source}_idp'].fillna(0.0)
        pivot_quant_df[f'{source}_idp'] = pivot_quant_df[f'{source}_idp'].replace(0.0, 1.0)

    # Add identification data, variation in technical replicates and rename columns
    pivot_quant_df = add_id_data(pivot_quant_df, sources, config)
    pivot_quant_df = rename_raw_file_columns(pivot_quant_df, config, sources)

    # Ensure first two columns are peptide and accession:
    quant_cols = [col for col in pivot_quant_df.columns if col not in (PEPTIDE_KEY, ACCESSION_KEY)]
    pivot_quant_df = pivot_quant_df[[PEPTIDE_KEY, ACCESSION_KEY] + quant_cols]

    # Write output
    pivot_quant_df.to_csv(
        f'{config.output_folder}/quant/quantified_per_file.csv',
        index=False,
    )


def pivot_quant_data(quant_df, value_key, suffix):
    """ Function to pivot quantification data on the peptide column.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of quantified peptides across multiple raw files.
    value_key : str
        The key which
    suffix : str
        The suffix which should be added to the value columns.

    Return
    ------
    pivot_quant_df : pd.DataFrame
        Quantification data pivoted by peptide index.
    """
    pivot_quant_df = quant_df.pivot(
        index=[PEPTIDE_KEY],
        columns=SOURCE_KEY,
        values=value_key,
    )
    pivot_quant_df = pivot_quant_df.reset_index()
    sources = [x for x in pivot_quant_df.columns if x != PEPTIDE_KEY]

    pivot_quant_df = pivot_quant_df.rename(columns={
        source: f'{source}{suffix}' for source in sources
    })

    return pivot_quant_df


def add_id_data(pivot_quant_df, sources, config):
    """ Function to add 1/0 flags for each peptide of whether the peptide was identified
        in each raw file of the experiment.

    Parameters
    ----------
    pivot_quant_df : pd.DataFrame
        A DataFrame pivoted on peptide with quantification across each file.
    sources : list of str
        A list of the raw files in the quantification data.
    config : inspire.config.Config
        Config object which controls the experiment.

    Returns
    -------
    pivot_quant_df : pd.DataFrame
        The input DataFrame with identification columns added.
    """
    # Read and filter assignments by q-value and remove duplicates within a raw file.
    identification_df = pd.read_csv(f'{config.output_folder}/finalPsmAssignments.csv')
    identification_df = identification_df[
        identification_df[FINAL_Q_VALUE_KEY] < config.quantification_cut_off
    ]
    identification_df = identification_df.drop_duplicates(
        subset=[PEPTIDE_KEY, ACCESSION_KEY, SOURCE_KEY]
    )

    # Groupby peptide and accession and create columns for identified in each file.
    identification_df = identification_df.groupby(
        [PEPTIDE_KEY, ACCESSION_KEY],
        as_index=False,
    )[SOURCE_KEY].apply(list)
    for source in sources:
        identification_df[f'{source}_identified'] = identification_df[
            SOURCE_KEY
        ].apply(lambda x, src=source : int(src in x))
    identification_df = identification_df.drop(SOURCE_KEY, axis=1)

    # Merge identification data with quantification data.
    pivot_quant_df = pd.merge(
        pivot_quant_df,
        identification_df,
        how='inner',
        on='peptide',
    )
    return pivot_quant_df


def rename_raw_file_columns(pivot_quant_df, config, sources):
    """ Function to rename columns to short names and write name mappings.

    Parameters
    ----------
    pivot_quant_df : pd.DataFrame
        DataFrame of quantification data.
    config : inspire.config.Config
        Config object which controls the experiment.
    sources : list of str
        A list of the raw files in the quantification data.

    Returns
    -------
    pivot_quant_df : pd.DataFrame
        DataFrame of quantification data with updated column names.
    """
    if config.control_flags is not None:
        control_sources = sorted(
            [source for source in sources if is_control(source, config.control_flags)]
        )
        inf_sources = sorted(
            [source for source in sources if not is_control(source, config.control_flags)]
        )
        for source_idx, source in enumerate(control_sources):
            pivot_quant_df = pivot_quant_df.rename(columns={
                f'{source}_raw': f'controlFile{source_idx+1}_raw',
                f'{source}_idp': f'controlFile{source_idx+1}_idp',
                f'{source}_ratio': f'controlFile{source_idx+1}_ratio',
                f'{source}_identified': f'controlFile{source_idx+1}_identified',
            })
        for source_idx, source in enumerate(inf_sources):
            pivot_quant_df = pivot_quant_df.rename(columns={
                f'{source}_raw': f'infectionFile{source_idx+1}_raw',
                f'{source}_idp': f'infectionFile{source_idx+1}_idp',
                f'{source}_ratio': f'infectionFile{source_idx+1}_ratio',
                f'{source}_identified': f'infectionFile{source_idx+1}_identified',
            })
        file_renaming_df = pd.DataFrame({
            'source': control_sources + inf_sources,
            'renamed': [
                f'controlFile{idx+1}' for idx in range(len(control_sources))
            ] + [
                f'infectionFile{idx+1}' for idx in range(len(inf_sources))
            ],
            'status': ['control']*len(control_sources) + ['infected']*len(inf_sources),
        })
    else:
        sources = sorted(sources)
        for source_idx, source in enumerate(sources):
            pivot_quant_df = pivot_quant_df.rename(columns={
                f'{source}_raw': f'file{source_idx+1}_raw',
                f'{source}_idp': f'file{source_idx+1}_idp',
                f'{source}_ratio': f'file{source_idx+1}_ratio',
                f'{source}_identified': f'file{source_idx+1}_identified',
            })
        file_renaming_df = pd.DataFrame({
            'source': sources,
            'renamed': [
                f'file{idx+1}' for idx in range(len(sources))
            ],
        })

    file_renaming_df.to_csv(f'{config.output_folder}/quant/metadata.csv', index=False)

    return pivot_quant_df
