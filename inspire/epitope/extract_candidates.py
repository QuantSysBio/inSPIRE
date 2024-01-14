""" Function for investigating IP samples containing potential spliced epitopes.
"""
from copy import deepcopy
from math import floor
import shutil
import os

import pandas as pd

from inspire.constants import (
    ENGINE_SCORE_KEY,
    FINAL_SCORE_KEY,
    FINAL_POSTEP_KEY,
    OUT_POSTEP_KEY,
    PEPTIDE_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.epitope.proteome_mapping import filter_pathogen_only_peptides
from inspire.epitope.plot_utils import (
    bar_plot,
    plot_binding_clustermap,
    plot_quant_pca,
    swarm_plots,
)
from inspire.epitope.report_template import create_epitope_report
from inspire.plot_spectra.plot_spectra import plot_spectra
from inspire.utils import is_control

ANTIGEN_KEY = 'protein'
ANTIGEN_LOC_KEY = 'proteinLocation'
ANTIGEN_DESC_KEY = 'proteinDescription'
EPITOPE_CONF_KEY = 'confidence'
EPITOPE_N_FILES = 'numberOfUniqueFiles'
BASE_REPORTING_COLUMNS = [
    ANTIGEN_KEY,
    ANTIGEN_LOC_KEY,
    ANTIGEN_DESC_KEY,
    FINAL_SCORE_KEY,
    EPITOPE_CONF_KEY,
    ENGINE_SCORE_KEY,
    SPECTRAL_ANGLE_KEY,
    'deltaRT',
]


def extract_epitope_candidates(config):
    """ Function for extracting epitope candidates from inSPIRE identifications.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object which controls the experiment.
    """
    final_df = pd.read_csv(f'{config.output_folder}/finalPsmAssignments.csv')

    # Add columns of interest
    final_df['peptideLength'] = final_df['peptide'].apply(len)
    final_df[EPITOPE_CONF_KEY] = (
        (1 - final_df[FINAL_POSTEP_KEY])*100
    ).apply(
        lambda x : floor(x*100)/100
    )

    # Initial filters:
    final_df = final_df[
        (final_df[FINAL_POSTEP_KEY] < config.epitope_candidate_cut_off) &
        (final_df['peptideLength'] <= config.epitope_length_cut_off)
    ]

    # Add accession data:
    final_df, multi_mapped_df = filter_pathogen_only_peptides(final_df, config)

    # Write peptides which map to both pathogen and host proteomes (not as epitope
    # candidates but potentially interesting)
    write_excluded_peptides(
        multi_mapped_df,
        config,
        'multi_mapped_peptides',
    )

    # Filter those peptides found in control.
    final_df['foundInControl'] = final_df.groupby(PEPTIDE_KEY)[SOURCE_KEY].transform(
        lambda source : found_in_control(source, config.control_flags)
    )
    write_excluded_peptides(
        final_df[final_df['foundInControl']],
        config,
        'control_excluded_peptides',
    )
    final_df = final_df[~final_df['foundInControl']]

    # Write best MS2 spectrum per peptide for plotting later.
    final_df.drop_duplicates(subset=[PEPTIDE_KEY]).to_csv(
        f'{config.output_folder}/plotData.csv', index=False,
    )

    # Add useful columns for number of files where peptide identified and found by search engine.
    final_df[EPITOPE_N_FILES] = final_df.groupby(
        PEPTIDE_KEY
    )[SOURCE_KEY].transform('nunique')
    final_df = check_engine_results(final_df, config)

    # Separate and rename protein ID data
    final_df[ANTIGEN_LOC_KEY] = final_df['pathogenAccessions'].apply(
        lambda x : ','.join([str(y[1]) for y in x])
    )
    final_df[ANTIGEN_DESC_KEY] = final_df['pathogenAccessions'].apply(
        lambda x : ','.join([y[2] for y in x])
    )
    final_df = final_df.rename(columns={'pathogenProteins': ANTIGEN_KEY})

    # Select columns to be reported and drop duplicate PSMs
    final_df = final_df.sort_values(by=FINAL_SCORE_KEY, ascending=False)
    final_peptide_df = final_df[
        [
            PEPTIDE_KEY, EPITOPE_N_FILES, 'foundBySearchEngine'
        ] + BASE_REPORTING_COLUMNS + sorted(
            [
                x for x in final_df.columns if (
                    x.endswith('Affinity') or
                    x.endswith('BindLevel') or
                    x.endswith('%Rank_BA')
                )
            ]
        )
    ].drop_duplicates(subset='peptide')
    final_peptide_df = final_peptide_df.reset_index(drop=True)

    for feature in (ENGINE_SCORE_KEY, SPECTRAL_ANGLE_KEY, 'deltaRT'):
        final_peptide_df[feature] = final_peptide_df[feature].apply(
            lambda x : round(x, 2)
        )

    # Add all data to single csv file and write separated data to csv.
    final_peptide_df.to_csv(
        f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv',
        index=False,
    )
    write_excel_report(final_peptide_df, final_df, config)

    bar_plot(config)
    plot_binding_clustermap(config)
    swarm_plots(config)

    # Provide MS2 spectral plots for identified peptides.
    plot_spectra(config)
    if os.path.exists(f'{config.output_folder}/spectralPlots.pdf'):
        shutil.move(
            f'{config.output_folder}/spectralPlots.pdf',
            f'{config.output_folder}/epitope/spectralPlots.pdf',
        )

    plot_quant_pca(config)

    create_epitope_report(config)

def write_excluded_peptides(excluded_df, config, file_name):
    """ Function to write peptides which are excluded either due to their presence in
        control files or multi-mapping between host and pathogen.
    """
    if not excluded_df.shape[0]:
        excluded_df.to_csv(
            f'{config.output_folder}/epitope/{file_name}.csv', index=False
        )
        return

    sources = excluded_df[SOURCE_KEY].unique().tolist()
    excluded_df = excluded_df.groupby(
        ['peptide', 'proteins'], as_index=False
    )['source'].apply(list)
    if not excluded_df.shape[0]:
        return

    for source in sources:
        excluded_df[f'{source}_identified'] = excluded_df['source'].apply(
            lambda x, src=source : int(src in x)
        )

    excluded_df['nInfectedFiles'] = excluded_df[[
        f'{source}_identified'for source in sources if not is_control(source, config.control_flags)
    ]].sum(axis=1)

    excluded_df['nControlFiles'] = excluded_df[[
        f'{source}_identified'for source in sources if is_control(source, config.control_flags)
    ]].sum(axis=1)

    excluded_df[
        [PEPTIDE_KEY, 'proteins', 'nInfectedFiles', 'nControlFiles'] +
        [f'{source}_identified' for source in sources]
    ].to_csv(
        f'{config.output_folder}/epitope/{file_name}.csv', index=False
    )

def found_in_control(sources, control_flags):
    """ Function to check if control flags are found in the source files where
        a peptide was identified.

    sources : list of str
        A list of source files from which a single peptide was identified.
    control_flags : list of str
        A list of the control flags which mark a source file as a control file.
    """
    if control_flags is None:
        return False

    for source in sources:
        if is_control(source, control_flags):
            return True
    return False


def check_engine_results(pathogen_df, config):
    """ Function to check if identified PSMs were found using the original search engine.

    Parameters
    ----------
    pathogen_df : pd.DataFrame
        DataFrame of identified pathogen peptides.
    config : inspire.config.Config
        The Config object which controls the experiment.

    Returns
    -------
    pathogen_df : pd.DataFrame
        The input DataFrame with information added on the original search engine identifications.
    """
    pathogen_df['maxEngineScore'] = pathogen_df.groupby(
        'peptide'
    )[ENGINE_SCORE_KEY].transform('max')

    if config.engine_score_cut is not None:
        pathogen_df['foundBySearchEngine'] = pathogen_df['maxEngineScore'].apply(
            lambda x : 'Yes' if x > config.engine_score_cut else 'No'
        )
    else:
        non_spect_df = pd.read_csv(
            f'{config.output_folder}/non_spectral.{config.rescore_method}.psms.txt',
            sep='\t',
        )
        non_spect_df = non_spect_df[
            (non_spect_df[OUT_POSTEP_KEY[config.rescore_method]] < config.epitope_candidate_cut_off)
        ]
        non_spect_df = non_spect_df.drop_duplicates(subset='peptide')
        non_spect_df = non_spect_df[['peptide']]
        non_spect_df['peptide'] = non_spect_df['peptide'].apply(
            lambda peptide : peptide.split('.')[1] if '.' in peptide else peptide
        )
        non_spect_df['foundBySearchEngine'] = 'Yes'

        pathogen_df = pd.merge(pathogen_df, non_spect_df, how='left', on='peptide')
        pathogen_df['foundBySearchEngine'] = pathogen_df['foundBySearchEngine'].apply(
            lambda x : x if x == 'Yes' else 'No'
        )

    pathogen_df = pathogen_df.drop('maxEngineScore', axis=1)

    return pathogen_df


def write_excel_report(final_peptide_df, final_spectra_df, config):
    """ Function to write a full report of the identified pathogen peptides in Microsoft Excel.

    Parameters
    ----------
    final_df : pd.DataFrame
        The DataFrame of all identified pathogen peptides.
    """
    if final_peptide_df is None:
        # Empty Excel file to be written.
        xl_writer = pd.ExcelWriter( # pylint: disable=abstract-class-instantiated
            f'{config.output_folder}/epitope/potentialEpitopeCandidates.xlsx',
            engine='xlsxwriter',
        )
        return

    reporting_columns = [
        PEPTIDE_KEY,
        ANTIGEN_KEY,
        ANTIGEN_LOC_KEY,
        ANTIGEN_DESC_KEY,
        'foundBySearchEngine',
        EPITOPE_N_FILES,
        EPITOPE_CONF_KEY,
        ENGINE_SCORE_KEY,
        SPECTRAL_ANGLE_KEY,
        'deltaRT',
    ]

    final_id_df = final_peptide_df[reporting_columns]
    total_df = deepcopy(final_id_df)

    with pd.ExcelWriter( # pylint: disable=abstract-class-instantiated
        f'{config.output_folder}/epitope/potentialEpitopeCandidates.xlsx',
        engine='xlsxwriter',
    ) as xl_writer:
        final_id_df.to_excel(
            xl_writer,
            index=False,
            sheet_name='finalCandidates',
        )
        final_spectra_df[[SOURCE_KEY, SCAN_KEY, PEPTIDE_KEY, 'modifiedSequence']].to_excel(
            xl_writer,
            index=False,
            sheet_name='spectraIdentified',
        )

        if config.use_binding_affinity is not None:
            total_df, xl_writer = add_binding_affinity_data(total_df, final_peptide_df, xl_writer)

        if os.path.exists(f'{config.output_folder}/quant/fold_changes.csv'):
            total_df, xl_writer = add_quantification_data(final_peptide_df, xl_writer, config)

        if config.host_only_results is not None:
            total_df, xl_writer = add_host_only_validation(
                config,
                total_df,
                final_spectra_df,
                xl_writer,
            )

        total_df.to_excel(
            xl_writer,
            index=False,
            sheet_name='allInformation',
        )

        xl_writer = add_conditional_formatting(xl_writer)

def add_quantification_data(final_peptide_df, xl_writer, config):
    """ Function to add quantification data to the output .xlsx file.

    Parameters
    ----------
    final_peptide_df : pd.DataFrame
        The final DataFrame of unique peptides which are potential epitope candidates.
    xl_writer : pd.ExcelWriter
        Writer for output .xlsx file
    config : inspire.config.Config
        Config object which controls the experiment.
    """
    fc_df = pd.read_csv(f'{config.output_folder}/quant/fold_changes.csv')
    final_peptide_df['rank'] = final_peptide_df.index
    fc_df = pd.merge(
        final_peptide_df[['rank', 'peptide']],
        fc_df,
        how='left',
        on='peptide',
    )
    fc_df = fc_df.fillna(0.0)

    fc_df = fc_df.sort_values(by='rank')
    fc_df = fc_df.drop('rank', axis=1)

    fc_df['absChange'] = fc_df[['meanAreaInfected', 'meanAreaControl']].apply(
        lambda df_row : 2**df_row['meanAreaInfected'] - 2**df_row['meanAreaControl'],
        axis=1
    )
    fc_df.to_excel(
        xl_writer,
        index=False,
        sheet_name='quantificationData',
    )
    final_peptide_df = final_peptide_df.drop('rank', axis=1)

    total_df = pd.merge(
        final_peptide_df,
        fc_df.drop('peptide', axis=1),
        left_index=True,
        right_index=True
    )

    return total_df, xl_writer


def add_conditional_formatting(xl_writer):
    """ Function
    """
    workbook  = xl_writer.book
    format1 = workbook.add_format({'bg_color': 'orange'})

    worksheet = xl_writer.sheets['finalCandidates']
    worksheet.conditional_format(
        1, 100, 1, 500,
        {'type': 'formula', 'criteria': ''}
    )
    worksheet.conditional_format(
        'G1:G100',
        {
            'type': 'text',
            'criteria': 'containing',
            'value':    'Yes',
            'format':   format1,
        },
    )
    return xl_writer

def add_binding_affinity_data(total_df, final_peptide_df, xl_writer):
    """ Function to add binding affinity prediction data.

    Parameters
    ----------
    total_df : pd.DataFrame
    """
    binding_affinity_df = final_peptide_df[
        ['peptide'] +
        [
            x for x in final_peptide_df.columns if x.endswith(
                'Level'
            )  or x.endswith(
                'Affinity'
            ) or x.endswith(
                '%Rank_BA'
            )
        ]
    ]
    binding_affinity_df.to_excel(
        xl_writer,
        index=False,
        sheet_name='bindingAffinityData',
    )

    total_df = pd.merge(
        final_peptide_df,
        binding_affinity_df.drop('peptide', axis=1),
        left_index=True,
        right_index=True
    )
    return total_df, xl_writer


def add_host_only_validation(config, total_df, final_df, xl_writer):
    """ Function to add validation from identification from a search without pathogen proteome.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object which controls the experiment.
    """
    comp_df = pd.read_csv(
        config.host_only_results
    )

    original_final_df = pd.read_csv(
        f'{config.output_folder}/finalPsmAssignments.csv'
    )
    original_final_df = original_final_df[
        original_final_df['postErrProb'] < config.epitope_candidate_cut_off
    ]
    original_final_df = original_final_df[['source', 'scan', 'peptide']]
    original_final_df['potentialChimera'] = 'Yes'

    comp_df = pd.merge(
        comp_df,
        original_final_df,
        how='left',
        on=['source', 'scan', 'peptide']
    )

    comp_df['potentialChimera'] = comp_df['potentialChimera'].fillna('No')

    comp_df = comp_df[[
        'source', 'scan', 'potentialChimera', 'peptide', 'spectralAngle', 'engineScore',
    ]].rename(
        columns={
            'peptide':  'competitorPeptide',
            'spectralAngle': 'competitorSpectralAngle',
            'engineScore': 'competitorEngineScore',
        }
    )

    final_comp_df = pd.merge(
        final_df[
            ['source', 'scan', 'peptide', FINAL_SCORE_KEY]
        ].drop_duplicates(subset='peptide'),
        comp_df,
        how='left',
        on=['source', 'scan']
    )

    final_comp_df = final_comp_df.sort_values(by=FINAL_SCORE_KEY, ascending=False)

    final_comp_df = final_comp_df.drop(['source', 'scan', FINAL_SCORE_KEY], axis=1)

    final_comp_df.to_excel(
        xl_writer,
        index=False,
        sheet_name='competitorData',
    )

    total_df = pd.merge(
        total_df,
        final_comp_df.drop('peptide', axis=1),
        left_index=True,
        right_index=True
    )

    return total_df, xl_writer
