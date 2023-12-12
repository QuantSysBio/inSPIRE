""" Function for dealing with different accession groups.
"""
from copy import deepcopy

import polars as pl

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    ENGINE_SCORE_KEY,
    LABEL_KEY,
    PEPTIDE_KEY,
    SOURCE_KEY,
    SCAN_KEY
)
from inspire.utils import fetch_proteome

ACCESSION_SPLITTERS = {
    'mascot': ',',	
    'msfragger': ',',	
    'maxquant': ';',	
    'peaks': ':',	
}

def get_accession_group(accession, search_engine, accession_hierarchy, accession_groups):
    """ Function to get the accession group from the accession of a peptide.

    Parameters
    ----------
    accession : str
        The accession of the peptide.
    search_engine : str
        The search engine which provided the data.
    accession_hierarchy : list of str
        The order in which groups should be assigned in case of multi-mappers.
    accession_groups : dict
        A dictionary of mapping accession groups to the flag that will be seen
        in the accession.
    """
    splitter = ACCESSION_SPLITTERS.get(search_engine)
    if splitter is None:
        raise ValueError(f'Unrecognised Search Engine: {search_engine}')

    if accession in ('deNovo', 'unknown'):
        return len(accession_hierarchy)

    all_possible_accessions = accession.split(splitter)
    assignment = None
    for individiual_accession in all_possible_accessions:
        found = False
        for idx, acc in enumerate(accession_hierarchy):
            if accession_groups[acc] in individiual_accession:
                if assignment is None or idx < assignment:
                    assignment = idx
                found = True
        if not found:
            return 0
    return assignment

def get_invitro_spi_acc_group(accession, search_engine):
    """ Function to get the accession group of a PSM from the accession column in invitroSPI
        format.

    Parameters
    ----------
    accession : str
        The accession of a PSM.
    search_engine : str
        The search engine from which the results were generated.

    Returns
    -------
    assignment : int
        The accession group index as defined by the config.accession_hierarchy.
    """
    splitter = ACCESSION_SPLITTERS.get(search_engine)
    if splitter is None:
        raise ValueError(f'Unrecognised Search Engine: {search_engine}')

    all_possible_accessions = accession.split(splitter)
    assignment = 0
    for individiual_accession in all_possible_accessions:
        if 'PCP_' in individiual_accession:
            return 0
        if 'PSP_' in individiual_accession:
            assignment = 1

    return assignment

def validate_accession_stratum(df_row, proteome, rev_proteome, config):
    """ Function to validate the accession of a non-canonical PSM by checking for the
        sequence in the standard proteome.

    Parameters
    ----------
    df_row : pd.Series
        An entry in a DataFrame of search results.
    proteome : list of tuple
        The standard proteome.
    rev_proteome : list of tuple
        The reversed proteome.
    config : inspire.config.Config
        The Config object for the experiment.

    Returns
    -------
    df_row : pd.Series
        The input row updated with accession and accession group validated.
    """
    pep_seq = df_row[PEPTIDE_KEY].replace('I', 'L')
    if df_row[ACCESSION_STRATUM_KEY] == 0:
        return {
            ACCESSION_KEY: df_row[ACCESSION_KEY],
            ACCESSION_STRATUM_KEY: df_row[ACCESSION_STRATUM_KEY],
        }

    results ={}
    if df_row[LABEL_KEY] == 1:
        for entry in proteome:
            if pep_seq in entry[1]:
                if config.accession_format == 'invitroSPI':
                    position = entry[1].index(pep_seq) + 1
                    results[ACCESSION_KEY] = f'PCP_{position}_{position+len(pep_seq)}'
                    results[ACCESSION_STRATUM_KEY] = 0
                else:
                    results[ACCESSION_KEY] = entry[0]
                    results[ACCESSION_STRATUM_KEY] = 0
    else:
        for entry in rev_proteome:
            if pep_seq in entry[1]:
                if config.accession_format == 'invitroSPI':
                    position = entry[1].index(pep_seq) + 1
                    results[ACCESSION_KEY] = f'PCP_{position}_{position+len(pep_seq)}'
                    results[ACCESSION_STRATUM_KEY] = 0
                else:
                    results[ACCESSION_KEY] = entry[0]
                    results[ACCESSION_STRATUM_KEY] = 0

    return results


def process_accession_groups(main_df, config):
    """ Function to process the accession groups of a DataFrame of search results.

    Parameters
    ----------
    main_df : pd.DataFrame
        The DataFrame of search results.
    config : inspire.config.Config
        The Config object for the experiment.

    Returns
    -------
    main_df : pd.DataFrame
        The input DataFrame updated with accession group column, validated accession column
        and removed any PSMs of accession groups to ignore.
    """
    if config.accession_format == 'invitroSPI':
        main_df = main_df.with_columns(pl.col(ACCESSION_KEY).apply(
            lambda x : get_invitro_spi_acc_group(x, config.search_engine)
        ).alias(ACCESSION_STRATUM_KEY))
    else:
        main_df = main_df.with_columns(
            pl.col(ACCESSION_KEY).apply(
                lambda x : get_accession_group(
                    x,
                    config.search_engine,
                    config.accession_hierarchy,
                    config.accession_flags
                ) if isinstance(x, str) else 0
            ).alias(ACCESSION_STRATUM_KEY)
        )

    if config.proteome is not None:
        prot_sequences = fetch_proteome(config.proteome, with_desc=False)

        rev_prot_seqs = []
        for entry in deepcopy(prot_sequences):
            new_entry = ('reversed_' + entry[0], entry[1][::-1])
            rev_prot_seqs.append(new_entry)

        main_df = main_df.with_columns(
            pl.struct([ACCESSION_KEY, ACCESSION_STRATUM_KEY, PEPTIDE_KEY, LABEL_KEY]).apply(
                lambda x : validate_accession_stratum(x, prot_sequences, rev_prot_seqs, config),
            ).alias('results')
        )
        main_df = main_df.drop([ACCESSION_KEY, ACCESSION_STRATUM_KEY])
        main_df = main_df.unnest('results')

    main_df = main_df.sort(
        by=[LABEL_KEY, ACCESSION_STRATUM_KEY, ENGINE_SCORE_KEY, PEPTIDE_KEY],
        descending=[True, False, True, False],
    )
    main_df = main_df.unique(subset=[SOURCE_KEY, SCAN_KEY, PEPTIDE_KEY])

    if 'ignore' in config.accession_hierarchy:
        ignore_idx = config.accession_hierarchy.index('ignore')
        main_df = main_df.filter(pl.col(ACCESSION_STRATUM_KEY).ne(ignore_idx))

    return main_df
