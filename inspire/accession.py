""" Function for dealing with different accession groups.
"""
from copy import deepcopy

from Bio import SeqIO

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    LABEL_KEY,
    PEPTIDE_KEY,
)


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
    if search_engine == 'peaks':
        splitter = ':'
    elif search_engine == 'maxquant':
        splitter = ';'
    elif search_engine == 'mascot':
        splitter = ','
    else:
        raise ValueError(f'Unrecognised Search Engine: {search_engine}')
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
    if search_engine == 'peaks':
        splitter = ':'
    elif search_engine == 'maxquant':
        splitter = ';'
    elif search_engine == 'mascot':
        splitter = ','
    else:
        raise ValueError(f'Unrecognised Search Engine: {search_engine}')
    all_possible_accessions = accession.split(splitter)
    assignment = None
    for individiual_accession in all_possible_accessions:
        if 'PCP' in individiual_accession:
            return 0
        if 'PSP' in individiual_accession:
            data = individiual_accession.split('_')
            data = [x for x in data if x != 'Reversed']
            range_1 = set(range(int(data[1]), int(data[2])))
            range_2 = set(range(int(data[3]), int(data[4])))

            if (list(set(range_1) & set(range_2)) and assignment is None):
                assignment = 2
            else:
                assignment = 1
    return assignment

def validate_accession_group(df_row, proteome, rev_proteome, config):
    """ Function to validate the accession of a non-canonical PSM by checking for the
        sequence in the standard proteome.

    Parameters
    ----------
    df_row : pd.Series
        An entry in a DataFrame of search results.
    proteome : Bio.Seq
        The standard proteome.
    rev_proteome : Bio.Seq
        The reversed proteome.
    config : inspire.config.Config
        The Config object for the experiment.

    Returns
    -------
    df_row : pd.Series
        The input row updated with accession and accession group validated.
    """
    pep_seq = df_row[PEPTIDE_KEY]
    if df_row[ACCESSION_STRATUM_KEY] == 0:
        return df_row

    if df_row[LABEL_KEY] == 1:
        for entry in proteome:
            if pep_seq in entry.seq:
                if config.accession_format == 'invitroSPI':
                    position = entry.seq.index(pep_seq) + 1
                    df_row[ACCESSION_KEY] = f'PCP_{position}_{position+len(pep_seq)}'
                    df_row[ACCESSION_STRATUM_KEY] = 0
                else:
                    df_row[ACCESSION_KEY] = entry.id
                    df_row[ACCESSION_STRATUM_KEY] = 0
    else:
        for entry in rev_proteome:
            if pep_seq in entry.seq:
                if config.accession_format == 'invitroSPI':
                    position = entry.seq.index(pep_seq) + 1
                    df_row[ACCESSION_KEY] = f'PCP_{position}_{position+len(pep_seq)}'
                    df_row[ACCESSION_STRATUM_KEY] = 0
                else:
                    df_row[ACCESSION_KEY] = entry.id
                    df_row[ACCESSION_STRATUM_KEY] = 0

    return df_row


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
        main_df[ACCESSION_STRATUM_KEY] = main_df[ACCESSION_KEY].apply(
            lambda x : get_invitro_spi_acc_group(x, config.search_engine)
        )
    else:
        main_df[ACCESSION_STRATUM_KEY] = main_df[ACCESSION_KEY].apply(
            lambda x : get_accession_group(
                x,
                config.search_engine,
                config.accession_hierarchy,
                config.accession_groups
            ) if isinstance(x, str) else 0
        )

    if config.proteome is not None:
        with open(config.proteome, 'r', encoding='UTF-8') as fasta_file:
            prot_sequences = list(SeqIO.parse(fasta_file, 'fasta'))

        rev_prot_seqs = []
        for entry in deepcopy(prot_sequences):
            entry.seq = entry.seq[::-1]
            entry.id = 'reverse' + entry.id
            rev_prot_seqs.append(entry)
        main_df = main_df.apply(
            lambda x : validate_accession_group(x, prot_sequences, rev_prot_seqs, config),
            axis=1
        )

    if 'ignore' in config.accession_hierarchy:
        ignore_idx = config.accession_hierarchy.index('ignore')
        main_df = main_df[main_df[ACCESSION_STRATUM_KEY] != ignore_idx]

    return main_df
