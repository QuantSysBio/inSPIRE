""" Functions from proteome remapping in relation to epitope candidate extraction.
"""
import multiprocessing as mp
import pandas as pd

from inspire.constants import (
    ENDC_TEXT,
    OKCYAN_TEXT,
)
from inspire.utils import fetch_proteome


def filter_pathogen_only_peptides(final_df, config):
    """ Function to remap and filter the final assignments for pathogen only peptides.

    Parameters
    ----------
    final_df : pd.DataFrame
        PEP filtered DataFrame.
    config : inspire.config.Config
        The Config object which controls the experiment.

    Returns
    -------
    final_df : pd.DataFrame
        DataFrame of pathogen mapped peptides only.
    multi_mapped_df : pd.DataFrame
        DataFrame of pathogen mapped peptides that multimapped.
    """
    final_df = add_accession_data(final_df, config)

    total_count = final_df.shape[0]
    final_df, multi_mapped_df = filter_multi_mapped(final_df, config.host_proteome)
    no_host_accession_count = final_df.shape[0]

    print(
        OKCYAN_TEXT +
        f'\t{total_count - no_host_accession_count} PSMs dropped due to host accessions.'
        + ENDC_TEXT
    )

    if not no_host_accession_count:
        exit_with_no_peptides_identified(config)

    total_count = final_df.shape[0]
    final_df = final_df[final_df['pathogenProteins'].apply(lambda x : x != 'unknown')]
    known_accession_count = final_df.shape[0]

    print(
        OKCYAN_TEXT +
        f'\t{total_count - known_accession_count} PSMs dropped due to unknown accessions.'
        + ENDC_TEXT
    )

    if not known_accession_count:
        exit_with_no_peptides_identified(config)

    return final_df, multi_mapped_df


def _sub_add_accession(unique_df, host_proteome, pathogen_proteome):
    unique_df['hostAccessions'] = unique_df['peptide'].apply(
        lambda x : fetch_antigen_data(
            x, host_proteome, trace_accession=False,
        )
    )

    unique_df['pathogenAccessions'] = unique_df['peptide'].apply(
        lambda x : fetch_antigen_data(x, pathogen_proteome)
    )
    unique_df['pathogenProteins'] = unique_df['pathogenAccessions'].apply(
        lambda x : ','.join([y[0] for y in x]) if x else 'unknown'
    )
    return unique_df

def add_accession_data(final_df, config):
    """ Function to add accession data to the final assignments, remapping all peptides to
        the host and pathogen proteomes.

    Parameters
    ----------
    final_df : pd.DataFrame
        The inSPIRE final assignments DataFrame.
    config : inspire.config.Config
        The Config object for running the experiment.

    Returns
    -------
    final_df : pd.DataFrame
        The inSPIRE final assignments DataFrame updated with
    """
    host_proteome = fetch_proteome(config.host_proteome)
    pathogen_proteome = fetch_proteome(config.pathogen_proteome)

    unique_df = final_df[['peptide']].drop_duplicates(subset=['peptide'])

    batch_size = unique_df.shape[0]//config.n_cores

    batch_values = []
    for batch_idx in range(config.n_cores):
        batch_values.extend([batch_idx]*batch_size)

    if (additional_psms := unique_df.shape[0]%config.n_cores):
        batch_values.extend([config.n_cores-1]*additional_psms)

    unique_df['batch'] = batch_values

    pep_df_list = [unique_df[unique_df['batch'] == b_idx] for b_idx in range(config.n_cores)]
    func_args = []
    for sub_pep_df in pep_df_list:
        func_args.append([sub_pep_df, host_proteome, pathogen_proteome])

    with mp.get_context('spawn').Pool(processes=config.n_cores) as pool:
        pep_df_list = pool.starmap(_sub_add_accession, func_args)
    remapped_pep_df = pd.concat(pep_df_list)

    total_df = pd.merge(final_df, remapped_pep_df, how='inner', on='peptide')

    return total_df


def fetch_antigen_data(
        peptide,
        proteome,
        trace_accession=True,
    ):
    """ Function to check for the presence of an identified peptide as either canonical
        or spliced in the input proteome.
    """
    accession_stratum = []
    peptide = peptide.replace('I', 'L')
    for protein in proteome:
        if peptide in protein[1]:
            location = protein[1].index(peptide) + 1
            if trace_accession:
                if not accession_stratum:
                    accession_stratum = [(protein[0], location, protein[2])]
                else:
                    accession_stratum.append((protein[0], location, protein[2]))
            else:
                return 'PCP'
    if trace_accession:
        return accession_stratum
    return 'unknown'


def filter_multi_mapped(final_df, host_proteome_filename):
    """ Function to remove peptides which map to both host and pathogen proteomes.

    Parameters
    ----------
    final_df : pd.DataFrame
        A DataFrame of all peptides identified.
    host_proteome_filename : str
        The path to the file containing the host proteome.

    Returns
    -------
    final_df : pd.DataFrame
        A DataFrame of all peptides identified filterd to remove those from the
        host proteome.
    multi_mapped_df : pd.DataFrame
        A DataFrame of all peptides which map to both the host and pathogen proteomes.
    """
    host_proteome = fetch_proteome(host_proteome_filename)
    multi_mapped_df = final_df[
        (final_df['hostAccessions'] != 'unknown') &
        (final_df['pathogenProteins'] != 'unknown')
    ]
    multi_mapped_df['hostAccessions'] = multi_mapped_df['peptide'].apply(
        lambda x : fetch_antigen_data(
            x, host_proteome, trace_accession=False,
        )
    )
    final_df = final_df[
        final_df['hostAccessions'].apply(lambda x : x == 'unknown')
    ]

    return final_df, multi_mapped_df


def extract_protein_ids(pathogen_accessions):
    """ Function to extract protein ids.
    """
    accessions = pathogen_accessions.split(')')
    proteins = []

    for acc in accessions:
        if acc:
            proteins.append('_'.join(acc.strip('[').strip(']').split('_')[:-1]))

    return ','.join(proteins)


def exit_with_no_peptides_identified(config):
    """ Function to exit if no pathogen peptides are identified.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object which controls the experiment.
    """
    print(
        OKCYAN_TEXT +
        '\tNo pathogen peptides identified.'
        + ENDC_TEXT
    )

    _ = pd.ExcelWriter( # pylint: disable=abstract-class-instantiated
        f'{config.output_folder}/epitope/potentialEpitopeCandidates.xlsx',
        engine='xlsxwriter',
    )
