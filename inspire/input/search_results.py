""" Generic functions for reading in any search results.
"""
import multiprocessing
import os

import pandas as pd

from inspire.accession import process_accession_groups
from inspire.constants import SOURCE_KEY, SCAN_KEY, PEPTIDE_KEY, LABEL_KEY, ACCESSION_STRATUM_KEY, ENGINE_SCORE_KEY
from inspire.input.mascot import read_mascot_data
from inspire.input.maxquant import read_mq_data
from inspire.input.msfragger import read_ms_fragger_data
from inspire.input.peaks import read_peaks_data
from inspire.utils import add_fixed_modifications, filter_for_prosit

def generic_read_df(config, save_dfs=True, overwrite_reduce=False):
    """ Function to read in search results from any search engine.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    save_dfs : bool
        Flag indicating whether the formatted dataframes should be saved to disk.
    overwrite_reduce : bool
        Flag indicating whether to force reduction of Mascot dataframe to best hit
        (for CE calibration pipeline).

    Returns
    -------
    search_df : pd.DataFrame
        A DataFrame of search results.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found.
    """
    if overwrite_reduce:
        reduce_mascot = True
    else:
        reduce_mascot = config.reduce

    if os.path.exists(f'{config.output_folder}/formated_df.csv') and config.reuse_input:
        search_df = pd.read_csv(f'{config.output_folder}/formated_df.csv')
        mods_df = pd.read_csv(f'{config.output_folder}/formated_mods.csv')
    else:
        if config.search_engine == 'mascot':
            search_df, mods_df = read_mascot_data(
                config.search_results,
                config.scan_title_format,
                config.source_files,
                reduce_mascot,
                config.source_filename,
                with_accession=config.use_accession_stratum,
            )
        elif config.search_engine == 'maxquant':
            search_df, mods_df = read_mq_data(config.search_results)
        elif config.search_engine == 'peaks':
            search_df, mods_df = read_peaks_data(config.search_results)
        elif config.search_engine == 'msfragger':
            n_cores = min(config.n_cores, multiprocessing.cpu_count())
            search_df, mods_df = read_ms_fragger_data(
                config.search_results,
                config.fixed_modifications,
                n_cores,
            )
        else:
            raise ValueError(f'Unknown Search Engine: {config.search_engine}')

        if (
            config.fixed_modifications is not None and
            config.search_engine != 'msfragger'
        ):
            search_df, mods_df = add_fixed_modifications(
                search_df,
                mods_df,
                config.fixed_modifications
            )

        if config.use_accession_stratum:
            search_df = process_accession_groups(search_df, config)
            search_df = search_df.sort_values(by=[LABEL_KEY, ACCESSION_STRATUM_KEY, ENGINE_SCORE_KEY, PEPTIDE_KEY], ascending=[False, True, True, True])
            search_df['ilPep'] = search_df[PEPTIDE_KEY].apply(lambda x : x.replace('I', 'L'))

            search_df = search_df.drop_duplicates(
                subset=[SOURCE_KEY, SCAN_KEY, 'ilPep']
            )

        search_df = filter_for_prosit(search_df, config.use_accession_stratum)

        if save_dfs and config.reuse_input:
            search_df.to_csv(f'{config.output_folder}/formated_df.csv', index=False)
            mods_df.to_csv(f'{config.output_folder}/formated_mods.csv', index=False)

    return search_df, mods_df
