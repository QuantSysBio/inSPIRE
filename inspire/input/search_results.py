""" Generic functions for reading in any search results.
"""
import os

import pandas as pd

from inspire.input.mascot import read_mascot_data
from inspire.input.maxquant import read_mq_data
from inspire.input.peaks import read_peaks_data
from inspire.utils import add_fixed_modifications

def generic_read_df(config, save_dfs=True):
    """ Function to read in search results from any search engine.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.

    Returns
    -------
    search_df : pd.DataFrame
        A DataFrame of search results.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found.
    """
    if os.path.exists(f'{config.output_folder}/formated_df.csv') and config.reuse_input:
        search_df = pd.read_csv(f'{config.output_folder}/formated_df.csv')
        mods_df = pd.read_csv(f'{config.output_folder}/formated_mods.csv')
    else:
        if config.search_engine == 'mascot':
            search_df, mods_df = read_mascot_data(
                config.search_results,
                config.scan_title_format,
                config.source_files,
                config.reduce,
                config.source_filename,
            )
        elif config.search_engine == 'maxquant':
            search_df, mods_df = read_mq_data(config.search_results)
        elif config.search_engine == 'peaks':
            search_df, mods_df = read_peaks_data(config.search_results)
        else:
            raise ValueError(f'Unknown Search Engine: {config.search_engine}')

        if config.fixed_modifications is not None:
            search_df, mods_df = add_fixed_modifications(
                search_df,
                mods_df,
                config.fixed_modifications
            )

        if save_dfs and config.reuse_input:
            search_df.to_csv(f'{config.output_folder}/formated_df.csv', index=False)
            mods_df.to_csv(f'{config.output_folder}/formated_mods.csv', index=False)

    return search_df, mods_df
