""" Generic functions for reading in any search results.
"""
import multiprocessing
import os

import pandas as pd
import polars as pl

from inspire.accession import process_accession_groups
from inspire.input.mascot import read_mascot_data
from inspire.input.maxquant import read_mq_data
from inspire.input.msfragger import read_ms_fragger_data
from inspire.input.peaks import read_peaks_data
from inspire.utils import add_fixed_modifications

def generic_read_df(config, save_dfs=True, for_calibration=False):
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
    if for_calibration:
        reduce_results = True
    else:
        reduce_results = config.rescore_method != 'percolatorSeparate'

    n_cores = min(config.n_cores, multiprocessing.cpu_count())

    if os.path.exists(f'{config.output_folder}/formated_df.csv') and config.reuse_input:
        search_df = pl.read_csv(f'{config.output_folder}/formated_df.csv')
        mods_df = pd.read_csv(f'{config.output_folder}/formated_mods.csv')
    else:
        if config.search_engine == 'mascot':
            search_df, mods_df = read_mascot_data(
                config.search_results,
                config.scan_title_format,
                config.source_files,
                reduce_results,
                config.source_filename,
                with_accession=config.use_accession_stratum,
            )
            search_df = pl.from_pandas(search_df)
        elif config.search_engine == 'maxquant':
            search_df, mods_df = read_mq_data(config.search_results)
        elif config.search_engine == 'peaks':
            search_df, mods_df = read_peaks_data(config.search_results)
        elif config.search_engine == 'msfragger':
            search_df, mods_df = read_ms_fragger_data(
                config.search_results,
                config.fixed_modifications,
                n_cores,
                reduce_results,
            )
        else:
            raise ValueError(f'Unknown Search Engine: {config.search_engine}')

        if config.use_accession_stratum:
            search_df = process_accession_groups(search_df, config)
        if (
            config.fixed_modifications is not None and
            config.search_engine != 'msfragger'
        ):
            search_df, mods_df = add_fixed_modifications(
                search_df,
                mods_df,
                config.fixed_modifications
            )

        if save_dfs and config.reuse_input:
            search_df.write_csv(f'{config.output_folder}/formated_df.csv')
            mods_df.to_csv(f'{config.output_folder}/formated_mods.csv', index=False)

    return search_df, mods_df
