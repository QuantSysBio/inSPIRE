""" Generic functions for reading in any search results.
"""
from inspire.input.mascot import read_mascot_data
from inspire.input.maxquant import read_mq_data
from inspire.input.peaks import read_peaks_data

def generic_read_df(config):
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
    if config.search_engine == 'mascot':
        search_df, mods_df = read_mascot_data(
            config.search_results,
            config.scan_title_format,
            config.source_files,
            config.reduce,
        )
    elif config.search_engine == 'maxquant':
        search_df, mods_df = read_mq_data(config.search_results)
    elif config.search_engine == 'peaks':
        search_df, mods_df = read_peaks_data(config.search_results)
    else:
        raise ValueError(f'Unknown Search Engine: {config.search_engine}')

    return search_df, mods_df
