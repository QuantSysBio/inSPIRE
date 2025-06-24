""" Generic functions for reading in any search results.
"""
import multiprocessing
import os

import pandas as pd
import polars as pl
from sklearn.linear_model import TheilSenRegressor

from inspire.accession import process_accession_groups
from inspire.input.casanovo import read_casanovo
from inspire.input.mascot import read_mascot_data
from inspire.input.maxquant import read_mq_data
from inspire.input.msfragger import read_ms_fragger_data
from inspire.input.peaks import read_peaks_data
from inspire.input.peaks_de_novo import read_peaks_de_novo
from inspire.input.psms import read_psms
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
        search_df = search_df.with_columns(pl.col('source').cast(pl.String))
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
        elif config.search_engine == 'peaksDeNovo':
            search_df, mods_df = read_peaks_de_novo(config.search_results)
            search_df = search_df.drop([
                'Source File',
                'Scan',
                'local confidence (%)',
            ])
        elif config.search_engine == 'casanovo':
            search_df, mods_df = read_casanovo(
                config.search_results,
                config.scans_folder,
                config.scans_format,
            )
        elif config.search_engine == 'msfragger':
            search_df, mods_df = read_ms_fragger_data(
                config.search_results,
                config.fixed_modifications,
                n_cores,
                reduce_results,
            )
        elif config.search_engine == 'psms':
            search_df, mods_df = read_psms(config.search_results)
        else:
            raise ValueError(f'Unknown Search Engine: {config.search_engine}')

        if config.replace_il:
            search_df = search_df.with_columns(
                pl.col('peptide').str.replace_all('I', 'L')
            )
            search_df = search_df.unique(
                subset=['source', 'scan', 'peptide'], maintain_order=True
            )

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

        if config.additional_psms is not None:
            additional_psm_df, additional_mods_df = read_psms(config.additional_psms)
            additional_psm_df = additional_psm_df.rename({'engineScore': 'psmScore'})
            additional_psm_df = additional_psm_df.join(
                search_df.select(['source', 'scan', 'peptide', 'engineScore']),
                how='left', on=['source', 'scan', 'peptide',],
            )
            scored_df = additional_psm_df.filter(pl.col('engineScore').is_not_null())
            unscored_df = additional_psm_df.filter(pl.col('engineScore').is_null())
            reg = TheilSenRegressor().fit(
                scored_df.select(['psmScore', 'sequenceLength']).to_numpy(),
                scored_df['engineScore'].to_numpy(),
            )
            unscored_df = unscored_df.with_columns(
                pl.Series(
                    reg.predict(unscored_df.select(['psmScore', 'sequenceLength']).to_numpy())
                ).alias('engineScore')
            )
            for col in search_df.columns:
                if col not in unscored_df.columns:
                    search_df = search_df.drop(col)
                else:
                    unscored_df = unscored_df.with_columns(
                        pl.col(col).cast(search_df[col].dtype)
                    )
            search_df = pl.concat([search_df, unscored_df.select(search_df.columns)])
            mods_cols = ['Identifier', 'Name', 'Delta', 'isVar']
            mods_df = pd.merge(
                mods_df[mods_cols], additional_mods_df[mods_cols], how='outer', on=mods_cols,
            )
            mods_df = mods_df.drop_duplicates(subset=['Identifier', 'Delta'])

        if save_dfs and config.reuse_input:
            search_df.write_csv(f'{config.output_folder}/formated_df.csv')
            mods_df.to_csv(f'{config.output_folder}/formated_mods.csv', index=False)

    return search_df, mods_df
