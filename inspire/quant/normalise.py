""" Function to normalise quantifications by Skyline across raw files.

    Intermeidate columns created during execution of this script:
        - source_valid : 1/0 flag indicating if there is a valid measurement in this source file.
        - source_clean : raw values with null replacement if the measurement is invalid.
        - source_norm : the final normalised value for that source file.
"""
from math import log2
import numpy as np
import pandas as pd

from inspire.constants import PEPTIDE_KEY, ACCESSION_KEY

from inspire.quant.utils import plot_correlations, plot_quant_clustermap, plot_distros

AVERAGE_QUANT_KEY = 'average'
N_VALID_QUANT_KEY = 'nValid'

def normalise_intensities(config):
    """ Function to normalise intensities across RAW files with robust regression.

    Parameters
    ----------
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    # Read in quantification data
    quant_df = pd.read_csv(
        f'{config.output_folder}/quant/quantified_per_file.csv',
    )
    meta_df = pd.read_csv(
        f'{config.output_folder}/quant/metadata.csv'
    )
    sources = meta_df['renamed'].tolist()

    # Find the valid quantifications.
    quant_df = count_valid_and_identified_quantifications(quant_df, sources, config)

    # Fill invalid quantifications with null.
    quant_df = clean_quantification_values(quant_df, sources, config)
    quant_df[
        [PEPTIDE_KEY, ACCESSION_KEY] + [f'{source}_clean' for source in sources]
    ].to_csv(f'{config.output_folder}/quant/cleaned_quantification.csv', index=False)

    # Select relevant columns.
    quant_df = quant_df[
        [PEPTIDE_KEY, ACCESSION_KEY, N_VALID_QUANT_KEY]
        + [f'{source}_clean' for source in sources]
        + [f'{source}_valid' for source in sources]
    ]

    for source in sources:
        quant_df[f'{source}_clean'] = quant_df[f'{source}_clean'].apply(
            lambda intensity : log2(intensity) if (
                intensity is not None and intensity > 0
            ) else None
        )

    # Calculate average.
    quant_df[AVERAGE_QUANT_KEY] = quant_df[
        [f'{source}_clean' for source in sources]
    ].mean(axis=1, skipna=True, numeric_only=True)

    global_median = quant_df[[f'{source}_clean' for source in sources]].median(axis=None)
    for source in sources:
        quant_df = equalize_medians(quant_df, source, global_median)

    plot_distros(quant_df, config, sources)

    # Write output to csv.
    quant_df[
        [PEPTIDE_KEY, ACCESSION_KEY] + [f'{source}_norm' for source in sources]
    ].to_csv(f'{config.output_folder}/quant/normalised_quantification.csv', index=False)

    # Plot correlations between normalised intensities.
    plot_correlations('normalised_quantification', 'norm', config)
    plot_quant_clustermap(config)


def count_valid_and_identified_quantifications(quant_df, sources, config):
    """ Function find the number of valid quantifications where peptide also
        identified in that file.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of quantified peptides.
    sources : list of str
        List of the raw files where the peptide was quantified.
    config : inspire.config.Config
        Config object for the whole experiment.

    Returns
    -------
    quant_df : pd.DataFrame
        DataFrame of quantifications with flags indicating if they have valid quantification
        in each file.
    """
    for source in sources:
        quant_df[f'{source}_valid'] = quant_df.apply(
            lambda df_row, src=source : (
                (df_row[f'{src}_idp'] > config.skyline_idp_cut_off) &
                (df_row[f'{src}_ratio'] > config.skyline_bg_ratio_cut_off)
                # & (df_row[f'{src}_identified'] == 1)
            ),
            axis=1,
        )
    quant_df[N_VALID_QUANT_KEY] = quant_df[[f'{source}_valid' for source in sources]].sum(axis=1)

    return quant_df

def clean_quantification_values(quant_df, sources, config):
    """ Function to set invalid quantifications to null.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of quantified peptides.
    sources : list of str
        List of the raw files where the peptide was quantified.
    config : inspire.config.Config
        Config object for the whole experiment.

    Returns
    -------
    quant_df : pd.DataFrame
        DataFrame of quantifications with quantifications including null replacement for
        invalid quantifications.
    """
    for source in sources:
        quant_df[f'{source}_clean'] = quant_df.apply(
            lambda df_row, src=source : df_row[f'{src}_raw'] if (
                (df_row[f'{src}_idp'] > config.skyline_idp_cut_off) &
                (df_row[f'{src}_ratio'] > config.skyline_bg_ratio_cut_off)
                # & (df_row[f'{src}_identified'] == 1)
            ) else None,
            axis=1,
        )

    return quant_df


def equalize_medians(quant_df, source, global_median):
    """ Function to apply equalize medians for raw file normalisation of intensites
        from a given raw file.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of quantified peptides.
    source : list of str
        The raw file with intensities to be normalised.
    config : inspire.config.Config
        Config object for the whole experiment.

    Returns
    -------
    quant_df : pd.DataFrame
        DataFrame of quantifications, now including normalised intensities.
    """
    # Split between valid and invalid quantifications:
    valid_quant_df = quant_df[quant_df[f'{source}_clean'].notna()]
    invalid_quant_df = quant_df[quant_df[f'{source}_clean'].isna()]

    valid_quant_df[f'{source}_norm'] = (
        valid_quant_df[f'{source}_clean'] - valid_quant_df[f'{source}_clean'].median()
    ) + global_median

    # Fill with null for invalid samples and combine all entries
    invalid_quant_df[f'{source}_norm'] = np.nan
    quant_df = pd.concat([valid_quant_df, invalid_quant_df])

    return quant_df
