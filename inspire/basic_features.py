""" Functions for calculating basic features that do not require spectral data,
    retention time predictors, or binding affinity.
"""
import polars as pl

from inspire.constants import (
    MASS_DIFF_KEY,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_SEQ_KEY,
    SEQ_LEN_KEY,
)
from inspire.spectral_features import fetch_mod_weight_dict

def _check_reps(peptide):
    """ Function to count the number of repeated residues
        within a peptide.

    Parameters
    ----------
    peptide : str
        The peptide sequence.

    Returns
    -------
    n_reps : int
        The number of repeating residues.
    """
    n_reps = 0
    for idx in range(len(peptide)-1):
        if peptide[idx] == peptide[idx+1]:
            n_reps+=1
    return n_reps

def _add_modification_weights(df_row, mod_weight_dict):
    """ Function to add the wight of modifications to a peptide's average residue weight.

    Parameters
    ----------
    df_row : dict or pl.Series
        Colection including ptm_seq, avgResidueMass, and sequenceLength.
    mod_weight_dict : dict
        A dictionary mapping modification IDs to their molecular weight.

    Returns
    -------
    averageResidueWeight : float
        A value for the mean weight of residues including modifications.
    """
    if not isinstance(df_row[PTM_SEQ_KEY], str):
        return df_row['avgResidueMass']

    mod_wt_av = sum(
        mod_weight_dict.get(
            aa_char, 0.0
        ) for aa_char in df_row[PTM_SEQ_KEY]
    )/df_row[SEQ_LEN_KEY]

    return df_row['avgResidueMass'] + mod_wt_av

def create_basic_features(search_df, mods_df):
    """ Function to create the basic features used by percolator/mokapot.

    Parameters
    ----------
    search_df : pd.DataFrame
        A DataFrame of ms search results.

    Returns
    -------
    search_df : pd.DataFrame
        The input DataFrame with update columns containing features
        for percolator/mokapot.
    """
    seq_len_mean = pl.mean(search_df[SEQ_LEN_KEY])

    search_df = search_df.with_columns(
        (pl.col(SEQ_LEN_KEY) - seq_len_mean).abs().alias('seqLenMeanDiff')
    )

    search_df = search_df.with_columns(
        pl.col(MASS_DIFF_KEY).abs().alias('absMassDiff')
    )

    if PTM_IS_VAR_KEY in mods_df.columns:
        mods_df = mods_df[mods_df[PTM_IS_VAR_KEY]]

    if not mods_df.empty:
        mod_ids = [str(x) for x in mods_df[PTM_ID_KEY].tolist()]
        search_df = search_df.with_columns(
            pl.col(PTM_SEQ_KEY).apply(
                lambda x : len(
                    [char for char in x if char in mod_ids]
                ) if isinstance(x, str) else 0,
                skip_nulls=False,
            ).alias('nVarMods')
        )

        mod_weight_dict = fetch_mod_weight_dict(mods_df)
        search_df = search_df.with_columns(
            pl.struct([PTM_SEQ_KEY, SEQ_LEN_KEY, 'avgResidueMass']).apply(
                lambda x : _add_modification_weights(x, mod_weight_dict),
                skip_nulls=False,
            ).alias('avgResidueMass')
        )
    else:
        search_df = search_df.with_columns(
            pl.lit(0).alias('nVarMods')
        )

    search_df = search_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(_check_reps).alias('nRepeatedResidues')
    )

    search_df = search_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(lambda x : len(set(x))/len(x)).alias('fracUnique')
    )

    search_df = search_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(lambda x : x.count('C')/len(x)).alias('fracC')
    )

    search_df = search_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(
            lambda x : (x.count('K') + x.count('R'))/(len(x) - 1)
        ).alias('fracKR')
    )


    return search_df
