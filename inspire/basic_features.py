""" Functions for calculating basic features that do not require spectral data,
    retention time predictors, or binding affinity.
"""
import numpy as np

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
    search_df['seqLenMeanDiff'] = np.abs(
        search_df[SEQ_LEN_KEY] - search_df[SEQ_LEN_KEY].mean()
    )

    search_df['absMassDiff'] = search_df[MASS_DIFF_KEY].apply(abs)

    if PTM_IS_VAR_KEY in mods_df.columns:
        mods_df = mods_df[mods_df[PTM_IS_VAR_KEY]]

    if not mods_df.empty:
        mod_ids = [str(x) for x in mods_df[PTM_ID_KEY].tolist()]
        search_df['nVarMods'] = search_df[PTM_SEQ_KEY].apply(
            lambda x : len([
                char for char in x if char in mod_ids
            ]) if isinstance(x, str) else 0
        )
        mod_weight_dict = fetch_mod_weight_dict(mods_df)
        mod_weights = search_df.apply(
            lambda x : 0.0 if not isinstance(x[PTM_SEQ_KEY], str) else sum(
                (mod_weight_dict.get(aa_char, 0.0) for aa_char in x[PTM_SEQ_KEY])
            )/x[SEQ_LEN_KEY],
            axis=1
        )

        search_df['avgResidueMass'] += mod_weights
    else:
        search_df['nVarMods'] = 0

    search_df['nRepeatedResidues'] = search_df[PEPTIDE_KEY].apply(
        _check_reps
    )

    search_df['fracUnique'] = search_df[PEPTIDE_KEY].apply(
        lambda x : len(set(x))/len(x)
    )

    search_df['fracC'] = search_df[PEPTIDE_KEY].apply(
        lambda x : x.count('C')/len(x)
    )
    search_df['fracKR'] = search_df[PEPTIDE_KEY].apply(
        lambda x : (
            x.count('K') + x.count('R')
        )/(len(x) - 1)
    )

    return search_df
