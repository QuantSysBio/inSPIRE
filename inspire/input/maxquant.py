""" Functions for reading in MaxQuant search results.
"""
import pandas as pd

from inspire.constants import (
    ACCESSION_KEY,
    CHARGE_KEY,
    DELTA_SCORE_KEY,
    ENGINE_SCORE_KEY,
    KNOWN_PTM_WEIGHTS,
    LABEL_KEY,
    MASS_DIFF_KEY,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    RT_KEY,
    SCAN_KEY,
    SEQ_LEN_KEY,
    SOURCE_KEY,
)
from inspire.utils import filter_for_prosit

# Define the relevant column names from MaxQuant search results.
MQ_ACCESSION_KEY = 'Proteins'
MQ_CHARGE_KEY = 'Charge'
MQ_DECOY_KEY = 'Reverse'
MQ_DELTA_SCORE_KEY = 'Delta score'
MQ_LEN_KEY = 'Length'
MQ_MASS_KEY = 'Mass'
MQ_MISSED_CLEAVAGES_KEY = 'Missed cleavages'
MQ_MOD_SEQ_KEY = 'Modified sequence'
MQ_MODS_KEY = 'Modifications'
MQ_MZ_ERROR = 'Simple mass error [ppm]'
MQ_INTENSITY_KEY = 'Precursor Intensity'
MQ_RT_KEY = 'Retention time'
MQ_SCAN_KEY = 'Scan number'
MQ_SCORE_KEY = 'Score'
MQ_SEQ_KEY = 'Sequence'
MQ_SOURCE_KEY = 'Raw file'
MQ_RELEVANT_COLS = [
    MQ_ACCESSION_KEY,
    MQ_CHARGE_KEY,
    MQ_DECOY_KEY,
    MQ_DELTA_SCORE_KEY,
    MQ_INTENSITY_KEY,
    MQ_LEN_KEY,
    MQ_MASS_KEY,
    MQ_MISSED_CLEAVAGES_KEY,
    MQ_MOD_SEQ_KEY,
    MQ_MODS_KEY,
    MQ_MZ_ERROR,
    MQ_SCAN_KEY,
    MQ_SCORE_KEY,
    MQ_SOURCE_KEY,
    MQ_SEQ_KEY,
    MQ_RT_KEY,
]


def _create_mq_mods_df(mq_df):
    """ Helper function to create a DataFrame of the unique modifications found in
        MaxQuant search results.

    Parameters
    ----------
    mq_df : pd.DataFrame
        A DataFrame of MaxQuant search results.

    Returns
    -------
    mods_df : pd.DataFrame
        A small DataFrame containing the unique modifications found.
    """
    unique_mods = mq_df[MQ_MODS_KEY].apply(
        lambda x : x.split(',')
    ).explode().unique().tolist()
    unique_mods = list({x if x[1] != ' ' else x[2:] for x in unique_mods})
    unique_mods.remove('Unmodified')
    mods_df = pd.DataFrame({
        PTM_NAME_KEY: pd.Series(unique_mods),
        PTM_WEIGHT_KEY: pd.Series([KNOWN_PTM_WEIGHTS[mod] for mod in unique_mods]),
    })
    mods_df = mods_df.sort_values(by=PTM_NAME_KEY)
    mods_df.reset_index(drop=True, inplace=True)
    mods_df[PTM_ID_KEY] = mods_df.index + 1
    mods_df[PTM_IS_VAR_KEY] = True
    return mods_df

def _create_ptm_seq_col(modified_seq, unqiue_mods):
    """ Helper function to take MaxQuant modified peptide and return
        the sequence representing the ptms.

    Parameters
    ----------
    modified_seq : str
        A modified peptide sequence in MaxQuant format.
    unqiue_mods : dict
        A dictionary mapping ptm names to integer ids.

    Returns
    -------
    ptm_seq : str
        The ptm sequence by ID.
    """
    ptm_seq = ''
    if '(' not in modified_seq:
        return None
    split_seq = modified_seq.split('_')
    if split_seq[0]:
        ptm_seq += f'{unqiue_mods[split_seq[0]]}.'
        main_seq = split_seq[1]
    elif split_seq[1].startswith('('):
        main_seq = split_seq[1]
        main_seq = main_seq[1:]
        end_mod = main_seq.index(')') + 1
        mod = main_seq[:end_mod]
        main_seq = main_seq[end_mod + 1:]
        ptm_seq += f'{str(unqiue_mods[mod])}.'
    else:
        ptm_seq += '0.'
        main_seq = split_seq[1]
    while main_seq:
        assert main_seq[0] != '('
        if len(main_seq) == 1:
            ptm_seq += '0'
            break

        if main_seq[1] != '(':
            ptm_seq += '0'
            main_seq = main_seq[1:]
        else:
            main_seq = main_seq[2:]
            end_mod = main_seq.index(')') + 1
            mod = main_seq[:end_mod]
            main_seq = main_seq[end_mod + 1:]
            ptm_seq += str(unqiue_mods[mod])

    if split_seq[2]:
        ptm_seq += f'.{unqiue_mods[split_seq[2]]}'
    else:
        ptm_seq += '.0'

    return ptm_seq

def read_single_mq_data(mq_data):
    """ Function to read in MaxQuant search results from a single file.

    Parameters
    ----------
    df_loc : str
        A location of MaxQuant search results.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    mq_df = pd.read_csv(
        mq_data,
        sep='\t',
        usecols=MQ_RELEVANT_COLS,
    )

    # Separate PTMs.
    mods_df = _create_mq_mods_df(mq_df)
    var_mod_dict = dict(zip(mods_df[PTM_NAME_KEY].tolist(), mods_df[PTM_ID_KEY].tolist()))
    mq_df[PTM_SEQ_KEY] = mq_df[MQ_MOD_SEQ_KEY].apply(
        lambda x : _create_ptm_seq_col(x, var_mod_dict)
    )

    # Rename to match inSPIRE naming scheme.
    mq_df = mq_df.rename(
        columns={
            MQ_CHARGE_KEY: CHARGE_KEY,
            MQ_DELTA_SCORE_KEY: DELTA_SCORE_KEY,
            MQ_LEN_KEY: SEQ_LEN_KEY,
            MQ_MZ_ERROR: MASS_DIFF_KEY,
            MQ_RT_KEY: RT_KEY,
            MQ_SCAN_KEY: SCAN_KEY,
            MQ_SCORE_KEY: ENGINE_SCORE_KEY,
            MQ_SEQ_KEY: PEPTIDE_KEY,
            MQ_SOURCE_KEY: SOURCE_KEY,
            MQ_MISSED_CLEAVAGES_KEY: 'missedCleavages',
            MQ_INTENSITY_KEY: 'ms1Intensity',
        }
    )

    # Filter for Prosit, clean up accession data, and add label.
    mq_df = filter_for_prosit(mq_df)

    mq_df[ACCESSION_KEY] = mq_df[[MQ_ACCESSION_KEY, MQ_DECOY_KEY]].apply(
        lambda x : 'reverseSeq' if x[MQ_DECOY_KEY] == '+' else x[MQ_ACCESSION_KEY],
        axis=1
    )
    mq_df['scanCounts'] = mq_df.groupby([SOURCE_KEY, SCAN_KEY])[ENGINE_SCORE_KEY].transform('count')
    mq_df['fromChimera'] = mq_df['scanCounts'].apply(lambda x : 1 if x > 1 else 0)

    mq_df['avgResidueMass'] = mq_df[[MQ_MASS_KEY, SEQ_LEN_KEY]].apply(
        lambda x : x[MQ_MASS_KEY]/x[SEQ_LEN_KEY],
        axis=1,
    )
    mq_df[LABEL_KEY] = mq_df[MQ_DECOY_KEY].apply(
        lambda x : -1 if x == '+' else 1
    )
    mq_df = mq_df.drop(
        [
            MQ_ACCESSION_KEY,
            MQ_MOD_SEQ_KEY,
            MQ_DECOY_KEY,
            MQ_MASS_KEY,
            MQ_MODS_KEY,
            'scanCounts',
        ],
        axis=1,
    )
    mq_df = mq_df.dropna(subset=[PEPTIDE_KEY, ACCESSION_KEY])

    return mq_df, mods_df

def read_mq_data(mq_data):
    """ Function to read in MaxQuant search results from one or more files.

    Parameters
    ----------
    peaks_data : str or list of str
        A single location of MaxQuant search results or a list of locations.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    if isinstance(mq_data, list):
        hits_dfs = []
        variable_mods_dfs = []

        for mq_file in mq_data:
            hits_df, mods_df = read_single_mq_data(mq_file)
            hits_dfs.append(hits_df)
            variable_mods_dfs.append(mods_df)

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pd.concat(hits_dfs)
        for i in range(len(variable_mods_dfs)-1):
            assert variable_mods_dfs[i].equals(variable_mods_dfs[i+1])
        return hits_df, variable_mods_dfs[0]

    return read_single_mq_data(mq_data)
