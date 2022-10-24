""" Functions for reading in PEAKS search results.
"""
from copy import deepcopy

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
    PROTON,
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
from inspire.utils import filter_for_prosit, remove_source_suffixes

# Define the relevant column names from PEAKS DB search results.
PEAKS_ACCESSION_KEY = 'Accession'
PEAKS_ASCORE_KEY = 'AScore'
PEAKS_CHARGE_KEY = 'Z'
PEAKS_CHIMERA_KEY = 'from Chimera'
PEAKS_LEN_KEY = 'Length'
PEAKS_MASS_KEY = 'Mass'
PEAKS_MZ_KEY = 'm/z'
PEAKS_PEPTIDE_KEY = 'Peptide'
PEAKS_PTM_KEY = 'PTM'
PEAKS_RETENTION_TIME_KEY = 'RT'
PEAKS_SCAN_KEY = 'Scan'
PEAKS_SCORE_KEY = '-10lgP'
PEAKS_SOURCE_KEY = 'Source File'
PEAKS_INTENSITY_KEY = 'Area'
PEAKS_RELEVANT_COLUMNS = [
    PEAKS_ACCESSION_KEY,
    PEAKS_ASCORE_KEY,
    PEAKS_CHARGE_KEY,
    PEAKS_CHIMERA_KEY,
    PEAKS_LEN_KEY,
    PEAKS_INTENSITY_KEY,
    PEAKS_MASS_KEY,
    PEAKS_MZ_KEY,
    PEAKS_PEPTIDE_KEY,
    PEAKS_PTM_KEY,
    PEAKS_RETENTION_TIME_KEY,
    PEAKS_SCAN_KEY,
    PEAKS_SCORE_KEY,
    PEAKS_SOURCE_KEY,
]

def _extract_terminus_ptms(mod_list, var_mods, terminus):
    """ Helper function to extract any ptms at the N or C termini.

    Parameters
    ----------
    mod_list : list of str
        A list of the ptms for the sequence.
    var_mods : dict
        A dictionary mapping ptm names to integer IDs.
    terminus : str
        A flag for the terminus (either N or C).

    Returns
    -------
    mod_list : list of str
        The updated list of the ptms for the sequence without the terminus modification.
    term_flag : str
        A ID of the terminus ptm.
    """
    term_mod = -1
    term_flag = '0'
    for mod in mod_list:
        if mod.endswith(f'({terminus}-term)'):
            term_mod = mod
            break
    if term_mod != -1:
        term_flag = f'{var_mods[term_mod]}'
        mod_list.remove(term_mod)
    return mod_list, term_flag

def _separate_peaks_ptms(df_row, var_mods):
    """ Helper function to remove ptm markers from the Peaks Peptide column and create a
        separate ptm_seq column containing ptm data.

    Parameters
    ----------
    df_row : pd.Series
        A row of the Peaks search results DataFrame.
    var_mods : dict
        A dictionary mapping ptm names to integer ids.

    Returns
    -------
    df_row : pd.Series
        The updated row with peptide and ptms separated.
    """
    peptide = df_row[PEAKS_PEPTIDE_KEY]
    ptms = df_row[PEAKS_ASCORE_KEY]
    ptm_prefix = '0'
    ptm_suffix = '0'

    if isinstance(ptms, str):
        ptm_list = [a.strip(' ') for a in ptms.split(';')]
        ptm_list = [b.split(':')[1].strip(' ') for b in ptm_list]
    else:
        df_row[PEPTIDE_KEY] = peptide
        df_row[PTM_SEQ_KEY] = None
        return df_row
    pep_seq = ''
    ptm_seq = ''
    idx = 0

    while peptide:
        assert peptide[0] != '('
        if len(peptide) == 1:
            pep_seq += peptide[0]
            ptm_seq += '0'
            break

        if peptide[1] != '(':
            pep_seq += peptide[0]
            ptm_seq += '0'
            peptide = peptide[1:]
        else:
            pep_seq += peptide[0]
            peptide = peptide[1:]

            end_mod = peptide.index(')') + 1
            mod_list = []
            while peptide[0] == '(':
                end_mod = peptide.index(')') + 1

                mod_list.append(ptm_list[0])
                ptm_list = ptm_list[1:]

                peptide = peptide[end_mod:]
                if not peptide:
                    break

            if idx == 0:
                mod_list, ptm_prefix = _extract_terminus_ptms(mod_list, var_mods, 'N')

            if not peptide:
                mod_list, ptm_suffix = _extract_terminus_ptms(mod_list, var_mods, 'C')

            if mod_list:
                assert len(mod_list) == 1
                ptm_seq += str(var_mods[mod_list[0]])
            else:
                ptm_seq += '0'
        idx += 1

    df_row[PEPTIDE_KEY] = pep_seq
    df_row[PTM_SEQ_KEY] = '.'.join([ptm_prefix, ptm_seq, ptm_suffix])
    return df_row

def _collect_peaks_var_mods(peaks_df):
    """ Helper function to collect all of the ptms present in PEAKS DB search results.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        A DataFrame of PEAKS DB search results.

    Returns
    -------
    ptms_df : pd.DataFrame
        A small DataFrame listing the unique ptms found in the data.
    """
    var_mod_df = deepcopy(peaks_df[[PEAKS_PTM_KEY, PEAKS_PEPTIDE_KEY]])
    var_mod_df['ptm_names'] = var_mod_df[PEAKS_PTM_KEY].apply(
        lambda x: None if not isinstance(x, str) else [a.strip(' ') for a in x.split(';')]
    )
    var_mod_df['peaks_ptm_weights'] = var_mod_df[PEAKS_PEPTIDE_KEY].replace(
        to_replace=r'[A-Za-z ]',
        value='',
        regex=True,
    ).apply(lambda x : [a.strip('(').strip(')') for a in x.split(')(')])

    var_mod_df = var_mod_df[
        (var_mod_df['ptm_names'].notnull()) & (var_mod_df['peaks_ptm_weights'].notnull())
    ]

    if not var_mod_df.shape[0]:
        return pd.DataFrame({
            PTM_NAME_KEY: [],
            PTM_WEIGHT_KEY: [],
            PTM_ID_KEY: [],
        })

    var_mod_df['combined_ptm_data'] = var_mod_df.apply(
        lambda x : list(zip(x['ptm_names'], x['peaks_ptm_weights'])),
        axis=1
    )

    ptms = var_mod_df['combined_ptm_data'].explode().unique()

    ptm_names = [name for name, _ in ptms]

    ptm_weights = [KNOWN_PTM_WEIGHTS.get(name, float(weight)) for name, weight in ptms]

    ptms_df = pd.DataFrame({
        PTM_NAME_KEY: pd.Series(ptm_names),
        PTM_WEIGHT_KEY: pd.Series(ptm_weights),
    })
    ptms_df = ptms_df.drop_duplicates(PTM_NAME_KEY)

    ptms_df = ptms_df.sort_values(by=PTM_NAME_KEY)
    ptms_df.reset_index(inplace=True, drop=True)
    ptms_df[PTM_ID_KEY] = ptms_df.index +1
    ptms_df[PTM_IS_VAR_KEY] = True

    return ptms_df

def read_single_peaks_data(df_loc):
    """ Function to read in PEAKS DB search results from a single file.

    Parameters
    ----------
    df_loc : str
        A location of PEAKS DB search results.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    peaks_df = pd.read_csv(df_loc, usecols=PEAKS_RELEVANT_COLUMNS)

    # Separate PTMs.
    var_mods = _collect_peaks_var_mods(peaks_df)
    var_mod_dict = dict(zip(var_mods[PTM_NAME_KEY].tolist(), var_mods[PTM_ID_KEY].tolist()))

    if var_mod_dict:
        peaks_df = peaks_df.apply(
            lambda x: _separate_peaks_ptms(x, var_mod_dict),
            axis=1
        )
    else:
        peaks_df = peaks_df.rename( # pylint: disable=no-member
            columns={PEAKS_PEPTIDE_KEY: PEPTIDE_KEY}
        )
        peaks_df[PTM_SEQ_KEY] = None

    # Rename to match inSPIRE naming scheme.
    peaks_df = peaks_df.rename(columns={
        PEAKS_ACCESSION_KEY: ACCESSION_KEY,
        PEAKS_CHARGE_KEY: CHARGE_KEY,
        PEAKS_LEN_KEY: SEQ_LEN_KEY,
        PEAKS_RETENTION_TIME_KEY: RT_KEY,
        PEAKS_SCORE_KEY: ENGINE_SCORE_KEY,
        PEAKS_INTENSITY_KEY: 'ms1Intensity',
    })

    # Filter for Prosit and add feature columns not present.
    peaks_df['fromChimera'] = peaks_df[PEAKS_CHIMERA_KEY].apply(
        lambda x : 1 if x == 'Yes' else 0
    )

    peaks_df[ACCESSION_KEY].fillna('unknown', inplace=True)

    peaks_df = filter_for_prosit(peaks_df)
    adjusted_masses = peaks_df[PEAKS_MASS_KEY] + peaks_df[CHARGE_KEY]*PROTON
    peaks_df[MASS_DIFF_KEY] = (
        (peaks_df[PEAKS_MZ_KEY]*peaks_df[CHARGE_KEY]) - adjusted_masses
    )
    peaks_df['avgResidueMass'] = peaks_df[PEAKS_MASS_KEY]/peaks_df[SEQ_LEN_KEY]
    peaks_df[DELTA_SCORE_KEY] = 0
    peaks_df['missedCleavages'] = 0

    # Clean source and scan columns if required, add label.
    peaks_df[SOURCE_KEY] = peaks_df[PEAKS_SOURCE_KEY].apply(
        remove_source_suffixes
    )
    peaks_df[SCAN_KEY] = peaks_df[PEAKS_SCAN_KEY].apply(
        lambda x : x if isinstance(x, int) else int(x.split(':')[-1])
    )
    peaks_df[LABEL_KEY] = peaks_df[ACCESSION_KEY].apply(
        lambda x : -1 if isinstance(x, str) and '#DECOY#' in x else 1
    )

    peaks_df = peaks_df.drop(
        [
            PEAKS_ASCORE_KEY,
            PEAKS_CHIMERA_KEY,
            PEAKS_MASS_KEY,
            PEAKS_MZ_KEY,
            PEAKS_PTM_KEY,
            PEAKS_PEPTIDE_KEY,
            PEAKS_SCAN_KEY,
            PEAKS_SOURCE_KEY,
        ],
        axis=1,
    )

    return peaks_df, var_mods

def read_peaks_data(peaks_data):
    """ Function to read in PEAKS DB search results from one or more files.

    Parameters
    ----------
    peaks_data : str or list of str
        A single location of PEAKS DB search results or a list of locations.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    if isinstance(peaks_data, list):
        peaks_dfs = []
        variable_mods_dfs = []
        for mq_file in peaks_data:
            hits_df, mods_df = read_single_peaks_data(mq_file)
            peaks_dfs.append(hits_df)
            variable_mods_dfs.append(mods_df)

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pd.concat(peaks_dfs)
        for i in range(len(variable_mods_dfs)-1):
            assert variable_mods_dfs[i].equals(variable_mods_dfs[i+1])

        return hits_df, variable_mods_dfs[0]

    return read_single_peaks_data(peaks_data)
