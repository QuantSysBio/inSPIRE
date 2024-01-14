""" Functions for reading in PEAKS search results.
"""
from copy import deepcopy

import pandas as pd
import polars as pl

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
PEAKS_PPM_KEY = 'ppm'
PEAKS_PTM_KEY = 'PTM'
PEAKS_RETENTION_TIME_KEY = 'RT'
PEAKS_SCAN_KEY = 'Scan'
PEAKS_SCORE_KEY = '-10lgP'
PEAKS_SOURCE_KEY = 'Source File'
PEAKS_RELEVANT_COLUMNS = [
    PEAKS_ACCESSION_KEY,
    PEAKS_ASCORE_KEY,
    PEAKS_CHARGE_KEY,
    PEAKS_CHIMERA_KEY,
    PEAKS_LEN_KEY,
    PEAKS_MASS_KEY,
    PEAKS_MZ_KEY,
    PEAKS_PEPTIDE_KEY,
    PEAKS_PPM_KEY,
    PEAKS_PTM_KEY,
    PEAKS_RETENTION_TIME_KEY,
    PEAKS_SCAN_KEY,
    PEAKS_SCORE_KEY,
    PEAKS_SOURCE_KEY,
]

ID_NUMBERS = {
    'Deamidation (NQ)': 7,
    'Deamidation (N)': 8,
    'Deamidation (Q)': 7,
    'Phospho (S)': 6,
    'Phospho (T)': 5,
    'Phospho (Y)': 4,
    'Acetylation (N-term)': 3,
    'Acetylation (Protein N-term)': 3,
    'Oxidation (M)': 2,
    'Carbamidomethyl (C)': 1,
    'Carbamidomethylation': 1,
}

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

def separate_peaks_ptms(df_row, var_mods):
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
    results = {}
    peptide = df_row[PEAKS_PEPTIDE_KEY]
    ptms = df_row[PEAKS_ASCORE_KEY]
    ptm_prefix = '0'
    ptm_suffix = '0'

    if isinstance(ptms, str):
        ptm_list = [a.strip(' ') for a in ptms.split(';')]
        ptm_list = [b.split(':')[1].strip(' ') for b in ptm_list]
    else:
        results[PEPTIDE_KEY] = peptide
        results[PTM_SEQ_KEY] = None
        return results
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
                ptm_seq += str(var_mods[mod_list[0]])
            else:
                ptm_seq += '0'
        idx += 1

    results[PEPTIDE_KEY] = pep_seq
    results[PTM_SEQ_KEY] = '.'.join([ptm_prefix, ptm_seq, ptm_suffix])
    return results

def collect_peaks_var_mods(peaks_df):
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
    var_mod_df = var_mod_df.with_columns(
        pl.col(PEAKS_PTM_KEY).apply(
            lambda x: None if not isinstance(x, str) else [a.strip(' ') for a in x.split(';')],
            skip_nulls=False,
        ).alias('ptm_names')
    )

    var_mod_df = var_mod_df.with_columns(
        pl.col(PEAKS_PEPTIDE_KEY).str.replace_all(r'[A-Za-z]', '').apply(
            lambda x : [a.strip('(').strip(')') for a in x.split(')(')]
        ).alias('peaks_ptm_weights')
    )

    var_mod_df = var_mod_df.filter(
        (pl.col('ptm_names').is_not_null()) & (pl.col('peaks_ptm_weights').is_not_null())
    )

    if not var_mod_df.shape[0]:
        return pd.DataFrame({
            PTM_NAME_KEY: [],
            PTM_WEIGHT_KEY: [],
            PTM_ID_KEY: [],
        })

    var_mod_df = var_mod_df.with_columns(
        pl.struct(['ptm_names', 'peaks_ptm_weights']).apply(
            lambda x : ["~".join(y) for y in zip(x['ptm_names'], x['peaks_ptm_weights'])]
        ).alias('combined_ptm_data')
    )

    ptms = [x.split('~') for x in var_mod_df['combined_ptm_data'].explode().unique()]

    ptm_names = [name for name, _ in ptms]

    ptm_weights = [KNOWN_PTM_WEIGHTS.get(name, float(weight)) for name, weight in ptms]

    ptms_df = pd.DataFrame({
        PTM_NAME_KEY: pd.Series(ptm_names),
        PTM_WEIGHT_KEY: pd.Series(ptm_weights),
    })
    ptms_df = ptms_df.drop_duplicates(PTM_NAME_KEY)

    ptms_df[PTM_ID_KEY] = ptms_df[PTM_NAME_KEY].apply(
        lambda x : ID_NUMBERS.get(x, 9)
    )
    ptms_df = ptms_df.sort_values(by=PTM_ID_KEY).reset_index(drop=True)
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
    peaks_df = pl.read_csv(df_loc, columns=PEAKS_RELEVANT_COLUMNS)

    # Separate PTMs.
    var_mods = collect_peaks_var_mods(peaks_df)
    var_mod_dict = dict(zip(var_mods[PTM_NAME_KEY].tolist(), var_mods[PTM_ID_KEY].tolist()))

    if var_mod_dict:
        peaks_df = peaks_df.with_columns(
            pl.struct([PEAKS_PEPTIDE_KEY, PEAKS_ASCORE_KEY]).apply(
                lambda x: separate_peaks_ptms(x, var_mod_dict)
            ).alias('results')
        ).unnest('results')
        peaks_df = peaks_df.drop(PEAKS_PEPTIDE_KEY)
    else:
        peaks_df = peaks_df.rename(
            {PEAKS_PEPTIDE_KEY: PEPTIDE_KEY}
        )
        peaks_df = peaks_df.with_columns(
            pl.lit(None).alias(PTM_SEQ_KEY)
        )

    # Rename to match inSPIRE naming scheme.
    peaks_df = peaks_df.rename({
        PEAKS_ACCESSION_KEY: ACCESSION_KEY,
        PEAKS_CHARGE_KEY: CHARGE_KEY,
        PEAKS_LEN_KEY: SEQ_LEN_KEY,
        PEAKS_RETENTION_TIME_KEY: RT_KEY,
        PEAKS_SCORE_KEY: ENGINE_SCORE_KEY,
        PEAKS_PPM_KEY: MASS_DIFF_KEY,
    })

    # Filter for Prosit and add feature columns not present.
    peaks_df = peaks_df.with_columns(
        pl.col(PEAKS_CHIMERA_KEY).apply(
            lambda x : 1 if x == 'Yes' else 0
        ).alias('fromChimera'),
        pl.col(ACCESSION_KEY).fill_null(
            pl.lit('unknown'),
        ).alias(ACCESSION_KEY),
    )

    peaks_df = filter_for_prosit(peaks_df)
    peaks_df = peaks_df.with_columns(
        (pl.col(PEAKS_MASS_KEY)/pl.col(SEQ_LEN_KEY)).alias('avgResidueMass'),
        pl.lit(0).alias(DELTA_SCORE_KEY),
        pl.lit(0).alias('missedCleavages'),
        pl.col(PEAKS_SOURCE_KEY).apply(remove_source_suffixes).alias(SOURCE_KEY),
        pl.col(PEAKS_SCAN_KEY).apply(
            lambda x : x if isinstance(x, int) else int(x.split(':')[-1])
        ).alias(SCAN_KEY),
        pl.col(ACCESSION_KEY).apply(
            lambda x : -1 if isinstance(x, str) and ('DECOY' in x or 'rev' in x) else 1
        ).alias(LABEL_KEY),
    )

    for col, dtype in peaks_df.schema.items():
        if dtype == pl.Null:
            peaks_df = peaks_df.with_columns(
                pl.lit('').alias(col)
            )

    peaks_df = peaks_df.drop(
        [
            PEAKS_ASCORE_KEY,
            PEAKS_CHIMERA_KEY,
            PEAKS_MASS_KEY,
            PEAKS_MZ_KEY,
            PEAKS_PTM_KEY,
            PEAKS_SCAN_KEY,
            PEAKS_SOURCE_KEY,
        ]
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
        for peaks_file in peaks_data:
            hits_df, mods_df = read_single_peaks_data(peaks_file)
            peaks_dfs.append(hits_df)
            variable_mods_dfs.append(mods_df)

        peaks_dfs = [x.select(peaks_dfs[0].columns) for x in peaks_dfs]

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pl.concat(peaks_dfs)

        return hits_df, variable_mods_dfs[0]

    return read_single_peaks_data(peaks_data)
