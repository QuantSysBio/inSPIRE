""" Functions for reading PEAKS de novo results.
"""
from statistics import mean

import pandas as pd
import polars as pl

from inspire.constants import (
    ACCESSION_KEY,
    CHARGE_KEY,
    ENGINE_SCORE_KEY,
    KNOWN_PTM_WEIGHTS,
    LABEL_KEY,
    MASS_DIFF_KEY,
    DELTA_SCORE_KEY,
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

PEAKS_DN_CHARGE_KEY = 'z'
PEAKS_DN_LC_KEY = 'local confidence (%)'
PEAKS_DN_SCORE_KEY = 'Denovo score'
PEAKS_DN_MASS_KEY = 'Mass'
PEAKS_DN_MZ_KEY = 'm/z'
PEAKS_DN_PEPTIDE_KEY = 'Peptide'
PEAKS_DN_RETENTION_TIME_KEY = 'RT'
PEAKS_DN_SCAN_KEY = 'Scan'
PEAKS_DN_SOURCE_KEY = 'Source File'
PEAKS_DN_RELEVANT_COLUMNS = [
    PEAKS_DN_CHARGE_KEY,
    PEAKS_DN_LC_KEY,
    PEAKS_DN_MASS_KEY,
    PEAKS_DN_MZ_KEY,
    PEAKS_DN_PEPTIDE_KEY,
    PEAKS_DN_RETENTION_TIME_KEY,
    PEAKS_DN_SCAN_KEY,
    PEAKS_DN_SCORE_KEY,
    PEAKS_DN_SOURCE_KEY,
]

ID_NUMBERS = {
    'Cysteinylation': 5,
    'Deamidation (NQ)': 4,
    'Deamidation (N)': 4,
    'Deamidation (Q)': 4,
    'Phospho (S)': 6,
    'Phospho (T)': 6,
    'Phospho (Y)': 6,
    'Acetylation (N-term)': 3,
    'Acetylation (Protein N-term)': 3,
    'Oxidation (M)': 2,
    'Carbamidomethyl (C)': 1,
    'Carbamidomethylation': 1,
}


def de_novo_ptm_convert(mod_seq, var_mod_dict):
    """ Function to convert the PEAKS modified prosit sequence to the PTM seq used elsewhere.

    Parameters
    ----------
    mod_seq : str
        The input sequence for Prosit.

    Returns
    -------
    ptm_seq : str
        The string of digits used to represent any PTMs present.
    """
    if '(' not in mod_seq:
        return ''

    ptm_seq = '0.'
    while mod_seq:
        if mod_seq[0] == '(':
            end_mod = mod_seq.index(')') + 1
            mod_code = mod_seq[:end_mod]
            assert mod_code == '(+42.01)'
            ptm_seq = ptm_seq.replace('0.', '3.')
            mod_seq = mod_seq[end_mod:]
        if len(mod_seq) == 1 or mod_seq[1] != '(':
            ptm_seq += '0'
            mod_seq = mod_seq[1:]
        else:
            end_mod = mod_seq.index(')') + 1
            mod_code = mod_seq[1:end_mod]
            if mod_code == '(+42.01)':
                ptm_seq = '3.'
                mod_seq = mod_seq[0] + mod_seq[end_mod:]
                continue
            ptm_seq += var_mod_dict[mod_code]
            mod_seq = mod_seq[end_mod:]

    ptm_seq += '.0'

    return ptm_seq

def read_single_peaks_de_novo(df_loc):
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
    peaks_df = pl.read_csv(df_loc, infer_schema_length=10000, columns=PEAKS_DN_RELEVANT_COLUMNS)

    ptms_df = pd.DataFrame({
        PTM_NAME_KEY: pd.Series([
            'Carbamidomethylation', 'Oxidation (M)',
            'Acetylation (N-term)', 'Deamidation (NQ)',
            'Cysteinylation',
        ]),
        PTM_WEIGHT_KEY: pd.Series([
            KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
            KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            KNOWN_PTM_WEIGHTS['Acetylation (N-term)'],
            KNOWN_PTM_WEIGHTS['Deamidation (NQ)'],
            KNOWN_PTM_WEIGHTS['Cysteinylation'],
        ]),
        PTM_IS_VAR_KEY: pd.Series([True, True, True, True, True])
    })

    ptms_df[PTM_ID_KEY] = ptms_df[PTM_NAME_KEY].apply(
        lambda x : ID_NUMBERS.get(x, 9)
    )

    # Separate PTMs.
    var_mod_dict = {
        '(+57.02)': '1',
        '(+15.99)': '2',
        '(+42.01)': '3',
        '(+.98)': '4',
        '(+119.00)': '5',
    }

    peaks_df = peaks_df.with_columns(
        pl.col(PEAKS_DN_PEPTIDE_KEY).map_elements(
            lambda x: de_novo_ptm_convert(x, var_mod_dict)
        ).alias(PTM_SEQ_KEY),
        pl.col(PEAKS_DN_PEPTIDE_KEY).map_elements(
            lambda x : x.replace('(+15.99)', '').replace('(+57.02)', '').replace(
                '(+.98)', ''
            ).replace('(+42.01)', '').replace('(+119.00)', '')
        ).alias(PEPTIDE_KEY)
    )

    peaks_df = peaks_df.drop(PEAKS_DN_PEPTIDE_KEY)

    # Rename to match inSPIRE naming scheme.
    peaks_df = peaks_df.rename({
        PEAKS_DN_CHARGE_KEY: CHARGE_KEY,
        PEAKS_DN_RETENTION_TIME_KEY: RT_KEY,
    })

    # Filter for Prosit and add feature columns not present.
    peaks_df = peaks_df.with_columns(
        pl.lit(0).alias('fromChimera'),
        pl.col(PEPTIDE_KEY).map_elements(len).alias(SEQ_LEN_KEY),
    )

    peaks_df = filter_for_prosit(peaks_df)

    peaks_df = peaks_df.with_columns(
        (
            pl.col(PEAKS_DN_MZ_KEY)*pl.col(CHARGE_KEY) - (
                pl.col(PEAKS_DN_MASS_KEY) + (pl.col(CHARGE_KEY)*PROTON)
            )
        ).alias(
            MASS_DIFF_KEY
        ),
        (pl.col(PEAKS_DN_MASS_KEY)/pl.col(SEQ_LEN_KEY)).alias('avgResidueMass'),
        pl.col('local confidence (%)').map_elements(
            lambda x : min([int(y) for y in x.split(' ')]),
            return_dtype=pl.Int64,
        ).alias(DELTA_SCORE_KEY),
        pl.col('local confidence (%)').map_elements(
            lambda x : mean([int(y) for y in x.split(' ')]),
            return_dtype=pl.Float64,
        ).alias(ENGINE_SCORE_KEY),
        pl.lit(0).alias('missedCleavages'),
        pl.col(PEAKS_DN_SOURCE_KEY).map_elements(remove_source_suffixes).alias(SOURCE_KEY),
        pl.col(PEAKS_DN_SCAN_KEY).map_elements(
            lambda x : x if isinstance(x, int) else int(x.split(':')[-1])
        ).alias(SCAN_KEY),
    )

    peaks_df = peaks_df.with_columns(pl.lit('deNovo').alias(ACCESSION_KEY))

    peaks_df = peaks_df.with_columns(
        pl.col(ACCESSION_KEY).map_elements(
            lambda x : -1 if isinstance(x, str) and (
                '#DECOY#' in x or 'rev' in x
            ) else 1,
            pl.Int16,
        ).alias(LABEL_KEY)
    )

    peaks_df = peaks_df.drop(
        [
            PEAKS_DN_MASS_KEY,
            PEAKS_DN_MZ_KEY,
            PEAKS_DN_SCAN_KEY,
            PEAKS_DN_SOURCE_KEY,
            PEAKS_DN_LC_KEY,
            PEAKS_DN_SCORE_KEY,
            RT_KEY,
        ]
    )

    return peaks_df, ptms_df


def read_peaks_de_novo(df_locs, retrieve_position_level=False):
    """ Function to read PEAKS de novo results.
    """
    if isinstance(df_locs, list):
        search_dfs = []
        for df_loc in df_locs:
            individual_search_df, ptms_df = read_single_peaks_de_novo(df_loc)
            search_dfs.append(individual_search_df)
        search_df = [x.select(x[0].columns) for x in search_dfs]
        search_df = pl.concat(search_dfs)
    else:
        search_df, ptms_df = read_single_peaks_de_novo(df_locs)

    if retrieve_position_level:
        search_df = search_df.rename({
            'local confidence (%)': 'perPositionScores',
        })
        search_df = search_df.with_columns(
            pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).map_elements(
                lambda x : modify_sequence_for_skyline(x, mod_weights),
                skip_nulls=False, return_dtype=pl.String,
            ).alias('modifiedSequence'),
            pl.col('perPositionScores').str.replace_all(' ', ','),
        )
        return search_df.select([
            'source', 'scan', 'peptide', 'modifiedSequence', 'perPositionScores'
        ]), ptms_df


    search_df = search_df.drop([
        'Source File',
        'Scan',
        'local confidence (%)',
    ])
    return search_df, ptms_df
