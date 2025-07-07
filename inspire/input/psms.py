""" Function for reading inSPIRE PSMs in.
"""
import pandas as pd
import polars as pl

from inspire.constants import (
    KNOWN_PTM_WEIGHTS,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_WEIGHT_KEY,
)
from inspire.utils import reverse_skyline_mod_seq, filter_for_prosit

def read_single_psms(psms_loc):
    """ Function to read a single PSMs file.
    """
    psms_df = pl.read_csv(psms_loc)
    if 'Score' in psms_df.columns:
        psms_df = psms_df.rename({'Score': 'engineScore'})
    else:
        psms_df = psms_df.with_columns(
            pl.lit(-1).alias('engineScore'),
        )

    if 'Label' not in psms_df.columns:
        psms_df = psms_df.with_columns(
            pl.lit(1).alias('Label'),
        )

    if 'proteins' not in psms_df.columns:
        psms_df = psms_df.with_columns(
            pl.lit('unknown').alias('proteins'),
        )

    psms_df = psms_df.with_columns(
        pl.col('peptide').str.len_chars().alias('sequenceLength'),
        pl.lit(0).alias('missedCleavages'),
        pl.lit(0).alias('massDiff'),
        pl.lit(-1).alias('deltaScore'),
        pl.lit(0).alias('fromChimera'),
        pl.lit(0).alias('avgResidueMass'),
    )

    psms_df = filter_for_prosit(psms_df)
    psms_df = psms_df.with_columns(
        pl.col('modifiedSequence').map_elements(
            reverse_skyline_mod_seq, return_dtype=pl.String,
        ).alias('ptm_seq')
    )
    return psms_df

def read_psms(psms_df_loc):
    """ Function to fetch PSMs
    """
    if isinstance(psms_df_loc, str):
        psms_df = read_single_psms(psms_df_loc)
    else:
        psms_df = pl.concat(
            [read_single_psms(x) for x in psms_df_loc]
        )

    mods_df = pd.DataFrame({
        PTM_ID_KEY: [1,2,3,4,5,6,7,8],
        PTM_NAME_KEY: pd.Series([
            'Carbamidomethylation', 'Oxidation (M)',
            'Acetylation (N-term)', 'Deamidation (NQ)', 'Cysteinylation',
            'Carbamylation', 'NH3 loss', 'Carbamylation and NH3 loss',
        ]),
        PTM_WEIGHT_KEY: pd.Series([
            KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
            KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            KNOWN_PTM_WEIGHTS['Acetylation (N-term)'],
            KNOWN_PTM_WEIGHTS['Deamidation (NQ)'],
            KNOWN_PTM_WEIGHTS['Cysteinylation'],
            KNOWN_PTM_WEIGHTS['Carbamylation'],
            KNOWN_PTM_WEIGHTS['NH3 loss'],
            KNOWN_PTM_WEIGHTS['Carbamylation and NH3 loss'],
        ]),
        PTM_IS_VAR_KEY: pd.Series([True, True, True, True, True, True, True, True])
    })

    return psms_df, mods_df
