""" Read in casanovo dataframe.
"""
import re

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
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    RESIDUE_WEIGHTS,
    RT_KEY,
    SCAN_KEY,
    SEQ_LEN_KEY,
    SOURCE_KEY,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.mzml import process_mzml_file

from inspire.utils import filter_for_prosit, modify_sequence_for_skyline

PTM_SEQ_DICT = {
    '+57.021': '1',
    '+15.995': '2',
    '+42.011': '3',
    '+0.984': '4',
    '+43.006': '6',
    '-17.027': '7',
    '+43.006-17.027': '8',
}

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

def generate_ptm_seq(sequence):
    """ Function to generate a ptm sequence from casanovo sequence.
    """
    ptm_seq = '0.'
    while sequence:
        if sequence[0] not in '+-':
            ptm_seq += '0'
            sequence = sequence[1:]
        else:
            mod_end = re.search(r'[A-Z]', sequence, re.I)
            if mod_end is None:
                mod_end = len(sequence)
            else:
                mod_end = mod_end.span()[0]

            if len(ptm_seq) > 2:
                ptm_seq = ptm_seq[:-1] + PTM_SEQ_DICT[sequence[:mod_end]]
            else:
                ptm_seq = PTM_SEQ_DICT[sequence[:mod_end]] + '.'

            sequence = sequence[mod_end:]

    return ptm_seq + '.0'

def read_casanovo_dataset(casa_loc, scans_folder, scans_format, retrieve_position_level=False):
    scan_file = None
    with open(casa_loc, 'r', encoding='UTF-8') as casa_file:
        while line := casa_file.readline():
            if 'MTD	ms_run[1]-location' in line:
                scan_file = line.split('.')[-2].split('/')[-1]
    if scan_file is None:
        raise InputError(
            'Casanovo file does not contain a scan file name marked by MTD	ms_run[1]-location'
        )
    casa_df = pl.read_csv(
        casa_loc, separator='\t', skip_rows=60,
        columns=[
            'sequence',	'search_engine_score[1]', 'exp_mass_to_charge',
            'calc_mass_to_charge', 'opt_ms_run[1]_aa_scores', 'spectra_ref',
        ]
    )
    casa_df = casa_df.with_columns(
        pl.col('spectra_ref').map_elements(
            lambda x : int(x.split('index=')[-1]), return_dtype=pl.Int64,
        ).alias('index')
    )

    if scans_format in ('mzML', 'mzML_rt'):
        scans_format = 'mzML'
        scan_df = process_mzml_file(
            f'{scans_folder}/{scan_file}.{scans_format}',
            None,
            with_charge=False,
            with_retention_time=False,
        )
    else:
        scan_df = process_mgf_file(
            f'{scans_folder}/{scan_file}.{scans_format}',
            None, None, None,
            with_charge=True,
            with_retention_time=True,
            with_ms1=True,
        )
    scan_df = scan_df.with_row_index('index')
    scan_df = scan_df.with_columns(pl.col('index').cast(pl.Int64))
    casa_df = casa_df.with_columns(pl.col('index').cast(pl.Int64))

    casa_df = casa_df.join(
        scan_df.select([
            'index', SOURCE_KEY, SCAN_KEY, CHARGE_KEY, RT_KEY, 'ms1Intensity',
        ]),
        how='inner', on='index'
    )

    regex = re.compile('[^a-zA-Z]')
    casa_df = casa_df.with_columns(
        pl.col('sequence').map_elements(
            lambda x : regex.sub('', x), return_dtype=pl.String
        ).str.replace_all('I', 'L').alias(PEPTIDE_KEY),
        pl.lit(1).alias(LABEL_KEY),
        pl.lit('deNovo').alias(ACCESSION_KEY),
        pl.lit(0).alias('missedCleavages'),
        (pl.col('exp_mass_to_charge') - pl.col('calc_mass_to_charge')).alias(MASS_DIFF_KEY),
        pl.lit(0).alias('fromChimera'),
        pl.col('opt_ms_run[1]_aa_scores').map_elements(
            lambda x : min([float(y) for y in x.split(',')]),
            return_dtype=pl.Float64,
        ).alias(DELTA_SCORE_KEY),
    )
    casa_df = casa_df.with_columns(
        pl.col(PEPTIDE_KEY).str.len_chars().alias(SEQ_LEN_KEY),
        pl.col('sequence').map_elements(
            generate_ptm_seq, return_dtype=pl.String,
        ).alias('ptm_seq'),
    )

    if retrieve_position_level:
        return casa_df.rename({
            'opt_ms_run[1]_aa_scores': 'perPositionScores',
        })
    casa_df = casa_df.with_columns(
        pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).map_elements(
            lambda x : sum(
                [RESIDUE_WEIGHTS[a_a] for a_a in x[PEPTIDE_KEY]]
            ) + sum(
                [KNOWN_PTM_WEIGHTS.get(ptm, 0.0) for ptm in x[PTM_SEQ_KEY]]
            ), return_dtype=pl.Float64,
        ).alias('avgResidueMass')
    )
    casa_df = casa_df.rename({'search_engine_score[1]': ENGINE_SCORE_KEY})
    casa_df = filter_for_prosit(casa_df)

    return casa_df.select([
        SOURCE_KEY,
        SCAN_KEY,
        PEPTIDE_KEY,
        LABEL_KEY,
        ACCESSION_KEY,
        'ptm_seq',
        SEQ_LEN_KEY,
        'missedCleavages',
        CHARGE_KEY,
        MASS_DIFF_KEY,
        RT_KEY,
        ENGINE_SCORE_KEY,
        'deltaScore',
        'ms1Intensity',
        'fromChimera',
        'avgResidueMass',
    ])

def read_casanovo(search_results, scans_folder, scans_format, retrieve_position_level=False):
    """ Function to read Casanovo outputs.
    """
    if isinstance(search_results, list):
        casa_df = pl.concat(
            read_casanovo_dataset(
                casa_loc, scans_folder, scans_format, retrieve_position_level
            ) for casa_loc in search_results
        )
    else:
        casa_df = read_casanovo_dataset(
            search_results, scans_folder, scans_format, retrieve_position_level
        )
    ptms_df = pd.DataFrame({
        PTM_ID_KEY: [1,2,3,4,6,7,8],
        PTM_NAME_KEY: pd.Series([
            'Carbamidomethylation', 'Oxidation (M)',
            'Acetylation (N-term)', 'Deamidation (NQ)',
            'Carbamylation', 'NH3 loss', 'Carbamylation and NH3 loss',
        ]),
        PTM_WEIGHT_KEY: pd.Series([
            KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
            KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            KNOWN_PTM_WEIGHTS['Acetylation (N-term)'],
            KNOWN_PTM_WEIGHTS['Deamidation (NQ)'],
            KNOWN_PTM_WEIGHTS['Carbamylation'],
            KNOWN_PTM_WEIGHTS['NH3 loss'],
            KNOWN_PTM_WEIGHTS['Carbamylation and NH3 loss'],
        ]),
        PTM_IS_VAR_KEY: pd.Series([True, True, True, True, True, True, True])
    })
    if retrieve_position_level:
        mod_weights = dict(zip(ptms_df[PTM_ID_KEY].tolist(), ptms_df[PTM_WEIGHT_KEY].tolist()))
        casa_df = casa_df.with_columns(
            pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).map_elements(
                lambda x : modify_sequence_for_skyline(x, mod_weights),
                skip_nulls=False, return_dtype=pl.String,
            ).alias('modifiedSequence')
        )
        return casa_df.select([
            'source', 'scan', 'peptide', 'modifiedSequence', 'perPositionScores',
        ]), ptms_df
    return casa_df, ptms_df
