""" Functions for applying the prosit-delta predictor.
"""
from copy import deepcopy
import numpy as np

from inspire.constants import (
    BLOSUM6_1_VALUES,
    KNOWN_PTM_WEIGHTS,
    PEPTIDE_KEY,
    PTM_SEQ_KEY,
    RESIDUE_WEIGHTS,
    SPECTRAL_ANGLE_KEY,
)

DELTA_PRO_FEATURE_SET = [
    'spectralAngle',
    'blosumC',
    'blosumN',
    'massDiff',
    'cNeighbourBlosum',
    'nNeighbourBlosum',
    'relPos',
    'bPrositIntesAtC',
    'yPrositIntesAtN',
    'yPrositIntesAtLoc',
    'bPrositIntesAtLoc',
    'yMatchedIntesAtN',
    'bMatchedIntesAtC',
    'yMatchedIntesAtLoc',
    'bMatchedIntesAtLoc',
    'cNeighbourOxidation',
    'nNeighbourOxidation',
    'cOxidation',
    'nOxidation',
]

DELTA_PRO_RESIDUE_WEIGHTS = deepcopy(RESIDUE_WEIGHTS)
DELTA_PRO_RESIDUE_WEIGHTS['C'] = 103.009185 + 57.021464

SIGNIFICANCE_THRESHOLD = -0.05

def check_oxidation(pep_len, ptm_seq, idx, ox_flag):
    """ Helper function to check if resiude is oxidised.
    """
    if not isinstance(ptm_seq, str) or ox_flag not in ptm_seq or idx < 0 or idx > pep_len:
        return 0
    if ptm_seq[idx+2] == ox_flag:
        return 1
    return 0

def get_mass_diff(peptide, ptm_seq, idx, ox_flag):
    """ Function to find the difference between two peptide masses.
    """
    if not isinstance(ptm_seq, str):
        return abs(
            DELTA_PRO_RESIDUE_WEIGHTS[peptide[idx]] - DELTA_PRO_RESIDUE_WEIGHTS[peptide[idx-1]]
        )

    c_wt = DELTA_PRO_RESIDUE_WEIGHTS[peptide[idx]]
    n_wt = DELTA_PRO_RESIDUE_WEIGHTS[peptide[idx-1]]
    if ptm_seq[idx+2] == ox_flag:
        c_wt += KNOWN_PTM_WEIGHTS['Oxidation (M)']
    if ptm_seq[idx+1] == ox_flag:
        n_wt += KNOWN_PTM_WEIGHTS['Oxidation (M)']
    return abs(c_wt - n_wt)

def get_intes_at_loc(pep_len, matched_intensities, prosit_ions, loc, letter):
    """ Function to calculate the sum of all intensities at a given location.

    Parameters
    ----------
    pep_len : int
        The length of the peptide.
    prosit_ions : dict
        The prosit predicted spectrum.
    loc : int
        The location at which to get intensities.
    letter : str
        The type of ion considered.

    Returns
    -------
    sum_inte : float
        The sum of intensities at the location.
    """
    sum_inte = 0.0
    if loc in (0, pep_len):
        return sum_inte

    if letter == 'y':
        loc = pep_len - loc

    for charge in ['', '^2', '^3']:
        for loss in ['', '-NH3', '-H2O']:
            full_code = letter + str(loc) + charge + loss
            inte_val = matched_intensities[prosit_ions == full_code]
            if inte_val.size:
                sum_inte += inte_val[0]
    return sum_inte

def get_deltas(df_row, prosit_ions, matched_intes, prosit_intes, reg_model, ox_flag):
    """ Function to get all predicted prosit-deltas.

    Parameters
    ----------
    df_row : pd.Series
        A row of a DataFrame containing spectral data.
    prosit_ions : dict
        The prosit predicted spectrum.
    reg_model : sklearn.ensemble.RandomForestRegressor
        The trained prosit-delta model.

    Returns
    -------
    df_row : pd.Series
        The updated row containing prosit-delta features.
    """
    peptide = df_row[PEPTIDE_KEY]

    pep_len = len(peptide)
    flip_inds = np.array([
        idx+1 for idx in range(
            len(df_row[PEPTIDE_KEY])-1
        ) if df_row[PEPTIDE_KEY][idx] != df_row[PEPTIDE_KEY][idx+1]
    ])

    if len(flip_inds):
        input_feats = np.zeros(
            shape=(len(flip_inds), len(DELTA_PRO_FEATURE_SET))
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('spectralAngle')] = df_row[SPECTRAL_ANGLE_KEY]
        input_feats[:,DELTA_PRO_FEATURE_SET.index('blosumN')] = np.array([
            BLOSUM6_1_VALUES[peptide[idx-1]] for idx in flip_inds
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('blosumC')] = np.array([
            BLOSUM6_1_VALUES[peptide[idx]] for idx in flip_inds
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('massDiff')] = np.array([
            get_mass_diff(peptide, df_row[PTM_SEQ_KEY], idx, ox_flag) for idx in flip_inds
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('cNeighbourBlosum')] = np.array(
            [0.0 if idx > pep_len-3 else BLOSUM6_1_VALUES[peptide[idx+2]] for idx in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('nNeighbourBlosum')] = np.array(
            [0.0 if idx < 2 else BLOSUM6_1_VALUES[peptide[idx-2]] for idx in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('relPos')] = np.array([
            [idx/pep_len for idx in flip_inds]
        ])

        input_feats[:, DELTA_PRO_FEATURE_SET.index('bPrositIntesAtC')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    prosit_intes,
                    prosit_ions,
                    flip_idx+1,
                    'b',
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('yPrositIntesAtN')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    prosit_intes,
                    prosit_ions,
                    flip_idx-1,
                    'y',
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('yPrositIntesAtLoc')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    prosit_intes,
                    prosit_ions,
                    flip_idx,
                    'y',
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('bPrositIntesAtLoc')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    prosit_intes,
                    prosit_ions,
                    flip_idx,
                    'b',
                ) for flip_idx in flip_inds
            ]
        )

        input_feats[:, DELTA_PRO_FEATURE_SET.index('yMatchedIntesAtN')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    matched_intes,
                    prosit_ions,
                    flip_idx-1,
                    'y',
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('bMatchedIntesAtC')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    matched_intes,
                    prosit_ions,
                    flip_idx+1,
                    'b'
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('yMatchedIntesAtLoc')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    matched_intes,
                    prosit_ions,
                    flip_idx,
                    'y'
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('bMatchedIntesAtLoc')] = np.array(
            [
                get_intes_at_loc(
                    pep_len,
                    matched_intes,
                    prosit_ions,
                    flip_idx,
                    'b'
                ) for flip_idx in flip_inds
            ]
        )

        input_feats[:, DELTA_PRO_FEATURE_SET.index('cNeighbourOxidation')] = np.array(
            [
                check_oxidation(
                    pep_len,
                    df_row[PTM_SEQ_KEY],
                    flip_idx+1,
                    ox_flag,
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('nNeighbourOxidation')] = np.array(
            [
                check_oxidation(
                    pep_len,
                    df_row[PTM_SEQ_KEY],
                    flip_idx-2,
                    ox_flag,
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('cOxidation')] = np.array(
            [
                check_oxidation(
                    pep_len,
                    df_row[PTM_SEQ_KEY],
                    flip_idx,
                    ox_flag,
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, DELTA_PRO_FEATURE_SET.index('nOxidation')] = np.array(
            [
                check_oxidation(
                    pep_len,
                    df_row[PTM_SEQ_KEY],
                    flip_idx-1,
                    ox_flag,
                ) for flip_idx in flip_inds
            ]
        )

        prot_deltas = reg_model.predict(input_feats)

        df_row['nDeltasAboveThreshold'] = len(
            prot_deltas[prot_deltas > -0.1]
        )/len(prot_deltas)
        df_row['nDeltasAboveZero'] = len(
            prot_deltas[prot_deltas > 0.0]
        )/len(prot_deltas)
        delta_q1 = np.quantile(prot_deltas, 0.25)
        delta_q3 = np.quantile(prot_deltas, 0.75)
        df_row['prositDeltaMedian'] = np.quantile(prot_deltas, 0.5)
        df_row['prositDeltaQuartile1'] = delta_q1
        df_row['prositDeltaQuartile3'] = delta_q3
        df_row['prositDeltaIQR'] = delta_q3 - delta_q1
    else:
        df_row['nDeltasAboveThreshold'] = -1.0
        df_row['nDeltasAboveZero'] = -1.0
        df_row['prositDeltaMedian'] = -1.0

        df_row['prositDeltaQuartile1'] = -1.0
        df_row['prositDeltaQuartile3'] = -1.0
        df_row['prositDeltaIQR'] = -1.0

    return df_row
