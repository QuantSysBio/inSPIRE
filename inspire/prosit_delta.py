""" Functions for applying the prosit-delta predictor.
"""
from math import acos, pi
from copy import deepcopy
import numpy as np

from inspire.constants import (
    BLOSUM6_1_VALUES,
    CHARGE_KEY,
    ION_OFFSET,
    KNOWN_PTM_WEIGHTS,
    PEPTIDE_KEY,
    PTM_SEQ_KEY,
    RESIDUE_WEIGHTS,
    SPECTRAL_ANGLE_KEY,
)
from inspire.mz_match import match_mz

DELTA_PRO_FEATURE_SET = (
    'spectralAngle',
    'blosumC',
    'blosumN',
    'nTermDist',
    'cTermDist',
    'charge',
    'cNeighbourBlosum',
    'nNeighbourBlosum',
    'yErrsAtLoc',
    'bErrsAtLoc',
    'collisionEnergy',
    'bPrositIntesAtC',
    'yPrositIntesAtN',
    'yPrositIntesAtLoc',
    'bPrositIntesAtLoc',
    'yMatchedIntesAtN',
    'bMatchedIntesAtC',
    'yMatchedIntesAtLoc',
    'bMatchedIntesAtLoc',
    'cOxidation',
    'nOxidation',
    'flipYNewIntensity',
    'flipBNewIntensity',
    'matchedCoverage',
)

SPECTRAL_ANGLE_INDEX = DELTA_PRO_FEATURE_SET.index('spectralAngle')
BLOSUM_N_INDEX = DELTA_PRO_FEATURE_SET.index('blosumN')
BLOSUM_C_INDEX = DELTA_PRO_FEATURE_SET.index('blosumC')
CHARGE_INDEX = DELTA_PRO_FEATURE_SET.index('charge')
MATCHED_COV_INDEX = DELTA_PRO_FEATURE_SET.index('matchedCoverage')
CE_INDEX = DELTA_PRO_FEATURE_SET.index('collisionEnergy')
C_NEIGHB_INDEX = DELTA_PRO_FEATURE_SET.index('cNeighbourBlosum')
N_NEIGHB_INDEX = DELTA_PRO_FEATURE_SET.index('nNeighbourBlosum')
N_TERM_INDEX = DELTA_PRO_FEATURE_SET.index('nTermDist')
C_TERM_INDEX = DELTA_PRO_FEATURE_SET.index('cTermDist')
Y_NEW_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('flipYNewIntensity')
B_NEW_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('flipBNewIntensity')
Y_ERR_INDEX = DELTA_PRO_FEATURE_SET.index('yErrsAtLoc')
B_ERR_INDEX = DELTA_PRO_FEATURE_SET.index('bErrsAtLoc')
B_PROSIT_C_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('bPrositIntesAtC')
Y_PROSIT_N_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('yPrositIntesAtN')
Y_PROSIT_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('yPrositIntesAtLoc')
B_PROSIT_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('bPrositIntesAtLoc')
Y_MATCHED_N_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('yMatchedIntesAtN')
B_MATCHED_C_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('bMatchedIntesAtC')
Y_MATCHED_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('yMatchedIntesAtLoc')
B_MATCHED_INTE_INDEX = DELTA_PRO_FEATURE_SET.index('bMatchedIntesAtLoc')
N_OX_INDEX = DELTA_PRO_FEATURE_SET.index('nOxidation')
C_OX_INDEX = DELTA_PRO_FEATURE_SET.index('cOxidation')

DELTA_PRO_RESIDUE_WEIGHTS = deepcopy(RESIDUE_WEIGHTS)
DELTA_PRO_RESIDUE_WEIGHTS['C'] = 103.009185 + 57.021464
DELTA_PRO_RESIDUE_WEIGHTS['m'] = DELTA_PRO_RESIDUE_WEIGHTS['M'] + KNOWN_PTM_WEIGHTS['Oxidation (M)']
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
    """ Function to find the difference between two residue masses.
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

def compute_single_mz(sequence, modifications, ion_type, ion_idx, ptm_id_weights):
    """ Function to compute the molecular weights of potential fragments
        generated from a peptide (y & b ions, charges 1,2, or 3, and H2O
        or O2 losses).

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.
    reverse : bool
        Whether we are getting fragment mzs in the forward direction
        (eg for b ions), or backward direction (eg. for y ions).
    ptm_id_weights : dict
        Mapping of ptm ids to their molecular weights.

    Returns
    -------
    mzs : np.array of floats
        An array of all the possible mzs that coule be observed in
        the MS2 spectrum of a sequence.
    """
    if (
        modifications and
        isinstance(modifications, str) and
        modifications != 'nan'
        and modifications != 'None'
    ):
        ptms_list = modifications.split(".")
        mods_list = [int(mod) for mod in ptms_list[1]]

        if ion_type == 'y':
            ptm_start = int(ptms_list[2])
            mods_list = mods_list[::-1]
        else:
            ptm_start = int(ptms_list[0])
    else:
        mods_list = None
        ptm_start = 0.0

    if ion_type == 'y':
        sequence = sequence[::-1]

    tracking_mw = ptm_id_weights[ptm_start]

    for idx in range(ion_idx):
        tracking_mw += RESIDUE_WEIGHTS[sequence[idx]]
        if mods_list is not None and mods_list[idx]:
            tracking_mw += ptm_id_weights[mods_list[idx]]

    return tracking_mw + ION_OFFSET[ion_type]

def calculate_sa_from_dict(predicted, true):
    """ Function to calculate the spectral angle when the predicted and true spectrum
        are stored as dictionaries.

    Parameters
    ----------
    predicted : dict
        A dictionary of the prosit predicted spectrum.
    true : dict
        A dictionary of the experimental ions matched to the predicted spectrum.

    Returns
    -------
    spectral_angle : float
        The spectral angle between spectra.
    """
    true_l2_norm = np.linalg.norm(np.array(list(true.values())), ord=2)
    pred_l2_norm = np.linalg.norm(np.array(list(predicted.values())), ord=2)

    if true_l2_norm == 0 and pred_l2_norm == 0:
        return 0.0
    if true_l2_norm == 0 and pred_l2_norm == 0:
        return 1.0

    if pred_l2_norm > 0.0:
        for ion in predicted:
            predicted[ion] /= pred_l2_norm
    if true_l2_norm > 0.0:
        for ion in true:
            true[ion] /= true_l2_norm

    product = 0.0
    for ion in predicted:
        product += (predicted[ion] * true.get(ion, 0.0))

    product = max(min(product, 1.0), 0.0)
    spectral_distance = 2*acos(product)/pi
    return 1.0 - spectral_distance

def convert_ptm_seq(ptm_seq, flip_idx):
    """ Function to adjust a PTM sequence to match permutation of the sequence.

    Parameters
    ----------
    ptm_seq : str or None
        The the original PTM sequence (stored as string of digits)
    flip_idx : int
        The index which is to be permuted.

    Returns
    -------
    ptm_seq : str or None
        The adjusted PTM sequence.
    """
    if not isinstance(ptm_seq, str):
        return ptm_seq

    ptm_seq = (
        ptm_seq[:flip_idx+1] + ptm_seq[flip_idx+2] +
        ptm_seq[flip_idx+1] + ptm_seq[flip_idx+3:]
    )
    return ptm_seq

def calculate_delta_features(results, prot_deltas):
    """ Function to calculate the Prosit delta features.

    Parameters
    ----------
    df_row : pd.Series
        A single row of the PSM dataframe.
    prot_deltas : np.array
        The Prosit-delta values at each point in the sequence.

    Returns
    -------
    df_row : pd.Series
        The input row with Prosit-delta features added.
    """
    if prot_deltas.size != 0:
        results['nDeltasAboveThreshold'] = len(
            prot_deltas[prot_deltas > -0.1]
        )/len(prot_deltas)
        results['nDeltasAboveThresholdA'] = len(
            prot_deltas[prot_deltas > -0.05]
        )/len(prot_deltas)
        results['nDeltasAboveZero'] = len(
            prot_deltas[prot_deltas > 0.0]
        )/len(prot_deltas)
        delta_q1 = np.quantile(prot_deltas, 0.25)
        delta_q3 = np.quantile(prot_deltas, 0.75)
        results['prositDeltaMedian'] = float(np.quantile(prot_deltas, 0.5))
        results['prositDeltaQuartile1'] = float(delta_q1)
        results['prositDeltaQuartile3'] = float(delta_q3)
        results['minPrositDelta'] = float(np.min(prot_deltas))
        results['maxPrositDelta'] = float(np.max(prot_deltas))
    else:
        results['nDeltasAboveThreshold'] = -1.0
        results['nDeltasAboveThresholdA'] = -1.0
        results['nDeltasAboveZero'] = -1.0
        results['prositDeltaMedian'] = -1.0
        results['prositDeltaQuartile1'] = -1.0
        results['prositDeltaQuartile3'] = -1.0
        results['maxPrositDelta'] = -1.0
        results['minPrositDelta'] = -1.0

    return results

def compute_pd_single_mw(sequence, ion_type, ion_idx):
    """ Function to compute the mw of a single ion.

    Parameters
    ----------
    sequence : str
        The peptide sequence.
    ion_type : str
        y or b.
    ion_idx : int
        The index of ion.

    Returns
    -------
    mw : float
        The mw of the theoretical ion.
    """
    if ion_type == 'y':
        sequence = sequence[::-1]

    tracking_mw = 0.0

    for idx in range(ion_idx):
        tracking_mw += DELTA_PRO_RESIDUE_WEIGHTS[sequence[idx]]

    return tracking_mw + ION_OFFSET[ion_type]


def get_new_inte(peptide, flip_ind, charge, mzs, intes, mz_accuracy, ion_series):
    """ Function to get the potential new intensity.
    """
    if ion_series == 'b':
        base_mass = compute_pd_single_mw(peptide, 'b', flip_ind-1)
        base_mass += DELTA_PRO_RESIDUE_WEIGHTS[peptide[flip_ind]]
    else:
        base_mass = compute_pd_single_mw(peptide, 'y', len(peptide)-(flip_ind+1))
        base_mass += DELTA_PRO_RESIDUE_WEIGHTS[peptide[flip_ind-1]]
    match_inte = 0.0
    for frag_z in range(1, min(charge, 4)):
        mz_match, match_idx = match_mz(base_mass, frag_z, mzs, loss=0.0)
        if abs(mz_match) < mz_accuracy:
            match_inte += intes[match_idx]
    return match_inte

def get_err_at_loc(pep_len, matched_ions, prosit_ions, loc, letter):
    """ Function to get the error at a point in a spectrum between Prosit predicted and
        normalised experimental intensitiy.

    Parameters
    ----------
    pep_len : int
        Length of the peptide.
    matched_ions : dict
        Dictionary of the experimental intensities matched to Prosit ions.
    prosit_ions : dict
        Dictionary of the prosit predicted intensities.
    loc : int
        The location of interest.
    letter : str
        y or b

    Returns
    -------
    sum_error : float
        The sum of all errors at that location.
    """
    sum_err = 0
    if letter == 'y':
        loc = pep_len - loc
    for charge in ['', '^2', '^3']:
        code = letter + str(loc) + charge
        sum_err += abs(matched_ions.get(code, 0.0) - prosit_ions.get(code, 0.0))

    return sum_err

def get_deltas(
        results,
        df_row,
        mzs,
        intes,
        prosit_preds,
        prosit_ions,
        matched_intes,
        prosit_intes,
        reg_model,
        ox_flag,
        mz_accuracy,
        matched_dict,
    ):
    """ Function to get all predicted prosit-deltas.

    Parameters
    ----------
    df_row : pd.Series
        A row of a DataFrame containing spectral data.
    prosit_ions : dict
        The prosit predicted spectrum.
    reg_model : xgb.XGBRegressor
        The trained prosit-delta model.

    Returns
    -------
    df_row : pd.Series
        The updated row containing prosit-delta features.
    """
    peptide = df_row[PEPTIDE_KEY]
    charge = df_row[CHARGE_KEY]
    mod_seq = df_row['modified_sequence'].replace('M(ox)', 'm')

    true_l2_norm = np.linalg.norm(np.array(list(matched_dict.values())), ord=2)
    normed_matched_dict = {k:v/true_l2_norm for k,v in matched_dict.items()}
    pep_len = len(peptide)
    flip_inds = np.array([
        idx+1 for idx in range(
            len(df_row[PEPTIDE_KEY])-1
        ) if df_row[PEPTIDE_KEY][idx] != df_row[PEPTIDE_KEY][idx+1]
    ])

    if len(flip_inds):
        input_feats = np.zeros(
            shape=(len(flip_inds), len(DELTA_PRO_FEATURE_SET)), dtype=np.float64
        )
        input_feats[:, SPECTRAL_ANGLE_INDEX] = results[SPECTRAL_ANGLE_KEY]
        input_feats[:, BLOSUM_N_INDEX] = np.array([
            BLOSUM6_1_VALUES[peptide[idx-1]] for idx in flip_inds
        ])
        input_feats[:, BLOSUM_C_INDEX] = np.array([
            BLOSUM6_1_VALUES[peptide[idx]] for idx in flip_inds
        ])
        input_feats[:, CHARGE_INDEX] = df_row['charge']
        input_feats[:, MATCHED_COV_INDEX] = results['matchedCoverage']
        input_feats[:, CE_INDEX] = df_row['collisionEnergy']
        input_feats[:, C_NEIGHB_INDEX] = np.array(
            [-5.0 if idx > pep_len-3 else BLOSUM6_1_VALUES[peptide[idx+1]] for idx in flip_inds]
        )
        input_feats[:, N_NEIGHB_INDEX] = np.array(
            [-5.0 if idx < 2 else BLOSUM6_1_VALUES[peptide[idx-2]] for idx in flip_inds]
        )
        input_feats[:, N_TERM_INDEX] = np.array(flip_inds)
        input_feats[:, C_TERM_INDEX] = np.array([
            [pep_len-idx for idx in flip_inds]
        ])

        input_feats[:, Y_NEW_INTE_INDEX] = np.array(
            [
                get_new_inte(
                    mod_seq, flip_ind, charge, mzs, intes, mz_accuracy, 'y'
                ) for flip_ind in flip_inds
            ]
        )

        input_feats[:, B_NEW_INTE_INDEX] = np.array(
            [
                get_new_inte(
                    mod_seq, flip_ind, charge, mzs, intes, mz_accuracy, 'b'
                ) for flip_ind in flip_inds
            ]
        )
        input_feats[:, Y_ERR_INDEX] = np.array(
            [
                get_err_at_loc(
                    pep_len, normed_matched_dict, prosit_preds, flip_ind, 'y'
                ) for flip_ind in flip_inds
            ]
        )

        input_feats[:, B_ERR_INDEX] = np.array(
            [
                get_err_at_loc(
                    pep_len, normed_matched_dict, prosit_preds, flip_ind, 'b'
                ) for flip_ind in flip_inds
            ]
        )
        input_feats[:, B_PROSIT_C_INTE_INDEX] = np.array(
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
        input_feats[:, Y_PROSIT_N_INTE_INDEX] = np.array(
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

        input_feats[:, Y_PROSIT_INTE_INDEX] = np.array(
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
        input_feats[:, B_PROSIT_INTE_INDEX] = np.array(
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

        input_feats[:, Y_MATCHED_N_INTE_INDEX] = np.array(
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
        input_feats[:, B_MATCHED_C_INTE_INDEX] = np.array(
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
        input_feats[:, Y_MATCHED_INTE_INDEX] = np.array(
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
        input_feats[:, B_MATCHED_INTE_INDEX] = np.array(
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

        input_feats[:, C_OX_INDEX] = np.array(
            [
                check_oxidation(
                    pep_len,
                    df_row[PTM_SEQ_KEY],
                    flip_idx,
                    ox_flag,
                ) for flip_idx in flip_inds
            ]
        )
        input_feats[:, N_OX_INDEX] = np.array(
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
    else:
        prot_deltas = np.array([])

    return calculate_delta_features(results, prot_deltas)
