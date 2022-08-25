""" Functions for applying the prosit-delta predictor.
"""
from math import acos, pi
from copy import deepcopy
import numpy as np

from inspire.constants import (
    BLOSUM6_1_VALUES,
    INTENSITIES_KEY,
    ION_OFFSET,
    KNOWN_PTM_WEIGHTS,
    MZS_KEY,
    PEPTIDE_KEY,
    PROSIT_IONS_KEY,
    PTM_SEQ_KEY,
    RESIDUE_WEIGHTS,
    SPECTRAL_ANGLE_KEY,
)
from inspire.mz_match import match_mz

DELTA_PRO_FEATURE_SET = [
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
]

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
    true_l2_norm = np.linalg.norm(np.array(list(true.values())), ord=2)
    pred_l2_norm = np.linalg.norm(np.array(list(predicted.values())), ord=2)

    if true_l2_norm == 0 and pred_l2_norm == 0:
        return 0.0
    elif true_l2_norm == 0 and pred_l2_norm == 0:
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
    if not isinstance(ptm_seq, str):
        return ptm_seq

    ptm_seq = (
        ptm_seq[:flip_idx+1] + ptm_seq[flip_idx+2] +
        ptm_seq[flip_idx+1] + ptm_seq[flip_idx+3:]
    )
    return ptm_seq


def get_brute_force_deltas(df_row, matched_dict, ptm_id_weights, mz_accuracy, spectral_predictor):
    peptide = df_row[PEPTIDE_KEY]
    pep_len = len(peptide)
    deltas = []
    for i in range(1, 30):
        if not isinstance(df_row[f'flip{i}'], str) or df_row[f'flip{i}'] == peptide:
            continue

        flip_peptide = df_row[f'flip{i}'].replace('m', 'M')
        pred_ions = df_row[f'flip{i}Ions']
        if spectral_predictor == 'prosit':
            flip_ptm = convert_ptm_seq(df_row[PTM_SEQ_KEY], i)
        else:
            flip_ptm = df_row[f'flip{i}Ptms']
        rev_i = pep_len - i
        differing_ions = [
            f'b{i}', f'b{i}^2', f'b{i}^3', f'y{rev_i}', f'y{rev_i}^2', f'y{rev_i}^3',
        ]
        flip_matched = deepcopy(matched_dict)
        for diff_ion in differing_ions:
            flip_matched.pop(diff_ion, None)
        flip_pred_ions = [x for x in differing_ions if x in pred_ions]

        missing_preds = [x for x in pred_ions if x not in df_row['prositIons']]

        missing_finds = [x for x in df_row['prositIons'] if x not in pred_ions]
        for miss in missing_finds:
            flip_matched.pop(miss, None)

        all_recalc_ions = flip_pred_ions + missing_preds
        if all_recalc_ions:
            b_wt = compute_single_mz(flip_peptide, flip_ptm, 'b', i, ptm_id_weights)
            y_wt = compute_single_mz(flip_peptide, flip_ptm, 'y', rev_i, ptm_id_weights)
            for ion in flip_pred_ions:
                ion_type = ion[0]
                if ion_type == 'b':
                    base_mass = b_wt
                else:
                    base_mass = y_wt
                ion_data = ion.split('^')
                if len(ion_data) == 1:
                    frag_z = 1
                else:
                    frag_z = int(ion_data[-1])
                mz_match, match_idx = match_mz(base_mass, frag_z, df_row[MZS_KEY], loss=0.0)
                if abs(mz_match) < mz_accuracy:
                    flip_matched[ion] = df_row[INTENSITIES_KEY][match_idx]

        flip_sa = calculate_sa_from_dict(pred_ions, flip_matched)
        deltas.append(flip_sa - df_row[SPECTRAL_ANGLE_KEY])

    return calculate_delta_features(df_row, np.array(deltas))

def calculate_delta_features(df_row, prot_deltas):
    if prot_deltas.size != 0:
        df_row['nDeltasAboveThreshold'] = len(
            prot_deltas[prot_deltas > -0.1]
        )/len(prot_deltas)
        df_row['nDeltasAboveThresholdA'] = len(
            prot_deltas[prot_deltas > -0.05]
        )/len(prot_deltas)
        df_row['nDeltasAboveZero'] = len(
            prot_deltas[prot_deltas > 0.0]
        )/len(prot_deltas)
        delta_q1 = np.quantile(prot_deltas, 0.25)
        delta_q3 = np.quantile(prot_deltas, 0.75)
        df_row['prositDeltaMedian'] = np.quantile(prot_deltas, 0.5)
        df_row['prositDeltaQuartile1'] = delta_q1
        df_row['prositDeltaQuartile3'] = delta_q3
        df_row['minPrositDelta'] = np.min(prot_deltas)
        df_row['maxPrositDelta'] = np.max(prot_deltas)
    else:
        df_row['nDeltasAboveThreshold'] = -1.0
        df_row['nDeltasAboveThresholdA'] = -1.0
        df_row['nDeltasAboveZero'] = -1.0
        df_row['prositDeltaMedian'] = -1.0
        df_row['prositDeltaQuartile1'] = -1.0
        df_row['prositDeltaQuartile3'] = -1.0
        df_row['maxPrositDelta'] = -1.0
        df_row['minPrositDelta'] = -1.0

    return df_row

def compute_pd_single_mz(sequence, ion_type, ion_idx):
    if ion_type == 'y':
        sequence = sequence[::-1]

    tracking_mw = 0.0

    for idx in range(ion_idx):
        tracking_mw += DELTA_PRO_RESIDUE_WEIGHTS[sequence[idx]]

    return tracking_mw + ION_OFFSET[ion_type]


def get_new_inte(peptide, flip_ind, df_row, mz_accuracy, ion_series):
    if ion_series == 'b':
        base_mass = compute_pd_single_mz(peptide, 'b', flip_ind-1)
        base_mass += DELTA_PRO_RESIDUE_WEIGHTS[peptide[flip_ind]]
    else:
        base_mass = compute_pd_single_mz(peptide, 'y', len(peptide)-(flip_ind+1))
        base_mass += DELTA_PRO_RESIDUE_WEIGHTS[peptide[flip_ind-1]]
    match_inte = 0.0
    for frag_z in range(1, min(df_row['charge'], 4)):
        mz_match, match_idx = match_mz(base_mass, frag_z, df_row[MZS_KEY], loss=0.0)
        if abs(mz_match) < mz_accuracy:
            match_inte += df_row[INTENSITIES_KEY][match_idx]
    return match_inte

def get_err_at_loc(pep_len, matched_ions, prosit_ions, loc, letter):
    sum_err = 0
    if letter == 'y':
        loc = pep_len - loc
    for charge in ['', '^2', '^3']:
        code = letter + str(loc) + charge
        sum_err += abs(matched_ions.get(code, 0.0) - prosit_ions.get(code, 0.0))

    return sum_err

def get_deltas(df_row, prosit_ions, matched_intes, prosit_intes, reg_model, ox_flag, mz_accuracy, matched_dict):
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
            shape=(len(flip_inds), len(DELTA_PRO_FEATURE_SET))
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('spectralAngle')] = df_row[SPECTRAL_ANGLE_KEY]
        input_feats[:,DELTA_PRO_FEATURE_SET.index('blosumN')] = np.array([
            BLOSUM6_1_VALUES[peptide[idx-1]] for idx in flip_inds
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('blosumC')] = np.array([
            BLOSUM6_1_VALUES[peptide[idx]] for idx in flip_inds
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('charge')] = df_row['charge']
        input_feats[:,DELTA_PRO_FEATURE_SET.index('matchedCoverage')] = df_row['matchedCoverage']
        input_feats[:,DELTA_PRO_FEATURE_SET.index('collisionEnergy')] = df_row['collisionEnergy']
        input_feats[:,DELTA_PRO_FEATURE_SET.index('cNeighbourBlosum')] = np.array(
            [-5.0 if idx > pep_len-3 else BLOSUM6_1_VALUES[peptide[idx+1]] for idx in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('nNeighbourBlosum')] = np.array(
            [-5.0 if idx < 2 else BLOSUM6_1_VALUES[peptide[idx-2]] for idx in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('nTermDist')] = np.array(flip_inds)
        input_feats[:,DELTA_PRO_FEATURE_SET.index('cTermDist')] = np.array([
            [pep_len-idx for idx in flip_inds]
        ])
        input_feats[:,DELTA_PRO_FEATURE_SET.index('flipYNewIntensity')] = np.array(
            [get_new_inte(mod_seq, flip_ind, df_row, mz_accuracy, 'y') for flip_ind in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('flipBNewIntensity')] = np.array(
            [get_new_inte(mod_seq, flip_ind, df_row, mz_accuracy, 'b') for flip_ind in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('yErrsAtLoc')] = np.array(
            [get_err_at_loc(pep_len, normed_matched_dict, df_row[PROSIT_IONS_KEY], flip_ind, 'y') for flip_ind in flip_inds]
        )
        input_feats[:,DELTA_PRO_FEATURE_SET.index('bErrsAtLoc')] = np.array(
            [get_err_at_loc(pep_len, normed_matched_dict, df_row[PROSIT_IONS_KEY], flip_ind, 'b') for flip_ind in flip_inds]
        )
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
    else:
        prot_deltas = np.array([])

    return calculate_delta_features(df_row, prot_deltas)
