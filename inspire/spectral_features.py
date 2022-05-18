""" Functions for calculating spectral features.
"""

# from copyreg import pickle
from math import acos, pi
from statistics import median, variance

import pickle
import numpy as np
import pkg_resources
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import median_absolute_error

from inspire.constants import (
    CHARGE_KEY,
    INTENSITIES_KEY,
    LOSS_IONS_KEY,
    MAE_KEY,
    NEUTRAL_LOSSES,
    OBS_NOT_PRED_KEY,
    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,
    MATCHED_IONS_KEY,
    NOT_ASSIGNED_KEY,
    PRECURSOR_INTE_KEY,
    MZS_KEY,
    PEARSON_KEY,
    PEPTIDE_KEY,
    PROSIT_IONS_KEY,
    PROTON,
    PTM_ID_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    SPEARMAN_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.mz_match import get_ion_masses
from inspire.prosit_delta import get_deltas
from inspire.utils import get_ox_flag

SPECTRAL_FEATURES = [

    SPECTRAL_ANGLE_KEY,

    SPEARMAN_KEY,
    PEARSON_KEY,
    MAE_KEY,

    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,

    OBS_NOT_PRED_KEY,
    NOT_ASSIGNED_KEY,
    LOSS_IONS_KEY,
    PRECURSOR_INTE_KEY,

    'yIsDominantIonSeries',
    'bIsDominantIonSeries',

    'maxTypeSpectralAngle',
    'minTypeSpectralAngle',

    'maxTypeMajorMatchedFrags',
    'typePredRatio',
    'typeMatchedRatio',
    'typeIntensityRatio',

    'foundNotPredDivPred',
    'foundNotPredDivNotPred',
    'matchedIntensityRatio',

    'spectralAngleKR',
    'fracMatchedKR',
    'possibleKrFragsDivTotal',

    'fracPredNotFound',
    'fracNotMatchable',
    'matchedDivPred',
    'matchedDivObserved',

    'matchedCoverage',
    'majorMatchedCoverage',
    'minMatchedCoverage',
    'maxMatchedCoverage',
    'predCoverage',
    'predNotFoundCoverage',

    'nMajorMatchedDivFrags',
    'nMajorPredNotFoundDivFrags',
    'nMajorNotMatchableDivFrags',
    'spearmanMajorIons',

    'observedCoverage',
    'nMajorFoundNotPredDivFrags',
    'nMinorFoundNotPredDivFrags',

    'nMinorMatchedDivFrags',
    'nMinorPredNotFoundDivFrags',
    'nMinorNotMatchableDivFrags',
    'spearmanMinorIons',

    'maxTypeFoundNotPred',

    'nNotPredNotFoundDivFrags',
    'foundNotPredRatio',
    'maxTypePredNotFound',
    'minTypeFoundNotPred',

    'nRawPeaksDivFrags',
    'spectrumDensity',
]

DELTA_FEATURES = [
    'prositDeltaQuartile1',
    'prositDeltaQuartile3',
    'prositDeltaIQR',
    'prositDeltaMedian',
    'nDeltasAboveThreshold',
    'nDeltasAboveZero',
]

PROSIT_MAJOR_MINOR_CUT_OFF = 0.1
SPECTRUM_MAJOR_MINOR_CUT_OFF = 0.1

def calculate_spectral_angle(true, predicted):
    """ Function to calculate the spectral angle between the true and predicted
        spectra.

    Parameters
    ----------
    true : dict
        A dictionary of ion intensities from the true spectrum.
    predicted : dict
        A dictionary of ion intensities from the predicted spectrum.

    Returns
    -------
    spectral_angle : float
        The spectral angle between the two spectra.
    """
    true_l2_norm = np.linalg.norm(true, ord=2)
    pred_l2_norm = np.linalg.norm(predicted, ord=2)
    if true_l2_norm == 0 or pred_l2_norm == 0:
        return 0.0

    product = np.dot(true/true_l2_norm, predicted/pred_l2_norm)

    spectral_distance = 2*acos(product)/pi

    return 1.0 - spectral_distance

def check_for_precursor_peak(observed_mzs, obs_intes, precursor_mz, mz_accuracy, assigned_inds):
    """ Function to check for the presence of the precursor fragment in spectrum.

    Parameters
    ----------
    observed_mzs : np.array of float
        An array of mz values of fragments observed.
    precursor_mz : float
        The m/z of the precursor.
    mz_accuracy : float
        The m/z accuracy of the mass spectrometer.
    assigned_inds : list
        The indices already matched to an ion.

    Returns
    -------
    precursor_present : int
        Flag indicating if the precursor m/z is present.
    assigned_inds : list
        The input indices updated with the index of precursor matched ion.
    """
    matched_mz_ind = np.argmin(
        np.abs(observed_mzs - precursor_mz)
    )
    if abs(observed_mzs[matched_mz_ind] - precursor_mz) < mz_accuracy:
        assigned_inds.append(matched_mz_ind)
        return obs_intes[matched_mz_ind], assigned_inds
    return 0, assigned_inds

def match_mz(base_mass, frag_z, observed_mzs, loss=0.0):
    """ Function to match a fragment m/z to the nearest experimental m/z.

    Parameters
    ----------
    base_mass : float
        The mass of the b or y ion of that fragment index.
    frag_z : int
        The charge of the fragment ion.
    observed_mzs : np.array of float
        The experimentally observed fragment mzs.
    loss : float (default=0.0)
        The neutral loss weight to be applied.

    Returns
    -------
    mz_error : float
        The mz error on the nearest observed fragment ion.
    matched_mz_ind : int
        The index of the nearest observed fragment ion.
    """
    fragment_mz = (
        (base_mass + (frag_z * PROTON)) - loss
    )/frag_z
    matched_mz_ind = np.argmin(
        np.abs(observed_mzs - fragment_mz)
    )
    return observed_mzs[matched_mz_ind] - fragment_mz, matched_mz_ind


def get_matches(
        all_prosit_masses,
        prosit_intensities,
        observed_mzs,
        precursor_mz,
        observed_intensities,
        mz_accuracy,
        precursor_z,
    ):
    """ Function to find all the matched mz measuremnets between the
        predicted MS2 spectrum of a peptide and an observed MS2 spectra.

    Parameters
    ----------
    expected_mzs : np.array
        An array of the mz of expected fragments that could be generated
        for the peptide.
    prosit_intensities : np.array
        The predicted intensities from prosit.
    observed_mzs : np.array
        An array of the mz of the observed fragments from the MS spectrum.
    observed_intensities : np.array
        An array of the intensities observed for the corresponding observed
        mzs.
    mz_accuracy : float
        The accuracy of the m/z measurement for the observations.

    Returns
    -------
    matches : np.array
        A 2d array of shape (len(observed_mzs), n ion types considered).
        If there is no match between the observed mz and a given ion type
        the value of the corresponding array entry is 0. Otherwise it will
        be the fragment index.
    """
    # Loop over observed fragments matching them to all possible prosit ions.
    ordered_prosit = np.array(list(prosit_intensities.keys()), dtype='object')
    final_intensities = np.zeros(len(prosit_intensities))
    not_matched_intensities = {}
    possible_loss_ions = []
    mz_errors = []
    assigned_inds = []

    max_frag_charge = min(4, precursor_z+1)
    match_idx = 0
    for ion_type in all_prosit_masses:
        for charge in range(1, max_frag_charge):
            for fragment_idx in range(len(all_prosit_masses[ion_type])):
                mz_diff, matched_mz_ind = match_mz(
                    all_prosit_masses[ion_type][fragment_idx],
                    charge,
                    observed_mzs,
                    loss=0.0
                )
                ion_code = f'{ion_type}{fragment_idx+1}'
                if charge > 1:
                    ion_code += f'^{charge}'

                if abs(mz_diff) < mz_accuracy:
                    if ion_code in ordered_prosit:
                        match_idx = np.where(ordered_prosit == ion_code)[0]
                        final_intensities[match_idx] = observed_intensities[matched_mz_ind]
                        mz_errors.append(mz_diff)
                    else:
                        not_matched_intensities[ion_code] = observed_intensities[matched_mz_ind]
                    assigned_inds.append(matched_mz_ind)
                else:
                    for loss in NEUTRAL_LOSSES:
                        loss_mz_diff, loss_matched_mz_ind = match_mz(
                            all_prosit_masses[ion_type][fragment_idx],
                            precursor_z,
                            observed_mzs,
                            loss=loss
                        )
                        if abs(loss_mz_diff) < mz_accuracy:
                            possible_loss_ions.append(
                                observed_intensities[loss_matched_mz_ind]
                            )

    precursor_inte, assigned_inds = check_for_precursor_peak(
        observed_mzs,
        observed_intensities,
        precursor_mz,
        mz_accuracy,
        assigned_inds,
    )

    # Select the intensities not assigned to anything.
    unassigned_inds = [
        x for x in range(len(observed_intensities)) if x not in assigned_inds
    ]
    unassigned_intes = observed_intensities[unassigned_inds]

    match_info = {
        'matched_intensities': final_intensities,
        'ordered_prosit_ions': ordered_prosit,
        'ordered_prosit_intes': np.array(
            [prosit_intensities[x] for x in ordered_prosit]
        ),
    }

    return (
        match_info,
        precursor_inte,
        not_matched_intensities,
        unassigned_intes,
        mz_errors,
        possible_loss_ions,
    )

def _get_mz_error_stats(mz_errors):
    """ Function to get the median and variance of the mz errors on prosit matched fragments.

    Parameters
    ----------
    mz_errors : list of float
        The mz error for each matched ion.

    Returns
    -------
    median_mz_error : float
        The median mz error on the matched ions.
    mz_error_variance : float
        The variance of the mz errors.
    """
    # Get the error in matching each fragment m/z.
    if mz_errors:
        if len(mz_errors) > 1:
            mz_error_variance = variance(mz_errors)
        else:
            mz_error_variance = -1
        abs_mz_errors = [abs(x) for x in mz_errors]
        median_mz_error = median(abs_mz_errors)
        return median_mz_error, mz_error_variance
    return -1, -1

def get_coverage(pep_len, ion_array, per_letter=False):
    """ Function to get the coverage for an ion series.
    """
    y_inds = [
        int(ion.split('^')[0][1:]) for ion in ion_array if ion.startswith('y')
    ]
    b_inds = [
        int(ion.split('^')[0][1:]) for ion in ion_array if ion.startswith('b')
    ]
    y_rev_inds = [pep_len-x for x in y_inds]
    total_coverage = len(set(b_inds+y_rev_inds))/(pep_len-1)

    if per_letter:
        b_coverage = len(set(b_inds))/(pep_len-1)
        y_coverage = len(set(y_inds))/(pep_len-1)
        return total_coverage, b_coverage, y_coverage
    return total_coverage

def calculate_spectral_features(
        df_row,
        ptm_id_weights,
        mz_accuracy,
        model,
        ox_flag,
        spectral_angle_only=False,
    ):
    """ Function to extract the ion intensities from the true spectra which match
    """
    sequence = df_row[PEPTIDE_KEY]
    seq_len = len(sequence)
    n_frags_possible = seq_len - 1

    prosit_preds = df_row[PROSIT_IONS_KEY]
    potential_ion_mzs, precursor_weight = get_ion_masses(
        sequence,
        ptm_id_weights,
        df_row[PTM_SEQ_KEY]
    )

    precursor_mz = (precursor_weight + (PROTON*df_row[CHARGE_KEY]))/df_row[CHARGE_KEY]

    (
        match_info, precursor_inte, unmatched_intensities,
        unassigned_intensities, mz_errors, possible_alts,
    ) = get_matches(
        potential_ion_mzs,
        prosit_preds,
        df_row[MZS_KEY],
        precursor_mz,
        df_row[INTENSITIES_KEY],
        mz_accuracy,
        df_row[CHARGE_KEY],
    )

    matched_intensities = match_info['matched_intensities']
    ordered_prosit_ions = match_info['ordered_prosit_ions']
    ordered_prosit_intes = match_info['ordered_prosit_intes']
    truly_matched_intes = matched_intensities[matched_intensities > 0.0]

    total_frags = 6*n_frags_possible
    total_frags_per_series = 3*n_frags_possible
    n_not_pred_not_found = (
        total_frags - (len(ordered_prosit_ions) + len(unmatched_intensities))
    )
    n_not_pred_not_found_y = (
        total_frags_per_series - (
            len([x for x in ordered_prosit_ions if x.startswith('y')])
            - len([x for x in unmatched_intensities if x.startswith('y')])
        )
    )
    n_not_pred_not_found_b = (
        total_frags_per_series - (
            len([x for x in ordered_prosit_ions if x.startswith('b')])
            - len([x for x in unmatched_intensities if x.startswith('b')])
        )
    )
    df_row['spectrumDensity'] = len(df_row[MZS_KEY])/(df_row[MZS_KEY].max() - df_row[MZS_KEY].min())
    df_row['nNotPredNotFoundDivFrags'] = n_not_pred_not_found/n_frags_possible
    df_row['foundNotPredRatio'] = len(unmatched_intensities)/n_not_pred_not_found

    median_mz_error, mz_error_variance = _get_mz_error_stats(mz_errors)

    matched_l2_norm = np.linalg.norm(matched_intensities, ord=2)
    total_l2_norm = np.linalg.norm(df_row[INTENSITIES_KEY], ord=2)

    if matched_l2_norm:
        normed_matched_intensities = matched_intensities/matched_l2_norm
    else:
        normed_matched_intensities = matched_intensities

    df_row[SPECTRAL_ANGLE_KEY] = calculate_spectral_angle(
        normed_matched_intensities,
        ordered_prosit_intes,
    )
    if spectral_angle_only:
        return df_row

    major_pred_inds = (ordered_prosit_intes >= PROSIT_MAJOR_MINOR_CUT_OFF)

    major_prosit_preds = ordered_prosit_intes[major_pred_inds]
    minor_prosit_preds = ordered_prosit_intes[~major_pred_inds]

    major_matched_ions = normed_matched_intensities[major_pred_inds]
    minor_matched_ions = normed_matched_intensities[~major_pred_inds]

    n_major_pred = len(major_prosit_preds)
    n_minor_pred = len(minor_prosit_preds)

    if matched_l2_norm:
        major_prosit_l2_norm = np.linalg.norm(major_prosit_preds, ord=2)
        minor_prosit_l2_norm = np.linalg.norm(minor_prosit_preds, ord=2)

        major_l2_norm = np.linalg.norm(major_matched_ions, ord=2)
        minor_l2_norm = np.linalg.norm(minor_matched_ions, ord=2)

        n_major_matched = len(major_matched_ions[major_matched_ions > 0.0])
        n_minor_matched = len(truly_matched_intes) - n_major_matched

        if major_l2_norm:
            major_matched_ions /= major_l2_norm
        if minor_l2_norm:
            minor_matched_ions /= minor_l2_norm
        if major_prosit_l2_norm:
            major_prosit_preds /= major_prosit_l2_norm
        if minor_prosit_l2_norm:
            minor_prosit_preds /= minor_prosit_l2_norm

        if n_major_matched > 0:
            spearman_major = spearmanr(major_matched_ions, major_prosit_preds)[0]
            if not np.isnan(spearman_major):
                df_row['spearmanMajorIons'] = spearman_major
            else:
                df_row['spearmanMajorIons'] = -1.0
        else:
            df_row['spearmanMajorIons'] = 0.0

        if n_minor_matched > 0:
            spearman_minor = spearmanr(minor_matched_ions, minor_prosit_preds)[0]
            if not np.isnan(spearman_minor):
                df_row['spearmanMinorIons'] = spearman_minor
            else:
                df_row['spearmanMinorIons'] = -1.0
        else:
            df_row['spearmanMinorIons'] = 0.0

    else:
        n_major_matched = 0.0
        n_minor_matched = 0.0
        df_row['spearmanMajorIons'] = 0.0
        df_row['spearmanMinorIons'] = 0.0

    major_unmatched_intensities ={
        k: v for k, v in unmatched_intensities.items() if (
            v/total_l2_norm > SPECTRUM_MAJOR_MINOR_CUT_OFF
        )
    }
    n_major_unmatched = len(major_unmatched_intensities)
    n_minor_unmatched = len(unmatched_intensities) - n_major_unmatched

    df_row['yMajorFoundNotPred'] =  len(
        [x for x in major_unmatched_intensities if x.startswith('y')]
    )
    n_major_not_matchable = len(
        [x for x in unassigned_intensities if x/total_l2_norm > SPECTRUM_MAJOR_MINOR_CUT_OFF]
    )
    n_minor_not_matchable = len(unassigned_intensities) - n_major_unmatched


    df_row['nMajorPredNotFoundDivFrags'] = (n_major_pred - n_major_matched)/n_frags_possible
    df_row['nMinorPredNotFoundDivFrags'] = (n_minor_pred - n_minor_matched)/n_frags_possible
    df_row[MAE_KEY] = median_absolute_error(
        normed_matched_intensities,
        ordered_prosit_intes,
    )

    if len(truly_matched_intes) > 0:
        spearman_total = spearmanr(normed_matched_intensities, ordered_prosit_intes)[0]
        pearson_total = pearsonr(normed_matched_intensities, ordered_prosit_intes)[0]
        if not np.isnan(spearman_total):
            df_row[SPEARMAN_KEY] = spearman_total
        else:
            df_row[SPEARMAN_KEY] = -1.0
        if not np.isnan(pearson_total):
            df_row[PEARSON_KEY] = pearson_total
        else:
            df_row[PEARSON_KEY] = -1.0
    else:
        df_row[SPEARMAN_KEY] = 0.0
        df_row[PEARSON_KEY] = 0.0
    df_row['nMajorMatchedDivFrags'] = n_major_matched/n_frags_possible
    df_row['nMinorMatchedDivFrags'] = n_minor_matched/n_frags_possible
    df_row[MATCHED_IONS_KEY] = len(truly_matched_intes)/n_frags_possible
    df_row['matchedIntensityRatio'] = (
            np.sum(truly_matched_intes) + precursor_inte
        )/np.sum(df_row[INTENSITIES_KEY])
    df_row['nMajorFoundNotPredDivFrags'] = n_major_unmatched/n_frags_possible
    df_row['nMinorFoundNotPredDivFrags'] = n_minor_unmatched/n_frags_possible

    df_row['nMajorNotMatchableDivFrags'] = n_major_not_matchable/n_frags_possible
    df_row['nMinorNotMatchableDivFrags'] = n_minor_not_matchable/n_frags_possible
    df_row[OBS_NOT_PRED_KEY] = len(unmatched_intensities)/n_frags_possible
    df_row[NOT_ASSIGNED_KEY] = len(unassigned_intensities)/n_frags_possible
    df_row[LOSS_IONS_KEY] = len(possible_alts)/n_frags_possible

    df_row['matchedDivPred'] = len(truly_matched_intes)/len(prosit_preds)
    df_row['matchedDivObserved'] = len(truly_matched_intes)/len(df_row[INTENSITIES_KEY])

    n_pred_not_found = len(prosit_preds) - len(truly_matched_intes)
    df_row['fracPredNotFound'] = n_pred_not_found/len(prosit_preds)

    df_row['foundNotPredDivPred'] = len(unmatched_intensities)/len(prosit_preds)
    df_row['foundNotPredDivNotPred'] = len(unmatched_intensities)/(
        (n_frags_possible*6) - len(prosit_preds)
    )

    df_row['fracNotMatchable'] = len(unassigned_intensities)/len(df_row[INTENSITIES_KEY])


    if precursor_inte:
        df_row[PRECURSOR_INTE_KEY] = precursor_inte/(matched_l2_norm+precursor_inte)
    else:
        df_row[PRECURSOR_INTE_KEY] = 0.0

    df_row[FRAG_MZ_ERR_MED_KEY] = median_mz_error
    df_row[FRAG_MZ_ERR_VAR_KEY] = mz_error_variance

    df_row = get_kr_feats(
        sequence, matched_intensities, ordered_prosit_intes, ordered_prosit_ions, df_row
    )

    y_inds = [idx for idx in range(len(ordered_prosit_ions)) if ordered_prosit_ions[idx][0] == 'y']
    y_matched_ions = matched_intensities[y_inds]
    n_y_pred_not_found = len(y_matched_ions[y_matched_ions == 0.0])
    n_b_pred_not_found = n_pred_not_found - n_y_pred_not_found
    y_pred_ions = ordered_prosit_intes[y_inds]

    ordered_matched_ions = ordered_prosit_ions[matched_intensities > 0.0]
    pred_not_found_ions = ordered_prosit_ions[matched_intensities == 0.0]

    df_row['predNotFoundCoverage'] = get_coverage(
        seq_len, pred_not_found_ions
    )

    major_matched_filter = (
        matched_intensities > 0.0
    ) & (
        ordered_prosit_intes > PROSIT_MAJOR_MINOR_CUT_OFF
    )
    df_row['observedCoverage'] = get_coverage(
        seq_len, ordered_matched_ions.tolist() + list(unmatched_intensities.keys())
    )

    df_row = get_coverage_features(
        df_row,
        seq_len,
        major_matched_filter,
        ordered_matched_ions,
        ordered_prosit_ions,
    )
    df_row = get_per_series_features(
        df_row,
        n_frags_possible,
        unmatched_intensities,
        n_not_pred_not_found,
        n_pred_not_found,
        n_y_pred_not_found,
        n_not_pred_not_found_y,
        n_b_pred_not_found,
        n_not_pred_not_found_b,
    )

    y_pred_ratio = (len(y_pred_ions)/len(prosit_preds))
    y_filter = [
        idx for idx in range(len(ordered_prosit_ions)) if ordered_prosit_ions[idx][0] == 'y'
    ]
    b_filter = [
        idx for idx in range(len(ordered_prosit_ions)) if ordered_prosit_ions[idx][0] == 'b'
    ]
    if not truly_matched_intes.size:
        df_row['maxTypeSpectralAngle'] = 0.0
        df_row['minTypeSpectralAngle'] = 0.0
        df_row['maxTypeMajorMatchedFrags'] = 0.0
        df_row['minTypeMajorMatchedFrags'] = 0.0
        df_row['typePredRatio'] = max([y_pred_ratio, 1-y_pred_ratio])
        df_row['typeMatchedRatio'] = -1
        df_row['typeIntensityRatio'] = -1
    else:
        y_spectral_angle = calculate_spectral_angle(
            normed_matched_intensities[y_filter],
            ordered_prosit_intes[y_filter],
        )
        b_spectral_angle = calculate_spectral_angle(
            normed_matched_intensities[b_filter],
            ordered_prosit_intes[b_filter],
        )
        y_matched_ratio = len(y_matched_ions)/len(matched_intensities)
        y_inte_ratio = sum(y_matched_ions)/matched_l2_norm

        if df_row['bIsDominantIonSeries'] == 1:
            df_row['maxTypeSpectralAngle'] = b_spectral_angle
            df_row['minTypeSpectralAngle'] = y_spectral_angle
            df_row['typeMatchedRatio'] = 1 - y_matched_ratio
            df_row['typePredRatio'] = 1 - y_pred_ratio
            df_row['typeIntensityRatio'] = 1 - y_inte_ratio
            df_row['maxTypeMajorMatchedFrags'] = len(
                [x for x in ordered_prosit_ions[major_matched_filter] if x.startswith('b')]
            )/n_frags_possible
        elif df_row['yIsDominantIonSeries'] == 1:
            df_row['maxTypeSpectralAngle'] = y_spectral_angle
            df_row['minTypeSpectralAngle'] = b_spectral_angle
            df_row['typePredRatio'] = y_pred_ratio
            df_row['typeMatchedRatio'] = y_matched_ratio
            df_row['typeIntensityRatio'] = y_inte_ratio
            df_row['maxTypeMajorMatchedFrags'] = len(
                [x for x in ordered_prosit_ions[major_matched_filter] if x.startswith('y')]
            )/n_frags_possible
        else:
            df_row['maxTypeSpectralAngle'] = max([b_spectral_angle, y_spectral_angle])
            df_row['minTypeSpectralAngle'] = min([b_spectral_angle, y_spectral_angle])
            df_row['typePredRatio'] = 0.5
            df_row['typeMatchedRatio'] = max([y_matched_ratio, 1-y_matched_ratio])
            df_row['typeIntensityRatio'] = max([y_inte_ratio, 1-y_inte_ratio])
            df_row['maxTypeMajorMatchedFrags'] = df_row['nMajorMatchedDivFrags']/2

    df_row['nRawPeaksDivFrags'] = len(df_row[INTENSITIES_KEY])/n_frags_possible

    df_row = get_deltas(
        df_row,
        ordered_prosit_ions,
        normed_matched_intensities,
        ordered_prosit_intes,
        model,
        ox_flag,
    )

    return df_row

def get_coverage_features(
        df_row,
        seq_len,
        major_matched_filter,
        ordered_matched_ions,
        ordered_prosit_ions
    ):
    """ Function to get coverage features.
    """
    df_row['predCoverage'], b_pred_coverage, y_pred_coverage = get_coverage(
        seq_len, ordered_prosit_ions, per_letter=True
    )

    df_row['matchedCoverage'], b_matched_coverage, y_matched_coverage = get_coverage(
        seq_len, ordered_matched_ions, per_letter=True
    )

    df_row['majorMatchedCoverage'] = get_coverage(
        seq_len, ordered_prosit_ions[major_matched_filter]
    )

    if b_pred_coverage < y_pred_coverage:
        df_row['minMatchedCoverage'] = b_matched_coverage
        df_row['maxMatchedCoverage'] = y_matched_coverage
        df_row['yIsDominantIonSeries'] = 1
        df_row['bIsDominantIonSeries'] = 0
    elif b_pred_coverage > y_pred_coverage:
        df_row['yIsDominantIonSeries'] = 0
        df_row['bIsDominantIonSeries'] = 1
        df_row['minMatchedCoverage'] = y_matched_coverage
        df_row['maxMatchedCoverage'] = b_matched_coverage
    else:
        df_row['minMatchedCoverage'] = min([y_matched_coverage, b_matched_coverage])
        df_row['maxMatchedCoverage'] = max([y_matched_coverage, b_matched_coverage])
        df_row['yIsDominantIonSeries'] = 0
        df_row['bIsDominantIonSeries'] = 0

    return df_row

def get_per_series_features(
        df_row,
        n_frags_possible,
        unmatched_intensities,
        n_not_pred_not_found,
        n_pred_not_found,
        n_y_pred_not_found,
        n_not_pred_not_found_y,
        n_b_pred_not_found,
        n_not_pred_not_found_b,
    ):
    """ Function to get features per ion series.
    """
    if df_row['yIsDominantIonSeries'] == 1:
        df_row['maxTypePredNotFound'] = n_y_pred_not_found/n_frags_possible
        df_row['minTypeFoundNotPred'] = len(
            [
                x for x in unmatched_intensities if x.startswith('b')
            ]
        )/n_frags_possible
        df_row['maxTypeFoundNotPred'] = len(
            [
                x for x in unmatched_intensities if x.startswith('y')
            ]
        )/n_frags_possible
        df_row['maxTypeNotFoundNotPred'] = n_not_pred_not_found_y/n_frags_possible
        df_row['minTypeNotFoundNotPred'] = n_not_pred_not_found_b/n_frags_possible
    elif df_row['bIsDominantIonSeries'] == 1:
        df_row['maxTypePredNotFound'] = n_b_pred_not_found/n_frags_possible
        df_row['minTypeFoundNotPred'] = len(
            [
                x for x in unmatched_intensities if x.startswith('y')
            ]
        )/n_frags_possible
        df_row['maxTypeFoundNotPred'] = len(
            [
                x for x in unmatched_intensities if x.startswith('b')
            ]
        )/n_frags_possible
        df_row['maxTypeNotFoundNotPred'] = n_not_pred_not_found_b/n_frags_possible
        df_row['minTypeNotFoundNotPred'] = n_not_pred_not_found_y/n_frags_possible
    else:
        df_row['maxTypePredNotFound'] = n_pred_not_found/(2*n_frags_possible)
        b_found_not_pred = len(
            [
                x for x in unmatched_intensities if x.startswith('y')
            ]
        )/n_frags_possible
        y_found_not_pred = len(
            [
                x for x in unmatched_intensities if x.startswith('b')
            ]
        )/n_frags_possible
        df_row['minTypeFoundNotPred'] = min([b_found_not_pred, y_found_not_pred])/n_frags_possible
        df_row['maxTypeFoundNotPred'] = max([b_found_not_pred, y_found_not_pred])/n_frags_possible
        df_row['maxTypeNotFoundNotPred'] = n_not_pred_not_found/(2*n_frags_possible)
        df_row['minTypeNotFoundNotPred'] = n_not_pred_not_found/(2*n_frags_possible)

    return df_row


def get_kr_feats(sequence, matched_intensities, ordered_prosit_intes, ordered_prosit_ions, df_row):
    """ Function to get the difference between the spectral angle of.
    """
    seq_len = len(sequence)
    y_kr = seq_len
    b_kr = seq_len
    n_possible_kr = 0
    for idx, entry in enumerate(sequence):
        if entry in 'KR':
            b_kr = seq_len - (idx+2)
            n_possible_kr = b_kr
            break

    if b_kr < seq_len:
        for idx, entry in enumerate(sequence[::-1]):
            if entry in 'KR':
                y_kr = seq_len - (idx+2)
                n_possible_kr += y_kr
                break

    kr_inds = []
    if b_kr < seq_len:
        for idx, ion_name in enumerate(ordered_prosit_ions):
            ion_code = ion_name[0]
            frag_idx = int(ion_name[1:].split('^')[0])
            if ion_code == 'b':
                if frag_idx > b_kr:
                    kr_inds.append(idx)
            else:
                if frag_idx > y_kr:
                    kr_inds.append(idx)

        kr_prosit_intensities = ordered_prosit_intes[kr_inds]
        kr_matched_intensities = matched_intensities[kr_inds]

        if kr_prosit_intensities.size:
            df_row['fracMatchedKR'] = len(kr_matched_intensities)/len(kr_prosit_intensities)
        else:
            df_row['fracMatchedKR'] = 0.0
        df_row['possibleKrFragsDivTotal'] = n_possible_kr/(seq_len-1)

        if len(kr_prosit_intensities) < 3:
            df_row['spectralAngleKR'] = df_row[SPECTRAL_ANGLE_KEY]
        else:
            kr_spec_angle = calculate_spectral_angle(
                kr_matched_intensities,
                kr_prosit_intensities,
            )
            df_row['spectralAngleKR'] = kr_spec_angle
        return df_row
    df_row['spectralAngleKR'] = df_row[SPECTRAL_ANGLE_KEY]
    df_row['fracMatchedKR'] = 0.0
    df_row['possibleKrFragsDivTotal'] = 0.0

    return df_row

def fetch_mod_weight_dict(mods_df):
    """ Function to fetch a dictionary of the weights of all modifications.

    Parameters
    ----------
    mods_df : pd.DataFrame
        A DataFrame of the modifications present in data.

    Returns
    -------
    ptm_id_weights : dict
        A dict mapping ptm ids to the change in mass cause by it.
    """
    mod_ids = mods_df[PTM_ID_KEY].tolist()
    mod_weights = mods_df[PTM_WEIGHT_KEY]

    ptm_id_weights = dict(zip(mod_ids, mod_weights))
    ptm_id_weights[0] = 0.0 # Add entry for no modification.

    return ptm_id_weights

def create_spectral_features(spectral_df, mods_df, mz_accuracy):
    """ Function to calculate spectral features between experimental and prosit predicted
        spectra.

    Parameters
    ----------
    spectral_df : pd.DataFrame
        A DataFrame containing experimental and prosit predicted spectra.
    mods_df : pd.DataFrame
        A small DataFrame detailing the unique modifications seen in the data.
    mz_accuracy : float
        The m/z accuracy of the instrument which measured the spectra.

    Returns
    -------
    spectral_df : pd.DataFrame
        The input DataFrame with features added to describe the match between experimental
        and prosit predicted spectra.
    """
    ox_flag = get_ox_flag(mods_df)
    ptm_id_weights = fetch_mod_weight_dict(mods_df)
    model_path = 'model/reg.pkl'
    with pkg_resources.resource_stream(__name__, model_path) as fid:
        model = pickle.load(fid)

    spectral_df = spectral_df.apply(
        lambda df_row : calculate_spectral_features(
            df_row,
            ptm_id_weights,
            mz_accuracy,
            model,
            str(ox_flag),
        ),
        axis=1
    )

    spectral_features = SPECTRAL_FEATURES + DELTA_FEATURES

    replace_feats = [
        FRAG_MZ_ERR_VAR_KEY,
        FRAG_MZ_ERR_MED_KEY,
        'prositDeltaMedian',
        'prositDeltaPercentile90',
        'nDeltasAboveThreshold',
        'minPrositDelta',
        'typeMatchedRatio',
        'typeIntensityRatio',
        'spearmanR',
        'spearmanMajorIons',
        'spearmanMinorIons',
    ]
    for feat in replace_feats:
        if feat in spectral_features:
            spectral_df_var_median = spectral_df[
                spectral_df[feat] > -1
            ][feat].median()

            spectral_df[feat] = spectral_df[feat].replace(
                to_replace=-1,
                value=spectral_df_var_median
            )
    return spectral_df
