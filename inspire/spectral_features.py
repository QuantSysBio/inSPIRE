""" Functions for calculating spectral features.
"""

# from copyreg import pickle
from copy import deepcopy
import gc
from math import acos, pi
from statistics import mean, variance

import polars as pl
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import median_absolute_error

from inspire.constants import (
    CHARGE_KEY,
    INTENSITIES_KEY,
    LOSS_IONS_KEY,
    MAE_KEY,
    NEUTRAL_LOSSES,
    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,
    MATCHED_IONS_KEY,
    NOT_ASSIGNED_KEY,
    PRECURSOR_INTE_KEY,
    MZS_KEY,
    PEARSON_KEY,
    PEPTIDE_KEY,
    PROSIT_IONS_KEY,
    PROSIT_INTES_KEY,
    PROSIT_SEQ_KEY,
    PROTON,
    PTM_ID_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    SPEARMAN_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.mz_match import get_ion_masses, match_mz
from inspire.prosit_delta import get_deltas
from inspire.utils import get_ox_flag


SPECTRAL_FEATURES = [
    SPECTRAL_ANGLE_KEY,
    SPEARMAN_KEY,
    PEARSON_KEY,
    MAE_KEY,
    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,
    NOT_ASSIGNED_KEY,
    LOSS_IONS_KEY,
    PRECURSOR_INTE_KEY,
    'yIsDominantIonSeries',
    'bIsDominantIonSeries',
    'maxTypeSpectralAngle',
    'minTypeSpectralAngle',
    'fracMatchedKR',
    'possibleKrFragsDivTotal',
    'matchedCoverage',
    'minMatchedCoverage',
    'maxMatchedCoverage',
    'predCoverage',
    'predNotFoundCoverage',
    'nMajorMatchedDivFrags',
    'nMajorPredNotFoundDivFrags',
    'nMajorNotMatchableDivFrags',
    'spearmanMajorIons',
    'nMinorMatchedDivFrags',
    'nMinorNotMatchableDivFrags',
    'spearmanMinorIons',
    'spectrumDensity',
]

DELTA_FEATURES = [
    'prositDeltaQuartile1',
    'prositDeltaQuartile3',
    'prositDeltaMedian',
    'nDeltasAboveThreshold',
    'nDeltasAboveThresholdA',
    'nDeltasAboveZero',
    'maxPrositDelta',
    'minPrositDelta',
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


def get_matches(
        all_prosit_masses,
        prosit_intensities,
        observed_mzs,
        precursor_weight,
        observed_intensities,
        mz_accuracy,
        precursor_z,
        mz_units,
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
    observed_mzs = np.array(observed_mzs)
    observed_intensities = np.array(observed_intensities)

    possible_loss_ions = []
    mz_errors = []
    assigned_inds = []

    max_frag_charge = min(4, precursor_z+1)
    match_idx = 0
    for ion_type in all_prosit_masses:
        for charge in range(1, max_frag_charge):
            for fragment_idx in range(len(all_prosit_masses[ion_type])):
                ion_code = f'{ion_type}{fragment_idx+1}'
                if charge > 1:
                    ion_code += f'^{charge}'

                if ion_code not in ordered_prosit:
                    continue

                mass = all_prosit_masses[ion_type][fragment_idx] + (charge * PROTON)
                if mz_units == 'ppm':
                    mz_err = (mass/charge)*mz_accuracy*(10**-6)
                else:
                    mz_err = mz_accuracy

                mz_diff, matched_mz_ind = match_mz(
                    all_prosit_masses[ion_type][fragment_idx],
                    charge,
                    observed_mzs,
                    loss=0.0
                )

                if abs(mz_diff) < mz_err:
                    if ion_code in ordered_prosit:
                        match_idx = np.where(ordered_prosit == ion_code)[0]
                        final_intensities[match_idx] = observed_intensities[matched_mz_ind]
                        mz_errors.append(mz_diff)
                        for loss in NEUTRAL_LOSSES:
                            loss_mz_diff, loss_matched_mz_ind = match_mz(
                                all_prosit_masses[ion_type][fragment_idx],
                                precursor_z,
                                observed_mzs,
                                loss=loss
                            )
                            if abs(loss_mz_diff) < mz_err:
                                possible_loss_ions.append(
                                    observed_intensities[loss_matched_mz_ind]
                                )
                        assigned_inds.append(matched_mz_ind)

    precursor_mz = (precursor_weight + (PROTON*precursor_z))/precursor_z

    if mz_units == 'ppm':
        mz_err = precursor_mz*mz_accuracy*(10**-6)
    else:
        mz_err = mz_accuracy

    precursor_inte, assigned_inds = check_for_precursor_peak(
        observed_mzs,
        observed_intensities,
        precursor_mz,
        mz_err,
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
        unassigned_intes,
        mz_errors,
        possible_loss_ions,
    )

def get_mz_error_stats(mz_errors, mz_accu):
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
            mz_error_variance = mz_accu
        abs_mz_errors = [abs(x) for x in mz_errors]
        median_mz_error = mean(abs_mz_errors)
        return median_mz_error, mz_error_variance
    return mz_accu, mz_accu

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
        mz_units,
        model,
        ox_flag,
        delta_method,
        minimal_features=False,
    ):
    """ Function to extract the ion intensities from the true spectra which match
    """
    results = {}
    sequence = df_row[PEPTIDE_KEY]
    seq_len = len(sequence)
    n_frags_possible = seq_len - 1
    mz_array = np.array(df_row[MZS_KEY])
    intes_array = np.array(df_row[INTENSITIES_KEY])

    prosit_preds = dict(zip(df_row[PROSIT_IONS_KEY], df_row[PROSIT_INTES_KEY]))

    potential_ion_mzs, precursor_weight = get_ion_masses(
        sequence,
        ptm_id_weights,
        df_row[PTM_SEQ_KEY]
    )

    (
        match_info, precursor_inte, unassigned_intensities, mz_errors, possible_alts
    ) = get_matches(
        potential_ion_mzs,
        prosit_preds,
        mz_array,
        precursor_weight,
        df_row[INTENSITIES_KEY],
        mz_accuracy,
        df_row[CHARGE_KEY],
        mz_units,
    )
    matched_intensities = match_info['matched_intensities']
    ordered_prosit_ions = match_info['ordered_prosit_ions']
    ordered_prosit_intes = match_info['ordered_prosit_intes']
    truly_matched_intes = matched_intensities[matched_intensities > 0.0]

    median_mz_error, mz_error_variance = get_mz_error_stats(mz_errors, min(mz_accuracy, 0.04))

    matched_l2_norm = np.linalg.norm(matched_intensities, ord=2)
    total_l2_norm = np.linalg.norm(df_row[INTENSITIES_KEY], ord=2)

    if matched_l2_norm:
        normed_matched_intensities = matched_intensities/matched_l2_norm
    else:
        normed_matched_intensities = matched_intensities

    ordered_matched_ions = ordered_prosit_ions[matched_intensities > 0.0]

    results[SPECTRAL_ANGLE_KEY] = calculate_spectral_angle(
        normed_matched_intensities,
        ordered_prosit_intes,
    )

    results = get_coverage_features(
        results,
        seq_len,
        ordered_matched_ions,
        ordered_prosit_ions,
    )

    if len(truly_matched_intes) > 0:
        spearman_total = spearmanr(normed_matched_intensities, ordered_prosit_intes)[0]
        pearson_total = pearsonr(normed_matched_intensities, ordered_prosit_intes)[0]
        if not np.isnan(spearman_total):
            results[SPEARMAN_KEY] = float(spearman_total)
        else:
            results[SPEARMAN_KEY] = 0.0
        if not np.isnan(pearson_total):
            results[PEARSON_KEY] = float(pearson_total)
        else:
            results[PEARSON_KEY] = -1.0
    else:
        results[SPEARMAN_KEY] = 0.0
        results[PEARSON_KEY] = 0.0

    if minimal_features:
        return results

    mz_range = (mz_array.max() - mz_array.min())
    if mz_range > 0:
        results['spectrumDensity'] = len(mz_array)/mz_range
    else:
        results['spectrumDensity'] = 1.0

    major_pred_inds = ordered_prosit_intes >= PROSIT_MAJOR_MINOR_CUT_OFF

    major_prosit_preds = ordered_prosit_intes[major_pred_inds]
    minor_prosit_preds = ordered_prosit_intes[~major_pred_inds]

    major_matched_ions = normed_matched_intensities[major_pred_inds]
    minor_matched_ions = normed_matched_intensities[~major_pred_inds]

    n_major_pred = len(major_prosit_preds)

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
                results['spearmanMajorIons'] = float(spearman_major)
            else:
                results['spearmanMajorIons'] = 0.0
        else:
            results['spearmanMajorIons'] = 0.0

        if n_minor_matched > 0:
            spearman_minor = spearmanr(minor_matched_ions, minor_prosit_preds)[0]
            if not np.isnan(spearman_minor):
                results['spearmanMinorIons'] = float(spearman_minor)
            else:
                results['spearmanMinorIons'] = 0.0
        else:
            results['spearmanMinorIons'] = 0.0

    else:
        n_major_matched = 0.0
        n_minor_matched = 0.0
        results['spearmanMajorIons'] = 0.0
        results['spearmanMinorIons'] = 0.0

    n_major_not_matchable = len(
        [x for x in unassigned_intensities if x/total_l2_norm > SPECTRUM_MAJOR_MINOR_CUT_OFF]
    )
    n_minor_not_matchable = len(unassigned_intensities)


    results['nMajorPredNotFoundDivFrags'] = (n_major_pred - n_major_matched)/n_frags_possible

    results[MAE_KEY] = float(median_absolute_error(
        normed_matched_intensities,
        ordered_prosit_intes,
    ))

    results['nMajorMatchedDivFrags'] = n_major_matched/n_frags_possible
    results['nMinorMatchedDivFrags'] = n_minor_matched/n_frags_possible
    results[MATCHED_IONS_KEY] = len(truly_matched_intes)/n_frags_possible

    results['nMajorNotMatchableDivFrags'] = n_major_not_matchable/n_frags_possible
    results['nMinorNotMatchableDivFrags'] = n_minor_not_matchable/n_frags_possible

    results[NOT_ASSIGNED_KEY] = len(unassigned_intensities)/n_frags_possible
    results[LOSS_IONS_KEY] = len(possible_alts)/n_frags_possible


    if precursor_inte:
        results[PRECURSOR_INTE_KEY] = float(precursor_inte/(matched_l2_norm+precursor_inte))
    else:
        results[PRECURSOR_INTE_KEY] = 0.0

    results[FRAG_MZ_ERR_MED_KEY] = float(median_mz_error)
    results[FRAG_MZ_ERR_VAR_KEY] = float(mz_error_variance)

    results = get_kr_feats(
        sequence, matched_intensities, ordered_prosit_intes, ordered_prosit_ions, results
    )

    pred_not_found_ions = ordered_prosit_ions[matched_intensities == 0.0]

    results['predNotFoundCoverage'] = get_coverage(
        seq_len, pred_not_found_ions
    )

    y_filter = [
        idx for idx in range(len(ordered_prosit_ions)) if ordered_prosit_ions[idx][0] == 'y'
    ]
    b_filter = [
        idx for idx in range(len(ordered_prosit_ions)) if ordered_prosit_ions[idx][0] == 'b'
    ]
    if not truly_matched_intes.size:
        results['maxTypeSpectralAngle'] = 0.0
        results['minTypeSpectralAngle'] = 0.0
    else:
        y_spectral_angle = calculate_spectral_angle(
            normed_matched_intensities[y_filter],
            ordered_prosit_intes[y_filter],
        )
        b_spectral_angle = calculate_spectral_angle(
            normed_matched_intensities[b_filter],
            ordered_prosit_intes[b_filter],
        )

        if results['bIsDominantIonSeries'] == 1:
            results['maxTypeSpectralAngle'] = b_spectral_angle
            results['minTypeSpectralAngle'] = y_spectral_angle
        elif results['yIsDominantIonSeries'] == 1:
            results['maxTypeSpectralAngle'] = y_spectral_angle
            results['minTypeSpectralAngle'] = b_spectral_angle
        else:
            results['maxTypeSpectralAngle'] = max([b_spectral_angle, y_spectral_angle])
            results['minTypeSpectralAngle'] = min([b_spectral_angle, y_spectral_angle])

    precursor_mz = (precursor_weight + (PROTON*df_row[CHARGE_KEY]))/df_row[CHARGE_KEY]
    if mz_units == 'ppm':
        mz_err = precursor_mz*mz_accuracy*(10**-6)
    else:
        mz_err = mz_accuracy

    if delta_method == 'predictor':
        matched_dict = dict(zip(ordered_matched_ions.tolist(), truly_matched_intes.tolist()))

        results = get_deltas(
            results,
            df_row,
            mz_array,
            intes_array,
            prosit_preds,
            ordered_prosit_ions,
            normed_matched_intensities,
            ordered_prosit_intes,
            model,
            ox_flag,
            mz_err,
            matched_dict,
        )

    return results

def get_coverage_features(
        results,
        seq_len,
        ordered_matched_ions,
        ordered_prosit_ions
    ):
    """ Function to get coverage features.
    """
    results['predCoverage'], b_pred_coverage, y_pred_coverage = get_coverage(
        seq_len, ordered_prosit_ions, per_letter=True
    )

    results['matchedCoverage'], b_matched_coverage, y_matched_coverage = get_coverage(
        seq_len, ordered_matched_ions, per_letter=True
    )

    if b_pred_coverage < y_pred_coverage:
        results['minMatchedCoverage'] = b_matched_coverage
        results['maxMatchedCoverage'] = y_matched_coverage
        results['yIsDominantIonSeries'] = 1
        results['bIsDominantIonSeries'] = 0
    elif b_pred_coverage > y_pred_coverage:
        results['yIsDominantIonSeries'] = 0
        results['bIsDominantIonSeries'] = 1
        results['minMatchedCoverage'] = y_matched_coverage
        results['maxMatchedCoverage'] = b_matched_coverage
    else:
        results['minMatchedCoverage'] = min([y_matched_coverage, b_matched_coverage])
        results['maxMatchedCoverage'] = max([y_matched_coverage, b_matched_coverage])
        results['yIsDominantIonSeries'] = 0
        results['bIsDominantIonSeries'] = 0

    return results


def get_kr_feats(sequence, matched_intensities, ordered_prosit_intes, ordered_prosit_ions, results):
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
            results['fracMatchedKR'] = len(
                kr_matched_intensities[kr_matched_intensities > 0]
            )/len(kr_prosit_intensities)
        else:
            results['fracMatchedKR'] = len(
                matched_intensities[matched_intensities > 0]
            )/len(ordered_prosit_ions)
        results['possibleKrFragsDivTotal'] = n_possible_kr/(seq_len-1)

        return results

    results['fracMatchedKR'] = 0.0
    results['possibleKrFragsDivTotal'] = 0.0

    return results

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
    if mods_df.empty:
        return {0: 0.0}
    mod_ids = mods_df[PTM_ID_KEY]
    mod_weights = mods_df[PTM_WEIGHT_KEY]

    ptm_id_weights = dict(zip(mod_ids, mod_weights))
    ptm_id_weights[0] = 0.0 # Add entry for no modification.

    return ptm_id_weights


def create_spectral_features(mods_df, config, task_id, model):
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
    spectral_df = pl.read_parquet(
        f'{config.output_folder}/temp_{task_id}_in.parquet'
    )

    ox_flag = get_ox_flag(mods_df)

    ptm_id_weights = fetch_mod_weight_dict(mods_df)

    spectral_df = spectral_df.with_columns(
        pl.struct([
            CHARGE_KEY,
            'collisionEnergy',
            INTENSITIES_KEY,
            MZS_KEY,
            PEPTIDE_KEY,
            PROSIT_INTES_KEY,
            PROSIT_IONS_KEY,
            PROSIT_SEQ_KEY,
            PTM_SEQ_KEY,
        ]).apply(
            lambda df_row : calculate_spectral_features(
                df_row,
                ptm_id_weights,
                config.mz_accuracy,
                config.mz_units,
                model,
                str(ox_flag),
                config.delta_method,
                minimal_features=config.minimal_features,
            ),
            skip_nulls=False,
        ).alias('spectralResults')
    )

    spectral_df = spectral_df.filter(pl.col('spectralResults').is_not_null())
    spectral_df = spectral_df.drop(
        [MZS_KEY, INTENSITIES_KEY, PROSIT_INTES_KEY, PROSIT_IONS_KEY]
    )
    spectral_df = spectral_df.unnest('spectralResults')

    spectral_features = deepcopy(SPECTRAL_FEATURES)
    if config.delta_method != 'ignore' and not config.minimal_features:
        spectral_features += DELTA_FEATURES


    spectral_df.write_parquet(
        f'{config.output_folder}/temp_{task_id}_out.parquet'
    )
    del spectral_df
    gc.collect()
