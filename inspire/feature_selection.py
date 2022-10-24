""" Functions for running automated feature selection.
"""
from itertools import combinations
from subprocess import CalledProcessError

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

from inspire.constants import (
    CHARGE_KEY,
    DELTA_SCORE_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,
    LABEL_KEY,
    LOSS_IONS_KEY,
    MASS_DIFF_KEY,
    NOT_ASSIGNED_KEY,
    OKCYAN_TEXT,
    OUT_SCORE_KEY,
    PEARSON_KEY,
    PRECURSOR_INTE_KEY,
    PREFIX_KEYS,
    SPEARMAN_KEY,
    SUFFIX_KEYS,
    SEQ_LEN_KEY,
    SOURCE_INDEX_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.rescore import apply_rescoring
from inspire.spectral_features import DELTA_FEATURES

BASE_FEATURES = [
    SPECTRAL_ANGLE_KEY,
    DELTA_SCORE_KEY,
    ENGINE_SCORE_KEY,
]

FIRST_ADDITIONS = [
    'deltaRT',
    SEQ_LEN_KEY,
]

SPEARMAN_METRICS = [
    SPEARMAN_KEY,
    'spearmanMinorIons',
    'spearmanMajorIons',
    'medianAbsoluteError',
    PEARSON_KEY,
]

COVERAGE_FEATURES = [
    'matchedCoverage',
    'minMatchedCoverage',
    'maxMatchedCoverage',
    'prositDeltaQuartile1',
    'prositDeltaQuartile3',
    'prositDeltaMedian',
    'nDeltasAboveThreshold',
    'nDeltasAboveZero',
    'minPrositDelta',
    'maxPrositDelta',
]

AA_FEATS = [
    'possibleKrFragsDivTotal',
    'fracMatchedKR',
    'fracKR',
    'fracC',
    'missedCleavages',
    'nVarMods',
    'fracUnique',
    'nRepeatedResidues',
    'seqLenMeanDiff',
]

INTENSITY_FEATURES = [
    'nMajorNotMatchableDivFrags',
    'nMinorNotMatchableDivFrags',
    PRECURSOR_INTE_KEY,
    'nMajorMatchedDivFrags',
    'nMinorMatchedDivFrags',
]

MZ_DISTRIBUTION = [
    CHARGE_KEY,
    'avgResidueMass',
    MASS_DIFF_KEY,
    FRAG_MZ_ERR_MED_KEY,
    FRAG_MZ_ERR_VAR_KEY,
    'spectrumDensity',
    LOSS_IONS_KEY,
    'fromChimera',
]



FOUND_NOT_PRED_FEATS = [
    NOT_ASSIGNED_KEY,
    'nMajorPredNotFoundDivFrags',
]


YB_DIFF_FEATS = [
    'maxTypeSpectralAngle',
    'minTypeSpectralAngle',
    'yIsDominantIonSeries',
    'bIsDominantIonSeries',
]


SPECTRAL_FEATURE_GROUPS = [
    SPEARMAN_METRICS,
    COVERAGE_FEATURES,
    AA_FEATS,
    INTENSITY_FEATURES,
    MZ_DISTRIBUTION,
    FOUND_NOT_PRED_FEATS,
    YB_DIFF_FEATS,
]

SPECTRAL_GROUP_NAMES = [
    'Spearman R Features',
    'Coverage Features',
    'Amino Acid Features',
    'Matched/Unmatched Intensity Features',
    'MZ Distribution Features',
    'Found but not Predicted Features',
    'Y/B Ion Difference Features',
]

DEFAULT_FEATURE_SET = [
    'engineScore',
    'deltaScore',
    'sequenceLength',
    'seqLenMeanDiff',
    'charge',
    'avgResidueMass',
    'nVarMods',
    'spectralAngle',
    'deltaRT',
    SPEARMAN_KEY,
    PEARSON_KEY,
    'spearmanMajorIons',
    'medianAbsoluteError',
    'matchedCoverage',
    'maxMatchedCoverage',
    'minMatchedCoverage',
    'minPrositDelta',
    'prositDeltaQuartile1',
    'prositDeltaMedian',
    'prositDeltaQuartile3',
    'maxPrositDelta',
    'nDeltasAboveZero',
    'nDeltasAboveThreshold',
    'fracUnique',
    'nRepeatedResidues',
    'medianFragmentMzError',
    'fragmentMzErrorVariance',
    'nMajorMatchedDivFrags',
    'nMinorMatchedDivFrags',
    'nMajorNotMatchableDivFrags',
    'nMinorNotMatchableDivFrags',
    'maxTypeSpectralAngle',
    'yIsDominantIonSeries',
    LOSS_IONS_KEY,
    'predNotFoundCoverage',
    'fracC',
    'fracKR',
    'fracMatchedKR',
    'spectrumDensity',
    'fromChimera',
    'missedCleavages',
]

OUTSTANDING_CONFOUNDING_VARIABLES = [
    SOURCE_INDEX_KEY,
]

THRESHOLD_VALUE = 20


def generate_one_hot_entries(all_features_df, key):
    """ Function to generate one hot feature columns.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        The DataFrame with Percolator features.
    key : str
        The column to one hot encoded.

    Returns
    -------
    all_features_df : pd.DataFrame
        The updated DataFrame with one hot encoding columns.
    one_hot_columns : list of str
        A list of the one hot columns created.
    """
    total_count = all_features_df.shape[0]
    grouped_df = all_features_df.groupby(key).size().reset_index(name='counts')

    if grouped_df.shape[0] > 10:
        grouped_df = grouped_df[grouped_df['counts'] >= total_count/THRESHOLD_VALUE]
    values_above_threshold = grouped_df[key].tolist()

    if len(values_above_threshold) > 10:
        values_above_threshold = values_above_threshold[:10]

    for uval in values_above_threshold:
        value = int(uval)
        all_features_df[f'{key}_{value}'] = all_features_df[key].apply(
            lambda x, val=value: 1 if x == val else 0
        )
    return all_features_df, [f'{key}_{int(value)}' for value in values_above_threshold]


def get_feature_set_performance(all_features_df, features, output_folder, fdr, rescore_method):
    """ Function to get the number of PSMs identified at a given FDR for a given feature set.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        A DataFrame containing all possible percolator features.
    features : list of str
        A list of features to use for this Percolator call.
    output_folder : str
        The output folder where inSPIRE writes.
    fdr : float
        The FDRs threshold required.

    Returns
    -------
    performance : int
        The number of psms identified at the given FDR threshold.
    """
    prefix_keys = PREFIX_KEYS[rescore_method]
    out_score_key = OUT_SCORE_KEY[rescore_method]

    input_df = all_features_df[prefix_keys + features + SUFFIX_KEYS[rescore_method]]
    input_df.to_csv(f'{output_folder}/feature_test.tab', sep='\t', index=False)
    try:
        psms = apply_rescoring(
            output_folder,
            'feature_test.tab',
            fdr,
            rescore_method,
            'feature_test',
        )
    except CalledProcessError as _:
        return 0

    return psms[psms[out_score_key] >= 0.0].shape[0]

def get_best_feature_set(fe_df, feature_set):
    """ Function to get the best combination of features from a feature set
        for each number of features.

    Parameters
    ----------
    fe_df : pd.DataFrame
        The DataFrame containg all features.
    feature_set : list of str
        A list of the features we wish to select between.

    Returns
    -------
    features_sets_to_test : list of list of str
        A list of the best feature combinations for each number of features.
    """
    fe_train, fe_test = train_test_split(fe_df, test_size=0.33, random_state=42)
    fe_train = fe_train[fe_train['testMetric'] != -1]
    fe_test['trueLabel'] = fe_test['testMetric'].apply(lambda x : 1 if x == -1 else x)
    features_sets_to_test = []
    for i in range(1, len(feature_set)+1):
        max_r2 = -1000
        best_features = []
        for feature_selection in combinations(feature_set, i):
            feature_selection = list(feature_selection)
            reg = LogisticRegression(max_iter=1000, random_state=42).fit(
                fe_train[feature_selection + BASE_FEATURES],
                fe_train['testMetric'],
            )

            preds = reg.predict_proba(
                fe_test[feature_selection + BASE_FEATURES]
            )[:, 1]

            precision, recall, _ = precision_recall_curve(fe_test['trueLabel'], preds)
            feat_set_r2 = auc(recall, precision)

            if feat_set_r2 > max_r2:
                max_r2 = feat_set_r2
                best_features = feature_selection

        features_sets_to_test.append(best_features)
    return features_sets_to_test

def add_feature_collection(
        all_features_df,
        feature_set,
        new_features,
        baseline_performance,
        config,
        max_features,
    ):
    """ Function to select the best feature set to add from a given subset of features.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        A DataFrame containing all possible percolator features.
    feature_set : list of str
        A list of already selected features.
    new_features : list of str
        A list of new features to be selected from.
    baseline_performance : int
        The number of PSMs identified from the existing feature set.
    config : inspire.config.Config
        The Config object for the experiment.
    with_spectral_angle : bool
        Flag indicating whether to use spectral angle with the other features.

    Returns
    -------
    baseline_performance : float
        The maximum number of PSMs found with a new feature set.
    feature_set : list of str
        The best feature set found.
    """
    new_features = remove_excluded_features(
        new_features, all_features_df, config.exclude_features
    )

    if not new_features:
        return baseline_performance, feature_set
    scaler = StandardScaler().fit(all_features_df[new_features + feature_set])
    all_features_df[new_features + feature_set] = scaler.transform(
        all_features_df[new_features + feature_set]
    )
    deltapro_feature_sets = get_best_feature_set(
        all_features_df,
        new_features,
    )
    max_new_performance = 0
    best_adjusted_performance = 0
    best_new_feature_set = []
    all_features_df[new_features + feature_set] = scaler.inverse_transform(
        all_features_df[new_features + feature_set]
    )
    for new_feature_set in deltapro_feature_sets:
        spectral_feature_performance = get_feature_set_performance(
            all_features_df,
            feature_set + new_feature_set,
            config.output_folder,
            config.fdr,
            config.rescore_method,
        )
        print(
            OKCYAN_TEXT +
            f'\t\tPSMs after adding {sorted(new_feature_set)}: {spectral_feature_performance}.' +
            ENDC_TEXT
        )
        if max_features is not None:
            total_feats_after_add = len(new_feature_set) + len(feature_set)
            if total_feats_after_add >= max_features:
                break
            adjusted_performance = (max_features - total_feats_after_add)*(
                spectral_feature_performance - baseline_performance
            )
            if adjusted_performance > best_adjusted_performance:
                max_new_performance = spectral_feature_performance
                best_new_feature_set = new_feature_set
                best_adjusted_performance = adjusted_performance
        elif spectral_feature_performance >= max_new_performance:
            max_new_performance = spectral_feature_performance
            best_new_feature_set = new_feature_set

    if max_new_performance >= baseline_performance:
        baseline_performance = max_new_performance
        feature_set.extend(best_new_feature_set)

    return baseline_performance, feature_set

def remove_excluded_features(feature_list, all_features_df, exclude_features):
    """ Function to get the maximum possible number of features for Percolator q-value
        calculation when dealing with small datasets.

    Parameters
    ----------
    feature_list : list of str
        A list of perspective features.
    all_features_df : pd.DataFrame
        A DataFrame of perspective Percolator inputs.
    exclude_features : list of str
        A list of features to be ignored.

    Returns
    -------
    feature_list : list of str
        A list of perspective features without excluded features.
    """
    feature_list = [
        x for x in feature_list if (
            x not in exclude_features and all_features_df[x].nunique() > 1
        )
    ]
    return feature_list

def create_test_metric(df_row, sa_cut_off, es_cut_off):
    """ Function to create the metric on which logistic models will be trained to
        quickly test feature set performance.

    Parameters
    ----------
    df_row : pd.Series
        The row of a DataFrame in which we wish to create a testMetric.

    Returns
    -------
    df_row : pd.Series
        The updated DataFrame row with testMetric added.
    """
    if (
        df_row[LABEL_KEY] > 0 and
        df_row[SPECTRAL_ANGLE_KEY] > sa_cut_off and
        df_row[ENGINE_SCORE_KEY] > es_cut_off
    ):
        return 1
    if df_row[LABEL_KEY] > 0:
        return -1
    return 0

def get_max_features(all_features_df):
    """ Function to get the maximum possible number of features for Percolator q-value
        calculation when dealing with small datasets.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        A DataFrame of perspective Percolator inputs.

    Returns
    -------
    max_features : int or None
        The maximum number of features allowed (if None, no limit).
    """
    n_neg_samples = all_features_df[all_features_df[LABEL_KEY] == -1].shape[0]
    n_pos_samples = all_features_df.shape[0] - n_neg_samples
    min_label_samples = min((n_neg_samples, n_pos_samples))
    max_features = min_label_samples//5
    if max_features <= (all_features_df.shape[1] - 5):
        return max_features
    return None

def prepare_for_feature_selection(all_features_df):
    """ Function to create the test metric used for automated feature selection.
    """
    sa_cut_off = all_features_df[
        all_features_df[LABEL_KEY] == 1
    ][SPECTRAL_ANGLE_KEY].quantile(0.8)
    es_cut_off = all_features_df[
        all_features_df[LABEL_KEY] == 1
    ][ENGINE_SCORE_KEY].quantile(0.8)

    all_features_df['testMetric'] = all_features_df.apply(
        lambda x : create_test_metric(x, sa_cut_off, es_cut_off), axis=1
    )
    return all_features_df

def filter_feature_set(all_features_df, config, base_features):
    """ Function to filter the full Percolator feature set.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        The percolator input DataFrame with all features.
    config : inspire.config.Config
        The Config object for the experiment.
    base_features : list of str
        A list of the base features which we add on top of.

    Returns
    -------
    all_features_df : pd.DataFrame
        The input DataFrame with all features plus one hot features.
    features : list of str
        A list of selected features.
    """
    # Target/Decoy Metric taking into account high and low confidence identifications.
    all_features_df = prepare_for_feature_selection(all_features_df)

    max_features = get_max_features(all_features_df)

    feature_set = remove_excluded_features(
        base_features,
        all_features_df,
        config.exclude_features
    )

    baseline_performance = get_feature_set_performance(
        all_features_df,
        feature_set,
        config.output_folder,
        config.fdr,
        config.rescore_method,
    )
    print(
        OKCYAN_TEXT +
        f'\tEstablished baseline performance ({baseline_performance} PSMs identified).' +
        ENDC_TEXT
    )

    key_features = remove_excluded_features(
        FIRST_ADDITIONS,
        all_features_df,
        config.exclude_features
    )

    for feature in key_features:
        if all_features_df[feature].nunique() <= 1:
            continue
        new_feature_performance = get_feature_set_performance(
            all_features_df,
            feature_set + [feature],
            config.output_folder,
            config.fdr,
            config.rescore_method,
        )
        print(
            OKCYAN_TEXT +
            f'\t\tPSMs after adding {feature}: {new_feature_performance}.' +
            ENDC_TEXT
        )
        if new_feature_performance >= baseline_performance:
            baseline_performance = new_feature_performance
            feature_set.append(feature)

    print(
        OKCYAN_TEXT +
        '\tAdded key explanatory features.' +
        ENDC_TEXT
    )

    for group_name, feature_group in zip(SPECTRAL_GROUP_NAMES, SPECTRAL_FEATURE_GROUPS):
        baseline_performance, feature_set = add_feature_collection(
            all_features_df,
            feature_set,
            feature_group,
            baseline_performance,
            config,
            max_features=max_features,
        )
        print(
            OKCYAN_TEXT +
            f'\tAdded {group_name}.' +
            ENDC_TEXT
        )

    conf_feats = OUTSTANDING_CONFOUNDING_VARIABLES
    conf_feats = remove_excluded_features(
        OUTSTANDING_CONFOUNDING_VARIABLES,
        all_features_df,
        config.exclude_features,
    )

    for feature in conf_feats:
        if all_features_df[feature].nunique() <= 1:
            continue
        if len(feature) == max_features:
            break

        new_feature_performance = get_feature_set_performance(
            all_features_df,
            feature_set + [feature],
            config.output_folder,
            config.fdr,
            config.rescore_method,
        )
        print(
            OKCYAN_TEXT +
            f'\t\tPSMs after adding {feature}: {new_feature_performance}.' +
            ENDC_TEXT
        )

        if feature == SOURCE_INDEX_KEY and max_features is None:
            all_features_df, one_hot_features = generate_one_hot_entries(
                all_features_df,
                feature
            )
            if len(one_hot_features) > 2:
                one_hot_performance = get_feature_set_performance(
                    all_features_df,
                    feature_set + one_hot_features,
                    config.output_folder,
                    config.fdr,
                    config.rescore_method,
                )
            else:
                one_hot_performance = 0

            if (
                one_hot_performance > new_feature_performance and
                one_hot_performance >= baseline_performance
            ):
                baseline_performance = one_hot_performance
                feature_set.extend(one_hot_features)
                continue

        if new_feature_performance >= baseline_performance:
            baseline_performance = new_feature_performance
            feature_set.append(feature)

    return all_features_df, feature_set


def select_features(config):
    """ Function to select the features used in the final percolator model.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    all_features_df = pd.read_csv(
        f'{config.output_folder}/input_all_features.tab',
        sep='\t'
    )

    if config.max_for_selection < all_features_df.shape[0]:
        feature_set = DEFAULT_FEATURE_SET
        if config.delta_method == 'ignore':
            feature_set = [x for x in feature_set if x not in DELTA_FEATURES]
        if config.use_binding_affinity == 'asFeature':
            feature_set += ['bindingAffinity']
        # all_features_df, one_hot_features = generate_one_hot_entries(
        #     all_features_df,
        #     'charge'
        # )
        # feature_set += one_hot_features
        # feature_set.remove('charge')
        if config.exclude_features is not None and config.exclude_features:
            exclude_features = (
                config.exclude_features +
                SUFFIX_KEYS[config.rescore_method] +
                PREFIX_KEYS[config.rescore_method]
            )
        elif config.include_features is not None:
            exclude_features = [
                col for col in all_features_df.columns # pylint: disable=no-member
                if col not in config.include_features
            ]
            if config.use_accession_stratum:
                exclude_features = [x for x in exclude_features if not x.startswith('accession')]
        else:
            exclude_features = (
                SUFFIX_KEYS[config.rescore_method] + PREFIX_KEYS[config.rescore_method]
            )

        feature_set = remove_excluded_features(
            feature_set, all_features_df, exclude_features
        )
    else:
        base_features = BASE_FEATURES
        if config.use_accession_stratum:
            base_features += [
                col for col in all_features_df.columns # pylint: disable=no-member
                if col.startswith('accession')
            ]

        all_features_df, feature_set = filter_feature_set(
            all_features_df, config, base_features
        )

    write_final_feature_set(all_features_df, feature_set, config)

def write_final_feature_set(all_features_df, feature_set, config):
    """ Function to write the final selected features for Percolator input.

    Parameters
    ----------
    all_features_df : pd.DataFrame
        DataFrame containing all possible percolator input features.
    feature_set : list of str
        A list of the feature names to be used.
    config : inspire.config.Config
        The Config object
    """
    with open(f'{config.output_folder}/selectedFeatures.yaml', 'w', encoding='UTF-8') as file:
        yaml.dump(feature_set, file)

    prefix_keys = PREFIX_KEYS[config.rescore_method]

    all_features_df = all_features_df[
        prefix_keys + feature_set + SUFFIX_KEYS[config.rescore_method]
    ]

    all_features_df.to_csv(
        f'{config.output_folder}/final_input.tab',
        sep='\t',
        index=False
    )
    print(
        OKCYAN_TEXT +
        '\tFinal Feature Set Written.' +
        ENDC_TEXT
    )
