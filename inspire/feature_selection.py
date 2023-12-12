""" Functions for running automated feature selection.
"""
from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
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
    MINIMAL_FEATURE_SET,
    NOT_ASSIGNED_KEY,
    OKCYAN_TEXT,
    PEARSON_KEY,
    PEPTIDE_KEY,
    PRECURSOR_INTE_KEY,
    PREFIX_KEYS,
    SPEARMAN_KEY,
    SUFFIX_KEYS,
    SEQ_LEN_KEY,
    SOURCE_INDEX_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)
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
    ENGINE_SCORE_KEY,
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


def convert_to_irt(all_features_df, config):
    """ Function to convert deltaRT in seconds to iRT value.
    """
    all_features_df[SOURCE_KEY] = all_features_df['specID'].apply(
        lambda x : '_'.join(x.split('_')[:-2])
    )
    sources = all_features_df[SOURCE_KEY].unique().tolist()
    irt_coeffs = {}
    for source in sources:
        try:
            irt_df = pd.read_csv(f'{config.output_folder}/rt_fit_{source}.csv')
            irt_coeffs[source] = irt_df['coefficents'].mean()
            if irt_coeffs[source] <= 0:
                irt_coeffs[source] = 1.0
        except FileNotFoundError:
            print(f'No file found for {source}')
            irt_coeffs[source] = 1.0

    all_features_df['deltaRT'] = all_features_df[['deltaRT', 'source']].apply(
        lambda df_row : abs(
            df_row['deltaRT']/irt_coeffs.get(df_row['source'], 1.0)
        ),
        axis=1,
    )
    return all_features_df


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

    if config.use_irt_diff:
        all_features_df = convert_to_irt(all_features_df, config)

    if config.minimal_features:
        feature_set = MINIMAL_FEATURE_SET
    else:
        feature_set = DEFAULT_FEATURE_SET
        if config.delta_method == 'ignore':
            feature_set = [x for x in feature_set if x not in DELTA_FEATURES]

    if config.use_binding_affinity == 'asFeature':
        feature_set += ['bindingAffinity']

    if config.use_accession_stratum:
        feature_set += [
            col for col in all_features_df.columns # pylint: disable=no-member
            if col.startswith('accession_')
        ]

    if config.exclude_features is not None and config.exclude_features:
        exclude_features = (
            config.exclude_features +
            SUFFIX_KEYS[config.rescore_method] +
            PREFIX_KEYS[config.rescore_method]
        )
    elif config.include_features is not None:
        feature_set = list(set(feature_set + config.include_features))
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

    all_features_df[PEPTIDE_KEY] = '-.' +all_features_df[PEPTIDE_KEY].astype(str) + '.-'
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
