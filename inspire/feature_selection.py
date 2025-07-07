""" Functions for running automated feature selection.
"""
from pathlib import Path
import pandas as pd
import xgboost as xgb
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
    PISCES_BA_FEATURE_SETS,
    PISCES_CASA_BA_FEATURE_SETS,
    PISCES_CASA_NOBA_FEATURE_SETS,
    PISCES_NOBA_FEATURE_SETS,
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

PISCES_MODEL_PATH = (
    '{home}/inSPIRE_models/pisces_models/{setting}/{dn_method}/model{model_step}/clf{model_idx}_all.json'
)

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


def get_pisces_scores(all_features_df, use_binding_affinity, enzyme, search_engine):
    """ Function to get pisces scores
    """
    engine_scored_df = all_features_df[all_features_df['engineScore'] > -1]
    unscored_df = all_features_df[all_features_df['engineScore'] == -1]
    if enzyme == 'trypsin':
        setting='trypsin'
        es_idx = 0
        noes_idx = 1
        if search_engine == 'peaksDeNovo':
            feature_sets = PISCES_NOBA_FEATURE_SETS
        else:
            feature_sets = PISCES_CASA_NOBA_FEATURE_SETS
    else:
        setting = 'ip'
        if use_binding_affinity == 'asFeature':
            if search_engine == 'peaksDeNovo':
                feature_sets = PISCES_BA_FEATURE_SETS
            else:
                feature_sets = PISCES_CASA_BA_FEATURE_SETS
            es_idx = 0
            noes_idx = 1
        else:
            if search_engine == 'peaksDeNovo':
                feature_sets = PISCES_NOBA_FEATURE_SETS
            else:
                feature_sets = PISCES_CASA_NOBA_FEATURE_SETS
            es_idx = 2
            noes_idx = 3

    scored_dfs = []
    home = str(Path.home())

    if engine_scored_df.shape[0]:
        clf = xgb.XGBClassifier() # or which ever sklearn booster you're are using
        clf.load_model(PISCES_MODEL_PATH.format(
            home=home,
            setting=setting,
            dn_method=search_engine,
            model_step=1,
            model_idx=es_idx,
        ))

        engine_scored_df['piscesScore1'] = clf.predict_proba(engine_scored_df[
            feature_sets[0]
        ])[:,1]
        engine_scored_df['modelUsed'] = 0
        scored_dfs.append(engine_scored_df)

    if unscored_df.shape[0]:
        clf = xgb.XGBClassifier() # or which ever sklearn booster you're are using
        clf.load_model(PISCES_MODEL_PATH.format(
            home=home,
            setting=setting,
            dn_method=search_engine,
            model_step=1,
            model_idx=noes_idx,
        ))

        unscored_df['piscesScore1'] = clf.predict_proba(unscored_df[
            feature_sets[1]
        ])[:,1]
        unscored_df['modelUsed'] = 1
        scored_dfs.append(unscored_df)

    return pd.concat(scored_dfs)


def convert_to_irt(all_features_df, config):
    """ Function to convert deltaRT in seconds to iRT value.
    """
    all_features_df[SOURCE_KEY] = all_features_df['specID'].map_elements(
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

    all_features_df['deltaRT'] = all_features_df[['deltaRT', 'source']].map_elements(
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

    if config.for_pisces:
        if config.search_engine != 'psms':
            search_eng_arg = config.search_engine
        else:
            search_eng_arg = config.pisces_dn_method

        all_features_df = get_pisces_scores(
            all_features_df, config.use_binding_affinity, config.enzyme, search_eng_arg,
        )
        feature_set = ['modelUsed', 'piscesScore1']
    elif config.minimal_features:
        feature_set = MINIMAL_FEATURE_SET
    else:
        feature_set = DEFAULT_FEATURE_SET
        if config.delta_method == 'ignore':
            feature_set = [x for x in feature_set if x not in DELTA_FEATURES]

    if config.use_binding_affinity == 'asFeature' and not config.for_pisces:
        feature_set += ['mhcpanPrediction', 'nuggetsPrediction']

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
