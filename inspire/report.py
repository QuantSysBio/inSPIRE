""" Functions for providing a report at the end of inSPIRE rescoring.
"""
import pickle
import pandas as pd
import yaml

from inspire.constants import (
    CHARGE_KEY,
    DELTA_SCORE_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    FINAL_Q_VALUE_KEY,
    FINAL_SCORE_KEY,
    MASS_DIFF_KEY,
    OKCYAN_TEXT,
    OUT_PSM_ID_KEY,
    OUT_Q_KEY,
    OUT_SCORE_KEY,
    PEPTIDE_KEY,
    PREFIX_KEYS,
    PSM_ID_KEY,
    SUFFIX_KEYS,
    PRED_ACCESSION_KEY,
    PRED_PEPTIDE_KEY,
    SEQ_LEN_KEY,
    TRUE_PEPTIDE_KEY,
)
from inspire.feature_selection import generate_one_hot_entries
from inspire.figures import (
    create_binders_fig,
    create_psms_fig,
    create_violin_fig,
    create_weights_fig,
)
from inspire.html_template import create_html_report
from inspire.input.mhcpan import read_mhcpan_output
from inspire.rescore import apply_rescoring

NON_SPECTRAL_FEATURES = [
    ENGINE_SCORE_KEY,
    DELTA_SCORE_KEY,
    MASS_DIFF_KEY,
    SEQ_LEN_KEY,
    'nVarMods',
    'avgResidueMass',
]
N_PR_STEPS = 200

def combine_results(output_folder, mhc_pan_df, input_file):
    """ Function to add NetMHCpan predictions to inSPIRE PSMs.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE output is written.
    mhc_pan_df : pd.DataFrame
        A DataFrame of NetMHCpan predictions.
    input_file : str
        The input file of rescored psms.

    Returns
    -------
    final_assignments_df : pd.DataFrame
        Combined PSMs and NetMHCpan predictions.
    """
    if 'non_spectral' in input_file:
        final_assignments_df = pd.read_csv(f'{output_folder}/{input_file}', sep='\t')
        if 'mokapot' in input_file:
            score_key = OUT_SCORE_KEY['mokapot']
        else:
            score_key = OUT_SCORE_KEY['percolator']
    else:
        final_assignments_df = pd.read_csv(f'{output_folder}/{input_file}')
        score_key = FINAL_SCORE_KEY

    mhc_pan_df = mhc_pan_df.rename(columns={'Peptide': PEPTIDE_KEY})
    mhc_pan_df = mhc_pan_df[[PEPTIDE_KEY, '%Rank_BA', 'BindLevel']]
    final_assignments_df = pd.merge(
        final_assignments_df,
        mhc_pan_df,
        on=PEPTIDE_KEY,
        how='left',
    )

    final_assignments_df = final_assignments_df.sort_values(
        by=score_key,
        ascending=False
    )

    return final_assignments_df

def apply_non_spectral_percolator(output_folder, fdr, rescore_method, use_score_only=False):
    """ Function to apply percolator without spectral features as a comparison with
        full inSPIRE.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE output is written.
    fdr : float
        The false discovery rate we are aiming.
    use_score_only : bool
        Flag indicating whether or not to use the engine score only.

    Returns
    -------
    non_spectral_psm_df : pd.DataFrame
        A DataFrame of results from Percolator trained without spectral features.
    """
    all_features_df = pd.read_csv(f'{output_folder}/input_all_features.tab', sep='\t')

    prefix_keys = PREFIX_KEYS[rescore_method]
    psm_id_key = PSM_ID_KEY[rescore_method]

    all_features_df, one_hot_features = generate_one_hot_entries(
        all_features_df,
        CHARGE_KEY
    )

    if use_score_only:
        non_spectral_df = all_features_df[
            prefix_keys +
            [ENGINE_SCORE_KEY] +
            SUFFIX_KEYS
        ]
    else:
        non_spectral_df = all_features_df[
            prefix_keys + NON_SPECTRAL_FEATURES + one_hot_features + SUFFIX_KEYS[rescore_method]
        ]

    non_spectral_df = non_spectral_df.dropna(axis=1)
    non_spectral_df.to_csv(
        f'{output_folder}/non_spectral_perc_input.tab',
        index=False,
        sep='\t',
    )

    non_spectral_psm_df = apply_rescoring(
        output_folder,
        'non_spectral_perc_input.tab',
        fdr,
        rescore_method,
        'non_spectral'
    ).rename(columns={OUT_PSM_ID_KEY[rescore_method]: psm_id_key})

    non_spectral_psm_df = pd.merge(
        non_spectral_psm_df,
        non_spectral_df[[psm_id_key, ENGINE_SCORE_KEY]],
        how='inner',
        on=psm_id_key,
    )

    return non_spectral_psm_df

def get_stats_at_cut_off(assignment_df, binders_df, q_cut, q_val_key):
    """ Function to calculate statistics on the PSMs above a given cut off.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        A DataFrame of PSM assignments.
    binders_df : pd.DataFrame
        A DataFrame of the predicted MHC binding of the PSMs.
    q_cut : float
        The q-value cut off to apply.
    q_val_key : str
        The name of the column containing q-values.

    Returns
    -------
    psm_count : int
        The number of psms above the threshold.
    binders_count : int
        The number of predicted binders above the threshold.
    pct_binders : float
        The percentage of predicted binders among qualifying PSMs.
    """
    psm_count = assignment_df[assignment_df[q_val_key] <= q_cut].shape[0]
    if binders_df is not None:
        binders_cut_df = binders_df[binders_df[q_val_key] <= q_cut]
        binders_divisor = binders_cut_df.shape[0]
        binders_count = binders_cut_df[
            (binders_cut_df['BindLevel'] == '<=WB') | (binders_cut_df['BindLevel'] == '<=SB')
        ].shape[0]
        pct_binders = 100*binders_count/binders_divisor
    else:
        binders_count = None
        pct_binders = None
    return psm_count, binders_count, pct_binders

def calculate_fdr_cut_offs(
        assignment_df,
        non_spectral_df,
        se_q_cut_key,
        binders_df=None,
        ns_binders_df=None
    ):
    """ Function calculate statistics on PSMs at different false discovery rates.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        A DataFrame of inSPIRE final assignments.
    non_spectral_df : pd.DataFrame
        A DataFrame of assignments from Percolator without spectral features.
    rescore_method : str
        Indictor on whether mokapot or percolator is being used.
    binders_df : pd.DataFrame or None
        A DataFrame of NetMHCpan binders for inSPIRE assignments.
    ns_binders_df : pd.DataFrame or None
        A DataFrame of NetMHCpan binders for Percolator assignments without spectral features.

    Returns
    -------
    fdrs_df : pd.DataFrame
        A DataFrame of statistics on PSMs at various FDR thresholds.
    """
    q_cut_offs = [0.01*i for i in range(1, 11)]
    n_spire_psms = []
    n_search_engine_psms = []
    n_spire_binders = []
    n_search_engine_binders = []
    pct_spire_binders = []
    pct_search_engine_binders = []

    for q_cut in q_cut_offs:
        car_stats = get_stats_at_cut_off(
            assignment_df,
            binders_df,
            q_cut,
            FINAL_Q_VALUE_KEY,
        )
        n_spire_psms.append(car_stats[0])
        n_spire_binders.append(car_stats[1])
        pct_spire_binders.append(car_stats[2])

        se_stats = get_stats_at_cut_off(
            non_spectral_df,
            ns_binders_df,
            q_cut,
            se_q_cut_key,
        )
        n_search_engine_psms.append(se_stats[0])
        n_search_engine_binders.append(se_stats[1])
        pct_search_engine_binders.append(se_stats[2])

    return pd.DataFrame(
        {
            'FDR': q_cut_offs,
            'nSpirePsms': n_spire_psms,
            'nSpireBinders': n_spire_binders,
            'SpirePercentageBinders': pct_spire_binders,
            'nSearchEnginePsms': n_search_engine_psms,
            'nSearchEngineBinders': n_search_engine_binders,
            'searchEnginePercentageBinders': pct_search_engine_binders,
        }
    )

def _get_counts(all_df, score_cut_off, score_key, acc_grp):
    """ Function to get the predicted and correct counts above a scoring threshold.

    Parameters
    ----------
    all_df : pd.DataFrame
        A DataFrame of all psms.
    score_cut_off : float
        A cut off on psms percolator score.
    score_key : str
        The column name of the percolator scores.
    acc_grp : str
        The accession group we are interested in.

    Returns
    -------
    pred_count : int
        The number of PSMs for the accession group above the threshold.
    correct_count : int
        The number of correct PSMs for the accession group above the threshold.
    """
    if acc_grp != 'discoverable':
        filtered_df = all_df[
            (all_df[score_key] >= score_cut_off) &
            (all_df[PRED_ACCESSION_KEY] == acc_grp)
        ]
    else:
        filtered_df = all_df[
            all_df[score_key] >= score_cut_off
        ]

    pred_count = filtered_df.shape[0]

    if score_key.startswith('prosit'):
        pep_key = 'prositOverPeptide'
    else:
        pep_key = PRED_PEPTIDE_KEY

    correct_count = filtered_df[filtered_df.apply(
        lambda x : x[TRUE_PEPTIDE_KEY].replace('I', 'L') == x[pep_key].replace('I', 'L'),
        axis=1
    )].shape[0]

    return pred_count, correct_count


def generate_report(config):
    """ Function for creating a report at the end of ininspire execution.

    Parameters
    ----------
    config : inspire.config.Config
        The settings for the whole pipeline.
    """
    figures = {}
    figures['table'], most_pos_feats, most_neg_feats = create_weights_table(
        config.output_folder,
        config.rescore_method,
    )
    figures['violin_fig'] = calculate_distributions(
        config,
        most_pos_feats,
        most_neg_feats
    )

    if config.rescore_method == 'mokapot':
        non_spectral_fdr = 0.05
    else:
        non_spectral_fdr = config.fdr

    non_spectral_df = apply_non_spectral_percolator(
        config.output_folder,
        non_spectral_fdr,
        config.rescore_method,
    )

    assignment_df = pd.read_csv(f'{config.output_folder}/finalAssignments.csv')

    if config.use_binding_affinity in ['asFeature', 'asValidation']:
        mhcpan_df = read_mhcpan_output(f'{config.output_folder}/mhcpan')
        binders_df = combine_results(config.output_folder, mhcpan_df, 'finalAssignments.csv')
        ns_binders_df = combine_results(
            config.output_folder,
            mhcpan_df,
            f'non_spectral.{config.rescore_method}.psms.txt'
        )
        binders_df.to_csv(f'{config.output_folder}/binderFinalAssignments.csv')

        fdrs_df = calculate_fdr_cut_offs(
            assignment_df,
            non_spectral_df,
            OUT_Q_KEY[config.rescore_method],
            binders_df,
            ns_binders_df
        )
    else:
        fdrs_df = calculate_fdr_cut_offs(
            assignment_df,
            non_spectral_df,
            OUT_Q_KEY[config.rescore_method],
        )

    if config.use_binding_affinity:
        figures['binders_fig'] = create_binders_fig(fdrs_df, config.search_engine)

    figures['psms_fig'] = create_psms_fig(fdrs_df, config.search_engine)
    fdrs_df.to_csv(f'{config.output_folder}/searchEngineFdrs.csv')
    print(
        OKCYAN_TEXT +
        '\tAll Figures Created.' +
        ENDC_TEXT
    )

    create_html_report(
        config,
        figures,
    )


def calculate_distributions(config, most_positive_features, most_negative_features):
    """ Function to produce violin plots.
    """
    out_score_key = OUT_SCORE_KEY[config.rescore_method]
    psm_id_key = PSM_ID_KEY[config.rescore_method]

    out_filename = f'final.{config.rescore_method}.psms.txt'
    input_df = pd.read_csv(f'{config.output_folder}/final_input.tab', sep='\t')
    psms_df = pd.read_csv(f'{config.output_folder}/{out_filename}', sep='\t').rename(
        columns={OUT_PSM_ID_KEY[config.rescore_method]: psm_id_key}
    )

    psms_df['Status'] = psms_df[out_score_key].apply(
        lambda x : 'Accepted' if x >= 0 else 'Rejected'
    )

    combined_df = pd.merge(
        input_df,
        psms_df[[psm_id_key, 'Status', OUT_SCORE_KEY[config.rescore_method]]],
        how='inner',
        on=psm_id_key,
    )

    violin_plot = create_violin_fig(combined_df, most_positive_features, most_negative_features)

    return violin_plot

def add_mokapot_weights(output_folder, rescore_method, acc_idx):
    """ Function to create a Figure of weights table.

    Parameters
    ----------
    output_folder : str
        The folder where all inspire output is written.
    rescore_method : str
        Indicator of whether mokapot or percolator is in use.

    Results
    -------
    weights_df : pd.DataFrame
        A DataFrame of the weights for different features.
    """
    with open(f'{output_folder}/selectedFeatures.yaml', 'r', encoding='UTF-8') as stream:
        feature_names = yaml.safe_load(stream)

    if acc_idx is None:
        prefix = 'final'
    else:
        prefix = f'final_{acc_idx}'

    feat_weights = {feature: [] for feature in feature_names}
    feat_weights['intercept'] = []
    for fold_idx in range(1, 4):
        with open(
            f'{output_folder}/{prefix}.{rescore_method}.model_fold-{fold_idx}.pkl',
            'rb',
        ) as model_file:
            model = pickle.load(model_file)
        weights = model.estimator.coef_[0]
        for feat_idx, name in enumerate(feature_names):
            feat_weights[name].append(round(weights[feat_idx], 2))
        feat_weights['intercept'].append(
            round(model.estimator.intercept_[0], 2)
        )

    return pd.DataFrame(feat_weights)

def clean_percolator_weights(weights_path):
    """ Function to remove the comments which are inexplicably added to a csv
        file.
    """
    with open(weights_path, 'r', encoding='UTF-8') as file :
        lines = file.readlines()
        relevant_lines = []
        for line in lines:
            if not line.startswith('#'):
                relevant_lines.append(line)

    with open(weights_path, 'w', encoding='UTF-8') as file:
        for line in relevant_lines:
            file.write(line)

def create_weights_table(output_folder, rescore_method, acc_idx=None):
    """ Function to create a Figure of weights table.

    Parameters
    ----------
    output_folder : str
        The folder where all inspire output is written.
    rescore_method : str
        Indicator of whether mokapot or percolator is in use.

    Returns
    -------
    fig : str
        A Figure showing weights converted to html.
    most_pos_feats : list of str
        A list of the most positively weighted features.
    most_neg_feats : list of str
        A list of the most negatively weighted features.
    """
    if rescore_method == 'mokapot':
        weights_df = add_mokapot_weights(output_folder, rescore_method, acc_idx)
    else:
        if acc_idx is None:
            weights_path = f'{output_folder}/final.{rescore_method}.weights.csv'
        else:
            weights_path = f'{output_folder}/final_{acc_idx}.{rescore_method}.weights.csv'
        clean_percolator_weights(weights_path)

        weights_df = pd.read_csv(
            weights_path,
            skiprows=lambda x : x not in [0, 1, 4, 7],
            sep='\t',
        )
    transposed_df = weights_df.transpose()

    weights_df_means = weights_df.mean()
    weight_keys = [f'weightFold{i}' for i in range(1, 4)]
    transposed_df.columns = weight_keys
    transposed_df['feature'] = transposed_df.index
    transposed_df['averageWeight'] = weights_df_means
    transposed_df = transposed_df.sort_values(by='averageWeight', ascending=False)
    m0_df = transposed_df[
        (transposed_df.index == 'm0') | (transposed_df.index == 'intercept')
    ]
    transposed_df = transposed_df[
        (transposed_df.index != 'm0') & (transposed_df.index != 'intercept')
    ]
    transposed_df = pd.concat([transposed_df, m0_df])

    weights_and_feats = sorted([
        (ave_wt, feat) for ave_wt, feat in zip(
            transposed_df.averageWeight.tolist(),
            transposed_df.feature.tolist()
        ) if feat != 'm0' and feat != 'intercept' and '_' not in feat
    ])
    most_neg_feats = [x[1] for x in weights_and_feats[:3]]
    most_pos_feats = [x[1] for x in weights_and_feats[-3:]]
    most_pos_feats.reverse()

    fig = create_weights_fig(transposed_df)
    return fig, most_pos_feats, most_neg_feats
