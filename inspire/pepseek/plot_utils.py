""" Script containing functions for writing plotting utils of PEPSeek Epitopes results.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA

from inspire.constants import ENGINE_SCORE_KEY, PLOT_AXIS_REQUIREMENTS, OUT_POSTEP_KEY
from inspire.logo_plot_utils import create_comparison_logo_plot
from inspire.utils import fetch_proteome

def _check_pep(peptide, proteome):
    if '.' in peptide:
        peptide = peptide.split('.')[1]
    for prot in proteome:
        if peptide.replace('I', 'L') in prot[1].replace('I', 'L'):
            return True
    return False

def _check_prot(proteins, proteome):
    proteins = proteins.split(' ')
    for prot in proteome:
        for found_prot in proteins:
            if found_prot == prot[0]:
                return True
    return False

def get_host_peps(config):
    if config.epitope_cut_level == 'psm':
        ep_df = pd.read_csv(f'{config.output_folder}/finalPsmAssignments.csv')
    else:
        ep_df = pd.read_csv(f'{config.output_folder}/finalPeptideAssignments.csv')

    host_prot = fetch_proteome(config.host_proteome, with_desc=False)
    ep_df = ep_df[(ep_df['qValue'] < 0.01) & (ep_df['postErrProb'] < 0.1)]
    ep_df = ep_df[
        (ep_df['peptide'].apply(len) <= 15) &
        (ep_df['peptide'].apply(len) >= 8)
    ]

    ep_df = ep_df.drop_duplicates(subset=['peptide'])
    ep_df = ep_df[ep_df['peptide'].apply(lambda x : _check_pep(x, host_prot))]

    if config.epitope_cut_level == 'psm':
        host_basic_df = pd.read_csv(
            f'{config.output_folder}/non_spectral.{config.rescore_method}.psms.txt', sep='\t',
        )
    else:
        host_basic_df = pd.read_csv(
            f'{config.output_folder}/non_spectral.{config.rescore_method}.peptides.txt', sep='\t',
        )

    host_basic_df = host_basic_df[
        (host_basic_df['q-value'] < 0.01) & (host_basic_df['posterior_error_prob'] < 0.1)
    ]
    host_basic_df = host_basic_df.drop_duplicates(subset=['peptide'])
    host_basic_df = host_basic_df[
        host_basic_df['peptide'].apply(lambda x : _check_pep(x, host_prot))
    ]
    host_basic_df['peptide'] = host_basic_df['peptide'].apply(
        lambda x : x.split('.')[1] if '.' in x else x
    )
    ep_df = pd.merge(ep_df, host_basic_df[['peptide']], how='left', on='peptide', indicator=True)
    ep_df['foundBySearchEngine'] = ep_df['_merge'].apply(lambda x : 'Yes' if x == 'both' else 'No')

    return ep_df

def bar_plot(config, host=False):
    """ Function generate bar plots of shared and PEPseek only identification counts.
    """
    if host:
        host_prot = fetch_proteome(config.host_proteome, with_desc=False)
        host_insp_df = pd.read_csv(
            f'{config.output_folder}/final.{config.rescore_method}.peptides.txt', sep='\t',
        )
        host_insp_df = host_insp_df[
            (host_insp_df['q-value'] < 0.01) & (host_insp_df['posterior_error_prob'] < 0.1)
        ]
        host_basic_df = pd.read_csv(
            f'{config.output_folder}/non_spectral.{config.rescore_method}.peptides.txt', sep='\t',
        )
        host_basic_df = host_basic_df[
            (host_basic_df['q-value'] < 0.01) & (host_basic_df['posterior_error_prob'] < 0.1)
        ]
        for name in ['peptides', 'proteins']:
            if name == 'peptides':
                joined_df = pd.merge(
                    host_insp_df[['peptide']].drop_duplicates(),
                    host_basic_df[['peptide']].drop_duplicates(),
                    how='outer', indicator=True,
                )
                joined_df = joined_df[joined_df['peptide'].apply(lambda x : _check_pep(x, host_prot))]
                insp_pep_count = joined_df[joined_df['_merge'] == 'left_only'].shape[0]
                shared_pep_count = joined_df[joined_df['_merge'] == 'both'].shape[0]
            else:
                joined_df = pd.merge(
                    host_insp_df[['proteinIds']].drop_duplicates(),
                    host_basic_df[['proteinIds']].drop_duplicates(),
                    how='outer', indicator=True,
                )
                joined_df = joined_df[joined_df['proteinIds'].apply(lambda x : _check_prot(x, host_prot))]
                insp_prot_count = joined_df[joined_df['_merge'] == 'left_only'].shape[0]
                shared_prot_count = joined_df[joined_df['_merge'] == 'both'].shape[0]
    else:
        ep_df = pd.read_csv(f'{config.output_folder}/PEPSeek/potentialEpitopeCandidates.csv')
        insp_df = ep_df[ep_df['foundBySearchEngine'] == 'No']
        shared_df = ep_df[ep_df['foundBySearchEngine'] == 'Yes']

        insp_prot_df = insp_df.drop_duplicates(subset='protein')[['protein']]
        shared_prot_df = shared_df.drop_duplicates(subset='protein')[['protein']]
        shared_pep_count = shared_df.shape[0]
        insp_pep_count = insp_df.shape[0]
        total_df = pd.merge(shared_prot_df, insp_prot_df, how='outer', on='protein', indicator=True)
        insp_prot_count = total_df[total_df['_merge'] == 'right_only'].shape[0]
        shared_prot_count = total_df[total_df['_merge'] != 'right_only'].shape[0]

    fig = px.bar(
        color=['Shared', 'Shared', 'PEPSeek Only', 'PEPSeek Only'],
        y=[shared_pep_count,shared_prot_count,insp_pep_count,insp_prot_count],
        x=['Epitope', 'Antigen', 'Epitope', 'Antigen',],
        color_discrete_map={
            'PEPSeek Only': '#FFBE00',
            'Shared': '#C0C0C0',
        },
    )
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_xaxes(PLOT_AXIS_REQUIREMENTS, dtick=1, title='')
    fig.update_yaxes(PLOT_AXIS_REQUIREMENTS, title='Number of Epitopes/Antigens',)
    fig.update_layout(
        barmode='group',
        font_family='Helvetica',
        font_color='black',
        width=500,
        height=300,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_showticklabels=True,
    )
    code = ''
    if host:
        code = '_host'
    fig.write_image(
        f'{config.output_folder}/img/PEPSeek{code}_bar_plot.svg', engine='kaleido'
    )


def check_engine_results(pathogen_df, config):
    """ Function to check if identified PSMs were found using the original search engine.

    Parameters
    ----------
    pathogen_df : pd.DataFrame
        DataFrame of identified pathogen peptides.
    config : inspire.config.Config
        The Config object which controls the experiment.

    Returns
    -------
    pathogen_df : pd.DataFrame
        The input DataFrame with information added on the original search engine identifications.
    """
    pathogen_df['maxEngineScore'] = pathogen_df.groupby(
        'peptide'
    )[ENGINE_SCORE_KEY].transform('max')

    if config.engine_score_cut is not None:
        pathogen_df['foundBySearchEngine'] = pathogen_df['maxEngineScore'].apply(
            lambda x : 'Yes' if x > config.engine_score_cut else 'No'
        )
    else:
        if config.epitope_cut_level == 'psm':
            non_spect_df = pd.read_csv(
                f'{config.output_folder}/non_spectral.{config.rescore_method}.psms.txt',
                sep='\t',
            )
        else:
            non_spect_df = pd.read_csv(
                f'{config.output_folder}/non_spectral.{config.rescore_method}.peptides.txt',
                sep='\t',
            )
        non_spect_df = non_spect_df[
            (non_spect_df[OUT_POSTEP_KEY[config.rescore_method]] < config.epitope_candidate_cut_off)
        ]
        non_spect_df = non_spect_df.drop_duplicates(subset='peptide')
        non_spect_df = non_spect_df[['peptide']]
        non_spect_df['peptide'] = non_spect_df['peptide'].apply(
            lambda peptide : peptide.split('.')[1] if '.' in peptide else peptide
        )
        non_spect_df['foundBySearchEngine'] = 'Yes'

        pathogen_df = pd.merge(pathogen_df, non_spect_df, how='left', on='peptide')
        pathogen_df['foundBySearchEngine'] = pathogen_df['foundBySearchEngine'].apply(
            lambda x : x if x == 'Yes' else 'No'
        )

    pathogen_df = pathogen_df.drop('maxEngineScore', axis=1)

    return pathogen_df

COLOUR_DICTS = {
    'No': '#FFBE00',
    'Yes': 'darkgrey',
}

def swarm_plots(config, host=False):
    """ Function to provide bee swarm plots of various metrics for shared and PEPSeek
        only identifications. 
    """
    if host:
        ep_df = get_host_peps(config)
    else:
        ep_df = pd.read_csv(f'{config.output_folder}/PEPSeek/potentialEpitopeCandidates.csv')
    fig = make_subplots(rows=1, cols=3)
    range_cuts = []
    for met_idx, metric in enumerate(['spectralAngle', 'engineScore', 'deltaRT']):
        if host:
            for found_val in ['No', 'Yes']:
                sub_df = ep_df[ep_df['foundBySearchEngine'] == found_val]
                fig.add_trace(go.Violin(
                    fillcolor=COLOUR_DICTS[found_val],
                    y=sub_df[metric],
                    x=sub_df[
                        'foundBySearchEngine'
                    ].str.replace('Yes', 'Shared').replace('No', 'PEPSeek Only'),
                    line_color='black', line_width=0.5,
                    points=False,
                    opacity=0.8,
                    meanline_visible=True,
                ), row=1, col=1+met_idx)
        else:
            mini_fig = px.strip(
                color=ep_df['foundBySearchEngine'],
                y=ep_df[metric],
                x=ep_df[
                    'foundBySearchEngine'
                ].str.replace('Yes', 'Shared').replace('No', 'PEPSeek Only'),
                color_discrete_map={
                    'No': '#FFBE00',
                    'Yes': 'darkgrey',
                },
            )
            for trace in mini_fig['data']:
                fig.add_trace(trace, row=1, col=1+met_idx)
        range_cuts.append(
            10 + 10*(ep_df[metric].max()//10)
        )

    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_xaxes(PLOT_AXIS_REQUIREMENTS, dtick=1)
    fig.update_yaxes(PLOT_AXIS_REQUIREMENTS)
    fig.update_layout(
        {
            'yaxis1': {'title': 'Spectral Angle', 'range':[0,1]},
            'yaxis2': {'title': 'Search Engine Score', 'range': [0,range_cuts[1]]},
            'yaxis3': {'title': 'Retention Time Error', 'range': [0,range_cuts[2]]},
        },
        barmode='group',
        font_family='Helvetica',
        font_color='black',
        width=900,
        height=400,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_showticklabels=True,
        showlegend=False,
    )
    code = ''
    if host:
        code = '_host'
    fig.write_image(
        f'{config.output_folder}/img/PEPSeek{code}_metrics.svg', engine='kaleido'
    )


def plot_binding_clustermap(config, host=False):
    """ Function to plot clustermap of peptides and their binding affinities.
    """
    if host:
        ep_df = get_host_peps(config)
    else:
        ep_df = pd.read_csv(f'{config.output_folder}/PEPSeek/potentialEpitopeCandidates.csv')

    ep_df.index = ep_df['peptide']
    binding_aff_cols = [col for col in ep_df.columns if col.endswith('%Rank_BA')]
    if len(binding_aff_cols) < 2:
        return
    ep_df = ep_df[binding_aff_cols]
    ep_df = ep_df.rename(
        columns={col:col.strip('_%Rank_BA') for col in ep_df.columns}
    )
    ep_df = ep_df.map(lambda x : round(x, 2))
    ep_df = ep_df.sort_values(by=list(ep_df.columns))

    if host:
        cluster_map = sns.clustermap(
            ep_df, cmap=sns.color_palette("blend:#FA3F37,#FAF4D3", as_cmap=True),
            yticklabels=False, figsize=(7,10), vmin=0, vmax=20,)
    else:
        fig_length = 1+(ep_df.shape[0]//6)

        cluster_map = sns.clustermap(
            ep_df, cmap=sns.color_palette("blend:#FA3F37,#FAF4D3", as_cmap=True),
            yticklabels=True, figsize=(7,fig_length), vmin=0, vmax=20,
            linewidths=0.5, linecolor='black')

    code = ''
    if host:
        code = '_host'
    cluster_map.figure.savefig(f'{config.output_folder}/img/PEPSeek{code}_affinity_cluster.svg')

def plot_quant_pca(config):
    """ Function to plot PCA first two components of quantification across files for pathogen
        peptides and a random selection of host peptides.
    """
    if not os.path.exists(f'{config.output_folder}/quant/quantified_per_file.csv'):
        return
    ep_df = pd.read_csv(f'{config.output_folder}/PEPSeek/potentialEpitopeCandidates.csv')
    quant_df = pd.read_csv(f'{config.output_folder}/quant/quantified_per_file.csv')
    quant_idp_cols = [col for col in quant_df.columns if col.endswith('_idp')]
    quant_raw_cols = [col for col in quant_df.columns if col.endswith('_raw')]
    quant_df = quant_df[['peptide'] + quant_raw_cols + quant_idp_cols]
    z_score_df = pd.DataFrame(
        zscore(quant_df[quant_raw_cols], axis=1, nan_policy='omit'),
        index=quant_df.index,
        columns=quant_raw_cols,
    )
    quant_df = quant_df.drop(quant_raw_cols, axis=1)
    quant_df = pd.merge(quant_df, z_score_df, left_index=True, right_index=True)

    merged_df = pd.merge(ep_df[['peptide']], quant_df, how='inner', on='peptide')
    host_samples = 100*(len(merged_df)//100) + 100

    quant_df = pd.merge(quant_df, merged_df[['peptide']], how='left', on='peptide', indicator=True)
    quant_df = quant_df[quant_df['_merge'] == 'left_only'].drop('_merge', axis=1).sample(
        n=host_samples,
        random_state=42,
    )

    merged_df['group'] = 'Pathogen'
    quant_df['group'] = 'Host'

    total_df = pd.concat([merged_df, quant_df])
    total_df = total_df.dropna(axis=0)

    pca = PCA(n_components=2).fit(total_df[quant_raw_cols+quant_idp_cols])

    component_1_key = f'Component 1 (Exp. Var. {round(pca.explained_variance_ratio_[0], 2)})'
    component_2_key = f'Component 2 (Exp. Var. {round(pca.explained_variance_ratio_[1], 2)})'

    pca_df = pd.DataFrame(
        pca.transform(total_df[quant_raw_cols+quant_idp_cols]),
        columns=[
            component_1_key, component_2_key,
        ],
        index=total_df.index
    )

    pca_df /= pca_df.abs().max().max()
    pca_df['Accession'] = total_df['group']

    fig = px.scatter(
        pca_df,
        x=component_2_key,
        y=component_1_key,
        color='Accession',
        color_discrete_map={'Host': 'darkgreen', 'Pathogen': '#B65FCF'},
    )
    fig.update_traces(opacity=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_traces(marker_size=8)
    fig.update_xaxes(PLOT_AXIS_REQUIREMENTS, range=[-1.5,1.5])
    fig.update_yaxes(PLOT_AXIS_REQUIREMENTS, range=[-1.5,1.5])
    fig.update_layout(
        barmode='group',
        font_family='Helvetica',
        font_color='black',
        width=600,
        height=500,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.write_image(
        f'{config.output_folder}/img/PEPSeek_pca.svg', engine='kaleido'
    )


def js_divergence(config, host_df):
    """ Function plot JS divergence between pathogen and host 9-mer peptides.
    """
    ep_df = pd.read_csv(f'{config.output_folder}/PEPSeek/potentialEpitopeCandidates.csv')
    ep_df = ep_df[['peptide']]
    ep_df = ep_df[ep_df['peptide'].apply(lambda x : len(x) == 9)]
    host_df = host_df[['peptide']].drop_duplicates()
    host_df = host_df[host_df['peptide'].apply(lambda x : len(x) == 9)]
    
    create_comparison_logo_plot(
        [ep_df, host_df], ['Pathogen', 'Host'], 9,
        f'{config.output_folder}/img', 'logo_comp_plots'
    )
