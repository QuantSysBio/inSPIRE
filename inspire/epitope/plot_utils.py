""" Script containing functions for writing plotting utils of inSPIRE epitope candidates results.
"""
import os

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA

from inspire.constants import PLOT_AXIS_REQUIREMENTS


def bar_plot(config):
    """ Function generate bar plots of shared and inSPIRE only identification counts.
    """
    ep_df = pd.read_csv(f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv')
    insp_df = ep_df[ep_df['foundBySearchEngine'] == 'No']
    shared_df = ep_df[ep_df['foundBySearchEngine'] == 'Yes']

    insp_prot_df = insp_df.drop_duplicates(subset='protein')[['protein']]
    shared_prot_df = shared_df.drop_duplicates(subset='protein')[['protein']]
    total_df = pd.merge(shared_prot_df, insp_prot_df, how='outer', on='protein', indicator=True)
    insp_prot_count = total_df[total_df['_merge'] == 'right_only'].shape[0]
    shared_prot_count = total_df[total_df['_merge'] != 'right_only'].shape[0]

    fig = px.bar(
        color=['Shared', 'Shared', 'inSPIRE Only', 'inSPIRE Only'],
        y=[shared_df.shape[0],shared_prot_count,insp_df.shape[0],insp_prot_count],
        x=['Epitope', 'Antigen', 'Epitope', 'Antigen',],
        color_discrete_map={
            'inSPIRE Only': '#FFBE00',
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

    fig.write_image(
        f'{config.output_folder}/img/epitope_bar_plot.svg', engine='kaleido'
    )


def swarm_plots(config):
    """ Function to provide bee swarm plots of various metrics for shared and inSPIRE
        only identifications. 
    """
    ep_df = pd.read_csv(f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv')

    fig = make_subplots(rows=1, cols=3)
    range_cuts = []
    for met_idx, metric in enumerate(['spectralAngle', 'engineScore', 'deltaRT']):
        mini_fig = px.strip(
            color=ep_df['foundBySearchEngine'],
            y=ep_df[metric],
            x=ep_df[
                'foundBySearchEngine'
            ].str.replace('Yes', 'Shared').replace('No', 'inSPIRE Only'),
            color_discrete_map={
                'No': '#FFBE00',
                'Yes': 'darkgrey',
            },
        )
        range_cuts.append(
            10 + 10*(ep_df[metric].max()//10)
        )
        for trace in mini_fig['data']:
            fig.add_trace(trace, row=1, col=1+met_idx)

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

    fig.write_image(
        f'{config.output_folder}/img/epitope_metrics.svg', engine='kaleido'
    )


def plot_binding_clustermap(config):
    """ Function to plot clustermap of peptides and their binding affinities.
    """
    ep_df = pd.read_csv(f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv')

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

    fig_length = 1+(ep_df.shape[0]//6)

    cluster_map = sns.clustermap(
        ep_df, cmap=sns.color_palette("blend:#FA3F37,#FAF4D3", as_cmap=True),
        yticklabels=True, figsize=(7,fig_length), vmin=0, vmax=20,
        linewidths=0.5, linecolor='black')
    cluster_map.figure.savefig(f'{config.output_folder}/img/epitope_affinity_cluster.svg')

def plot_quant_pca(config):
    """ Function to plot PCA first two components of quantification across files for pathogen
        peptides and a random selection of host peptides.
    """
    if not os.path.exists(f'{config.output_folder}/quant/quantified_per_file.csv'):
        return
    ep_df = pd.read_csv(f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv')
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
        f'{config.output_folder}/img/epitope_pca.svg', engine='kaleido'
    )
