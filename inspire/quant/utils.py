""" Functions used across inSPIRE quantification pipeline.
"""
from math import log2

import pandas as pd
from plotly.colors import n_colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA

from inspire.constants import PLOT_AXIS_REQUIREMENTS
from inspire.logo_plot_utils import create_comparison_logo_plot


def plot_correlations(quant_file_name, code, config):
    """ Function to calculate the correlation between raw files and plot as heat map
        and write as html/svg.

    Parameters
    ----------
    quant_file_name : str
        Name of csv file containing quantification results.
    code : str
        Either "raw" or "norm" indicating if raw or normalised intensities are being compared.
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    quant_df = pd.read_csv(
        f'{config.output_folder}/quant/{quant_file_name}.csv'
    )
    file_list = sorted([
        x for x in quant_df.columns if x.endswith(f'_{code}')
    ])

    if code == 'raw':
        for source in file_list:
            quant_df[source] = quant_df[source].apply(
                lambda intensity : log2(intensity) if intensity > 0 else None
            )

    correlations = {}
    for source in file_list:
        correlations[source] = []
        for source2 in file_list[::-1]:
            idp_key = source.replace(f'_{code}', '_idp')
            idp2_key = source2.replace(f'_{code}', '_idp')
            if code == 'raw':
                sub_quant_df = quant_df[
                    (quant_df[idp_key] > config.skyline_idp_cut_off) &
                    (quant_df[idp2_key] > config.skyline_idp_cut_off)
                ]

            sub_quant_df = quant_df[
                (quant_df[source].notna()) &
                (quant_df[source2].notna())
            ]

            correlations[source].append(
                round(
                    scipy.stats.pearsonr(
                        sub_quant_df[source],
                        sub_quant_df[source2],
                    )[0],
                    2,
                )
            )


    cor_df = pd.DataFrame(correlations)
    cor_df.index = file_list
    fig = go.Figure(
        go.Heatmap(
            z=cor_df.values,
            x=file_list,
            y=file_list[::-1],
            text=cor_df.values,
            texttemplate="%{text}",
            # textfont={"size":3},
            colorscale='RdBu_r',
            zmin=0,zmax=1,
        ),
    )

    fig.update_layout(
        width=500,
        height=300,
        font_size=8,
        font_family='Helvetica',
        font_color='black',
        margin={'r':25, 'l':25, 't':25, 'b':25},
    )

    fig.write_image(
        f'{config.output_folder}/img/{code}_correlation.svg', engine='kaleido'
    )


def plot_quant_clustermap(config):
    """ Function to plot clustermap of peptides and their binding affinities.

    Parameters
    ----------
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    normed_df = pd.read_csv(f'{config.output_folder}/quant/normalised_quantification.csv')

    normed_df.index = normed_df['peptide']
    normed_df = normed_df[[col for col in normed_df.columns if col.endswith('_norm')]]

    normed_df = normed_df.rename(
        columns={col:col.strip('_norm') for col in normed_df.columns}
    )

    normed_df = normed_df.sort_values(by=list(normed_df.columns))
    normed_df = normed_df[normed_df.notnull().min(axis=1)]

    z_score_df = pd.DataFrame(
        zscore(normed_df, axis=1, nan_policy='omit'),
        index=normed_df.index,
        columns=normed_df.columns,
    )

    cluster_map = sns.clustermap(
        z_score_df, cmap="vlag", figsize=(10,30), cbar_pos=None, dendrogram_ratio=(0.1,0.03),
    )
    cluster_map.figure.savefig(f'{config.output_folder}/img/quant_clustermap.svg')

    z_score_df = z_score_df.transpose()
    pca = PCA(n_components=2).fit(z_score_df)

    pca_df = pd.DataFrame(
        pca.transform(z_score_df),
        columns=[
            f'Component 1 (Exp. Var. {round(pca.explained_variance_ratio_[0], 2)})',
            f'Component 2 (Exp. Var. {round(pca.explained_variance_ratio_[1], 2)})',
        ],
        index=z_score_df.index)
    pca_df /= pca_df.abs().max()
    pca_df['file'] = z_score_df.index
    pca_df['color'] = pca_df['file'].apply(
        lambda x : 'red' if x.startswith('inf') else (
            'blue' if x.startswith('control') else 'green'
        )
    )
    n_inf_files = pca_df[pca_df['color']=='red'].shape[0]
    n_cont_files = pca_df[pca_df['color']=='blue'].shape[0]
    n_unknown_files = pca_df[pca_df['color']=='green'].shape[0]

    colors1 = n_colors('rgb(0, 0, 0)', 'rgb(211, 211, 211)', n_cont_files, colortype='rgb')
    colors2 = n_colors('rgb(0, 100, 0)', 'rgb(144, 238, 144)', n_inf_files, colortype='rgb')
    colors3 = n_colors('rgb(0, 0, 128)', 'rgb(30, 144, 255)', n_unknown_files, colortype='rgb')

    fig = px.scatter(pca_df, x=pca_df.columns[0], y=pca_df.columns[1], color='file',
                     color_discrete_sequence=colors1+colors2+colors3)
    fig.update_traces(opacity=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_traces(marker_size=8)
    fig.update_xaxes(PLOT_AXIS_REQUIREMENTS, range=[-2,2])
    fig.update_yaxes(PLOT_AXIS_REQUIREMENTS, range=[-2,2])
    fig.update_layout(
        barmode='group',
        font_family='Helvetica',
        font_color='black',
        width=600,
        height=500,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
    )
    fig.write_image(
        f'{config.output_folder}/img/quant_pca.svg', engine='kaleido'
    )


def plot_distros(quant_df, config, sources):
    """ Function to plot distribution of peptide intensities across raw files before and
        after normalisation.

    Parameters
    ----------
    quant_df : pd.DataFrame
        Quantifications of each peptide (rows) for each raw file (columns).
    config : inspire.config.Config
        Config object for the whole experiment.
    sources : list of str
        List of the raw files.
    """
    all_valids = []
    for source in sources:
        valid_quant_df = quant_df[
            (quant_df[f'{source}_clean'].apply(lambda a : a is not None)) &
            (quant_df[f'{source}_norm'].apply(lambda a : a is not None))
        ]
        valid_quant_df['File'] = source
        all_valids.append(valid_quant_df[[
            'File', f'{source}_clean', f'{source}_norm'
        ]].rename(columns={
            f'{source}_norm': 'Normalised Intensity',
            f'{source}_clean': 'Raw Intensity'
        }))
    total_df = pd.concat(all_valids)

    fig=go.Figure()
    fig.add_trace(go.Violin(
        x=total_df['File'],y=total_df['Raw Intensity'],
        fillcolor='#ACE1AF', meanline_visible=True,
        side='negative', name='Raw', points=False,
        line_color='black',
        line_width=0.5,
    ))
    fig.add_trace(go.Violin(
        x=total_df['File'],y=total_df['Normalised Intensity'],
        points=False,
        fillcolor='darkgreen',
        line_color='black',
        line_width=0.5,
        meanline_visible=True,side='positive', name='Normalised'))
    fig.update_traces(marker={'line': {'color': 'black', 'width': 0.5}})
    fig.add_hline(
        y=total_df['Normalised Intensity'].mean(),
        line_color='darkgreen',
        line_width=1.5,
        line_dash='dash',
    )
    fig.add_hline(
        y=total_df['Raw Intensity'].mean(),
        line_color='#ACE1AF',
        line_width=1.5,
        line_dash='dash',
    )
    fig.update_xaxes(PLOT_AXIS_REQUIREMENTS)
    fig.update_yaxes(PLOT_AXIS_REQUIREMENTS, range=[15,35])
    fig.update_layout(
        font_family='Helvetica',
        font_color='black',
        width=900,
        height=400,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
    )
    fig.write_image(
        f'{config.output_folder}/img/quant_distro.svg', engine='kaleido'
    )


def join_peptide(df_row, clusters):
    """ Function to join peptide sequences from different clusters into a single string.
    """
    peptide = ''
    for cluster in clusters:
        if isinstance(df_row[cluster], str):
            peptide += df_row[cluster]
        else:
            peptide += 'X'
    return peptide

def create_quant_logoplots(de_df, output_folder, centers):
    """ Function to create quantitative logoplots based on normalised inspire dataframe
    """
    sorted_clusters = [x[1] for x in sorted(list(zip(
        centers, list(range(7))
    )))]
    create_violin_cluster_plot(de_df, sorted_clusters, output_folder)
    center_cluster = sorted_clusters.pop(3)
    center_cluster_c_term = de_df[
        de_df['cluster'] == center_cluster
    ]['peptide'].apply(lambda x : x[-1]*6)

    center_df = pd.DataFrame({
        'peptide' : center_cluster_c_term
    })
    cluster_df = pd.DataFrame({
        idx: de_df[
            de_df['cluster'] == idx
        ]['peptide'].apply(lambda x : x[-1]) for idx in sorted_clusters
    })
    cluster_df['peptide'] = cluster_df.apply(
        lambda x : join_peptide(x, sorted_clusters), axis=1
    )

    create_comparison_logo_plot(
        [cluster_df, center_df],
        ['C-terminus: Different Clusters', 'Center Cluster'],
        6,
        f'{output_folder}/img',
        'quant_logo_plots',
        data_index=[1,2,3,5,6,7]
    )

def create_violin_cluster_plot(de_df, sorted_clusters, output_folder):
    """ Create figure showing distribution of fold changes across the cluste
    """
    violin_fig = go.Figure()
    for idx, cluster in enumerate(sorted_clusters):
        cluster_df = de_df[de_df['cluster'] == cluster]
        violin_fig.add_trace(
            go.Violin(
                x=[idx+1]*cluster_df.shape[0], y=cluster_df['foldChange'],
                fillcolor='pink', line_color='black', line_width=0.5,
                points=False
            )
        )
     # Clean axes and layout
    violin_fig.update_xaxes(
        showline=False,
        showticklabels=True,
        title_text='Cluster',
        linecolor='black',
        linewidth=0.5,
        showgrid=False,
        ticks="outside",
        dtick=1,
    )
    violin_fig.add_hline(y=0, line_color='black', line_width=0.5, line_dash='dash')
    violin_fig.update_yaxes(
        showline=True,
        showticklabels=True,
        linewidth=0,
        linecolor='black',
        title_text='Log Fold Change',
        showgrid=False,
        ticks="outside",
    )
    violin_fig.update_layout(
        width=500,
        height=300,
        paper_bgcolor='rgba(256,256,256,256)',
        plot_bgcolor='rgba(256,256,256,256)',
        showlegend=False,
        font_color='black',
        font_family='Helvetica',
    )

    pio.write_image(
        violin_fig, f'{output_folder}/img/quant_cluster_violin.svg',
    )
