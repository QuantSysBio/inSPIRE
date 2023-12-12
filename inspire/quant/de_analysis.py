""" Functions to perform differential expression analysis between infected and
    control measurements.
"""
from math import log2, ceil

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests

from inspire.constants import ACCESSION_KEY, PEPTIDE_KEY
from inspire.utils import is_control

FC_CUT = 0.5
P_VAL_CUT = 0.05
LOG_P_VAL_CUT = -log2(0.05)


def de_analysis(config):
    """ Function to perform differential expression analysis between infected and control
        measurements.

    Parameters
    ----------
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    if config.control_flags is None:
        return

    quant_df = pd.read_csv(
        f'{config.output_folder}/quant/normalised_quantification.csv'
    )

    quant_df, ctrl_samp_names, inf_samp_names = combine_samples(quant_df, config)

    quant_df = filter_by_n_valid_measurement(quant_df, ctrl_samp_names, inf_samp_names)

    quant_df = calculate_fold_change_and_p_val(quant_df, ctrl_samp_names, inf_samp_names)

    if quant_df.shape[0]:
        create_volcano_plot(quant_df, config)

        # Write outputs.
        quant_df['absFoldChange'] = quant_df['foldChange'].abs()
        quant_df = quant_df.sort_values(by='absFoldChange', ascending=False)
        quant_df[[
            PEPTIDE_KEY, ACCESSION_KEY,
        ] + ctrl_samp_names + inf_samp_names + [
            'foldChange',
            'pValue',
            'adjusted_pValue',
            'diffExprResult',
        ]].to_csv(f'{config.output_folder}/quant/de_peptide_results.csv', index=False)


def combine_samples(quant_df, config):
    """ Function to combine technical replicate values into an averaged sample value.
    """
    meta_df = pd.read_csv(f'{config.output_folder}/quant/metadata.csv')
    file_name_mapping = dict(zip(
        meta_df['source'].tolist(), meta_df['renamed'].tolist()
    ))

    ctrl_samples = 0
    inf_samples = 0
    sample_meta_list = []
    sample_names = []
    ctrl_samp_names = []
    inf_samp_names = []
    if config.technical_replicates is None:
        config.technical_replicates = [[raw_file] for raw_file in meta_df['source'].tolist()]
    for sample_group in config.technical_replicates:
        if is_control(sample_group[0], config.control_flags):
            ctrl_samples += 1
            name = f'controlSample{ctrl_samples}'
            ctrl_samp_names.append(name)
        else:
            inf_samples += 1
            name = f'infectionSample{inf_samples}'
            inf_samp_names.append(name)

        quant_df[name] = quant_df[
            [f'{file_name_mapping[raw_file]}_norm' for raw_file in sample_group]
        ].mean(axis=1)

        for raw_file in sample_group:
            sample_meta_list.append(
                {'source': raw_file, 'sample': name}
            )
        sample_names.append(name)

    # Add the sample to the metadata
    if 'sample' in meta_df.columns:
        meta_df = meta_df.drop('sample', axis=1)

    sample_meta_df = pd.DataFrame(sample_meta_list)
    meta_df = pd.merge(meta_df, sample_meta_df, on='source', how='inner')
    meta_df.to_csv(f'{config.output_folder}/quant/metadata.csv', index=False)

    quant_df = quant_df[[PEPTIDE_KEY, ACCESSION_KEY] + sample_names]

    return quant_df, ctrl_samp_names, inf_samp_names


def filter_by_n_valid_measurement(quant_df, ctrl_samp_names, inf_samp_names):
    """ Function to filter the quantification DataFrame for differential expression
        analysis so that it contains at least 3 samples for both control and infected.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of normalised quantified peptides.
    ctrl_samp_names : list of str
        A list of the sample names for control measurements.
    inf_samp_names : list of str
        A list of the sample names for infected measurements.

    Returns
    -------
    quant_df : pd.DataFrame
        The input DataFrame filtered by number of valid quantifications.
    """
    quant_df['nValidControl'] = quant_df[ctrl_samp_names].count(axis=1)
    quant_df['nValidInf'] = quant_df[inf_samp_names].count(axis=1)
    quant_df = quant_df[
        (quant_df['nValidControl'] >= 3) &
        (quant_df['nValidInf'] >= 3)
    ]

    return quant_df


def calculate_fold_change_and_p_val(quant_df, ctrl_samp_names, inf_samp_names):
    """ Function to calculate the fold change and p-values for difference between infected
        and control samples.

    Parameters
    ----------
    quant_df : pd.DataFrame
        DataFrame of normalised quantified peptides.
    ctrl_samp_names : list of str
        A list of the sample names for control measurements.
    inf_samp_names : list of str
        A list of the sample names for infected measurements.

    Returns
    -------
    quant_df : pd.DataFrame
        The input DataFrame with differential expression analysis performed.
    """
    # Get geometric mean values:
    quant_df['meanControl'] = quant_df[ctrl_samp_names].mean(axis=1)
    quant_df['meanInfected'] = quant_df[inf_samp_names].mean(axis=1)

    # Calculate fold change and p-value
    quant_df['foldChange'] = quant_df['meanInfected'] - quant_df['meanControl']
    quant_df['pValue'] = quant_df.apply(
        lambda df_row : ttest_ind(
            [df_row[x] for x in ctrl_samp_names if not np.isnan(df_row[x])],
            [df_row[x] for x in inf_samp_names if not np.isnan(df_row[x])],
            equal_var=True,
        )[1],
        axis=1,
    )

    quant_df['-lg2(pValue)'] = quant_df['pValue'].apply(lambda x : -log2(x))

    if not quant_df.shape[0]:
        return quant_df

    quant_df['adjusted_pValue'] = multipletests(quant_df['pValue'], method='fdr_bh')[1]

    # Define results for volcano plot.
    quant_df['diffExprResult'] = quant_df[['pValue', 'foldChange']].apply(
        lambda df_row : 'Not Significant' if df_row['pValue'] >= P_VAL_CUT or abs(
            df_row['foldChange']
        ) < FC_CUT else (
            'Upregulated' if df_row['foldChange'] > 0 else 'Downregulated'
        ),
        axis=1
    )

    return quant_df


def create_volcano_plot(quant_df, config):
    """ Function to create a volcano plot based on the peptide intensity changes.

    Parameters
    ----------
    quant_df : pd.DataFrame
        A DataFrame of quantified peptides with DE analysis performed.
    config : inspire.config.Config
        Config object for the whole experiment.
    """
    # Scatter plot fold change and -lg of p value
    fig = px.scatter(
        quant_df,
        x='foldChange',
        y='-lg2(pValue)',
        color='diffExprResult',
        color_discrete_map={
            'Downregulated': 'royalblue',
            'Upregulated': 'tomato',
            'Not Significant': 'grey',
        },
        hover_name='peptide',
        hover_data='peptide',
    )
    fig.update_traces(marker={'size': 4})

    # Add lines for significance cut-offs
    fig.add_hline(y=LOG_P_VAL_CUT, line_color='black', line_dash='dash', line_width=0.5)
    fig.add_vline(x=FC_CUT, line_color='black', line_dash='dash', line_width=0.5)
    fig.add_vline(x=-FC_CUT, line_color='black', line_dash='dash', line_width=0.5)

    # Calculate the range over which fold changes should be plotted
    fc_max = quant_df['foldChange'].max()
    fc_min = quant_df['foldChange'].min()
    fc_range = ceil(max(fc_max, abs(fc_min)))
    if fc_range % 2:
        fc_range += 1

    # Clean axes and layout
    fig.update_xaxes(
        showticklabels=True,
        range=[-fc_range, fc_range],
        title_text='Log Fold Change',
        linecolor='black',
        linewidth=0.5,
        showgrid=False,
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        showticklabels=True,
        linewidth=0,
        linecolor='black',
        title_text='-log2(pValue)',
        range=[0,ceil(quant_df['-lg2(pValue)'].max())],
        showgrid=False,
        ticks="outside",
    )
    fig.update_layout(
        width=500,
        height=300,
        paper_bgcolor='rgba(256,256,256,256)',
        plot_bgcolor='rgba(256,256,256,256)',
        showlegend=False,
        font_color='black',
        font_family='Helvetica',
    )

    # Save image as svg
    fig.write_image(
        f'{config.output_folder}/img/peptide_volcano.svg', engine='kaleido'
    )
