""" Functions to create figures needed for the inSPIRE report.
"""
import numpy as np
from plotly.colors import n_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_binders_fig(fdrs_df, search_engine):
    """ Function to create a plotly figure of percentage binders as predicted
        by NetMHCpan against q-value cut off.

    Parameters
    ----------
    fdrs_df : pd.DataFrame
        A DataFrame of counts of binders and PSMs at different q-values.
    search_engine : str
        The name of the original ms search engine used.

    Returns
    -------
    fig : str
        A plot of the percentage predicted binders against q-value converted to html.
    """
    trace1 = go.Scatter(
        x=fdrs_df['FDR'],
        y=fdrs_df['SpirePercentageBinders'],
        name='inSPIRE Percentage Binders',
        line={'color': 'olivedrab'},
        mode='lines+markers',
        connectgaps=True,
    )
    trace2 = go.Scatter(
        x=fdrs_df['FDR'],
        y=fdrs_df['searchEnginePercentageBinders'],
        name='Original Percentage Binders',
        line={'color': 'violet'},
        mode='lines+markers',
        connectgaps=True,
    )

    data = [trace1, trace2]
    layout2 = go.Layout(
        yaxis=dict(title='Percentage of all Identified PSMs'),
        xaxis=dict(title='q-value'),
    )
    fig = go.Figure(data=data, layout=layout2)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        title=f'Percentage Binders Identified with {search_engine.title()} and inSPIRE',
        width=1000,
        height=500,
        title_x=0.5
    )
    fig.update_xaxes(showline=True, linewidth=0.5, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=0.5, linecolor='black')

    return fig.to_html()


def create_psms_fig(fdrs_df, search_engine):
    """ Function for plotting the number of PSMs identified at varying q-values
        for inspire and the original search engine.

    Parameters
    ----------
    fdrs_df : pd.DataFrame
        A DataFrame of statistics for identifications at varying q-values.
    search_engine : str
        The name of the search engine used.

    Returns
    -------
    fig : str
        The plot of PSMs vs. q-value converted to html.
    """
    search_engine_trace = go.Scatter(
        x=fdrs_df['FDR'],
        y=fdrs_df['nSearchEnginePsms'],
        name='Number of Search Engine PSMs',
        line={'color': 'violet'},
        mode='lines+markers',
        connectgaps=True,
    )
    spire_trace = go.Scatter(
        x=fdrs_df['FDR'],
        y=fdrs_df['nSpirePsms'],
        name='Number of inSPIRE PSMs',
        line={'color': 'olivedrab'},
        mode='lines+markers',
        connectgaps=True,
    )

    data = [search_engine_trace, spire_trace]
    layout = go.Layout(
        yaxis=dict(title='Number of Identifications'),
        xaxis=dict(title='q-value'),
    )
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        title=f'PSMs Identified with {search_engine.title()} and inSPIRE',
        width=1000,
        height=500,
        title_x=0.5,
    )
    fig.update_xaxes(showline=True, linewidth=0.5, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=0.5, linecolor='black')

    return fig.to_html()


def create_violin_fig(combined_df, most_positive_features, most_negative_features):
    """ Function to create a plotly figure of violin plots of the distributions of
        key features above and below the 1% FDR cut off.

    Parameters
    ----------
    combined_df : pd.DataFrame
        A DataFrame of inSPIRE output with input features.
    most_positive_features : list of str
        The most positively weighted features by percolator/mokapot.
    most_negative_features : list of str
        The most negatively weighted features by percolator/mokapot.

    Returns
    -------
    fig : str
        Plots of the distribution of each feature for accepted and rejected PSMs
        converted to html.
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(most_positive_features + most_negative_features),
        vertical_spacing=0.12,
    )

    for row_idx, feature_group in enumerate([most_positive_features, most_negative_features]):
        for idx, feature in enumerate(feature_group):
            fig.add_trace(
                go.Violin(
                    x=combined_df[combined_df['Status'] == 'Accepted']['Status'],
                    y=combined_df[combined_df['Status'] == 'Accepted'][feature],
                    # color=combined_df['Status'],
                    name='Accepted',
                    fillcolor='#8FBC8F',
                    line_color='black',
                    line_width=0.5,
                    marker_size=0.25,
                    scalegroup=(row_idx*3) + idx,
                    # box_visible=True,
                    meanline_visible=True,
                ),
                row=row_idx+1,
                col=idx+1,
            )
            fig.add_trace(
                go.Violin(
                    x=combined_df[combined_df['Status'] == 'Rejected']['Status'],
                    y=combined_df[combined_df['Status'] == 'Rejected'][feature],
                    fillcolor='#a52a2a',
                    line_color='black',
                    line_width=0.5,
                    marker_size=0.25,
                    # color=combined_df['Status'],
                    name='Rejected',
                    scalegroup=(row_idx*3) +idx,
                    # box_visible=True,
                    meanline_visible=True,
                ),
                row=row_idx+1,
                col=idx+1,
            )

    fig.update_layout(
        title='Distribution of High Importance Features',
        width=1100,
        height=700,
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showline=True, linewidth=0.5, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=0.5, linecolor='black')

    return fig.to_html()

def _get_color_map(feature_weights, wt_max, wt_min):
    """ Helper function to get the color weighting of different features
        based on percolator feature importance.

    Parameters
    ----------
    feature_weights : pd.Series
        A Series of feature weights for a single fold.
    wt_max : float
        The maximum features weight across all folds.
    wt_min : float
        The minimum features weight across all folds.

    Returns
    -------
    col_weights : list of int
        A list of the value on the color spectrum for each feature.
    """
    col_weights = []
    for entry in feature_weights:
        if entry >= 0:
            normed_val = entry/wt_max
            col_weights.append(128 + int(round(127*normed_val)))
        else:
            normed_val = 1 - (entry/wt_min)
            col_weights.append(int(round(127*normed_val)))

    return col_weights

def create_weights_fig(
        weights_df
    ):
    """ Function to create a table of percolator/mokapot features and their
        weights.

    Parameters
    ----------
    weights_df : pd.DataFrame
        A Series of feature weights for a single fold.

    Returns
    -------
    fig : str
        A table of the features and their weights in Percolator converted to html.
    """
    colors1 = n_colors('rgb(255, 204, 204)', 'rgb(255, 255, 255)', 128, colortype='rgb')
    colors2 = n_colors('rgb(255, 255, 255)', 'rgb(204, 255, 204)', 128, colortype='rgb')
    colors = colors1 + colors2
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    'values': [
                        'feature',
                        'weightFold1',
                        'weightFold2',
                        'weightFold3',
                        'averageWeight',
                    ],
                    'line_color': 'black',
                },
                cells={
                    'values': [
                        weights_df.feature,
                        weights_df.weightFold1,
                        weights_df.weightFold2,
                        weights_df.weightFold3,
                        [round(x, 3) for x in weights_df.averageWeight],
                    ],
                    'fill_color': [
                        np.array(['#FFFFFF']*len(weights_df)),
                        np.array(colors)[_get_color_map(
                            weights_df.weightFold1,
                            weights_df.weightFold1.max(),
                            weights_df.weightFold1.min(),
                        )],
                        np.array(colors)[_get_color_map(
                            weights_df.weightFold2,
                            weights_df.weightFold2.max(),
                            weights_df.weightFold2.min(),
                        )],
                        np.array(colors)[_get_color_map(
                            weights_df.weightFold3,
                            weights_df.weightFold3.max(),
                            weights_df.weightFold3.min(),
                        )],
                        np.array(colors)[_get_color_map(
                            weights_df.averageWeight,
                            weights_df.averageWeight.max(),
                            weights_df.averageWeight.min(),
                        )],
                    ],
                    'line_color': 'black',
                }
            )
        ]
    )
    fig.update_layout(
        title='Weights of Features Selected by inSPIRE.',
        width=1000,
        height=1100,
        title_x=0.5
    )
    return fig.to_html()
