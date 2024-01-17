""" Functions for plotting spectra.
"""
import os

from PyPDF2 import PdfMerger
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from inspire.constants import (
    CHARGE_KEY,
    KNOWN_PTM_WEIGHTS,
    PEPTIDE_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)

from inspire.input.msp import msp_to_df
from inspire.predict_spectra import predict_spectra
from inspire.spectral_features import calculate_spectral_features
from inspire.utils import convert_mod_seq_to_ptm_seq, fetch_scan_data
from inspire.plot_spectra.plot_spec_utils import (
    convert_names_and_mzs, create_traces, get_plot_details, get_unmatched,
    experiment_match, get_npp_ions, add_legend, update_fig_layout,
)

PLOTS_PER_PAGE = 15
PLOTS_PER_LINE = 3


def pair_plot(df_row, mz_accuracy, mz_units):
    """ Function to generate the traces and annotations needed for the pair plots
        of the spectra.

    Parameters
    ----------
    df_row : pd.Series
        An individual row of the DataFrame.

    Returns
    -------
    traces : list of plot.graph_objects
        The bar plot traces for experimental and prosit predicted spectra.
    annotations : list of dict
        A list of the annotations needed for the bar plot.
    """
    index = df_row['index']
    peptide = df_row['peptide']
    pred_intes = [-x for x in df_row['prositIntes']]

    pred_mzs, plotting_names = convert_names_and_mzs(
        df_row['modified_sequence'],
        list(df_row['prositIons'])
    )

    matched_peaks, l2_norm, matched_p_mz, matched_names = experiment_match(
        df_row['mzs'], df_row['intensities'], pred_mzs, plotting_names, mz_accuracy, mz_units
    )
    non_prosit_pred_peaks, prec_peak = get_npp_ions(
        df_row['mzs'], df_row['intensities'], matched_peaks['mzs'], l2_norm, peptide,
        df_row['charge'],
        {
            0: 0.0,
            1: KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            2: KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
        },
        df_row['ptm_seq'], mz_accuracy, mz_units
    )

    unmatched_peaks = get_unmatched(
        df_row['mzs'],
        df_row['intensities'],
        matched_peaks['mzs'],
        non_prosit_pred_peaks['mzs'],
        prec_peak['mzs'],
        l2_norm,
    )

    annotations, colours, extra_traces = get_plot_details(
        index, pred_mzs, matched_p_mz, pred_intes,
        plotting_names, peptide, matched_names, non_prosit_pred_peaks
    )


    for npp_mz, npp_inte, npp_name in zip(
        non_prosit_pred_peaks['mzs'],
        non_prosit_pred_peaks['intensities'],
        non_prosit_pred_peaks['names'],
    ):
        if npp_inte > 0.4:
            if npp_name == 'precursor':
                font_colour = 'plum'
            else:
                font_colour = 'darkseagreen'
            annotations.append(
                {
                    'x': npp_mz,
                    'ax': npp_mz,
                    'y': npp_inte + 0.1,
                    'ay': npp_inte + 0.1,
                    'xref': f'x{index+1}',
                    'yref': f'y{index+1}',
                    'text': npp_name,
                    'font_size': 8,
                    'font_color': font_colour,
                    'showarrow': False,
                }
            )


    traces = create_traces(
        pred_mzs, pred_intes, unmatched_peaks, non_prosit_pred_peaks,
        prec_peak, matched_peaks, colours,
    ) + extra_traces

    return traces, annotations


def plot_spectra(config):
    """ Function to generate pair plots of selected PSMs (experimental vs. Prosit
        predicted spectra.).

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.
    """
    add_legend(config.output_folder, config.experiment_title, 'standard')

    input_df = pd.read_csv(f'{config.output_folder}/plotData.csv')

    get_charge_from_scan_file = not CHARGE_KEY in input_df.columns
    scan_df = fetch_scan_data(input_df, config, get_charge_from_scan_file)
    scan_df = scan_df.to_pandas()

    input_df = pd.merge(
        input_df,
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )

    input_df['pepLen'] = input_df['peptide'].apply(len)
    input_df = input_df[(input_df['pepLen'] > 6) & (input_df['pepLen'] < 31)]
    input_df = input_df[input_df['charge'] < 7]

    n_groups = input_df.shape[0] // PLOTS_PER_PAGE
    if input_df.shape[0] % PLOTS_PER_PAGE:
        n_groups += 1

    input_df['modified_sequence'] = input_df['modifiedSequence'].apply(
        lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
    )
    input_df['precursor_charge'] = input_df[CHARGE_KEY]

    if 'collisionEnergy' in input_df.columns:
        input_df = input_df.rename(columns={'collisionEnergy': 'collision_energy'})
    else:
        input_df['collision_energy'] = config.collision_energy

    input_df[['modified_sequence', 'precursor_charge', 'collision_energy']].to_csv(
        f'{config.output_folder}/plotInput.csv', index=False,
    )

    predict_spectra(config, pipeline='plotSpectra')
    prosit_df = msp_to_df(
        f'{config.output_folder}/plotPredictions.msp', 'prosit', None
    )
    prosit_df = prosit_df.to_pandas()
    prosit_df = prosit_df.drop_duplicates(subset=['modified_sequence', CHARGE_KEY])

    input_df = pd.merge(
        input_df,
        prosit_df,
        how='inner',
        on=['modified_sequence', CHARGE_KEY]
    )

    input_df = input_df.reset_index(drop=True)
    input_df['group'] = input_df.index // PLOTS_PER_PAGE
    input_df['index'] = input_df.index % PLOTS_PER_PAGE
    input_df['ptm_seq'] = input_df['modifiedSequence'].apply(
        convert_mod_seq_to_ptm_seq
    )
    input_df['plot_data'] = input_df.apply(
        lambda x : pair_plot(x, config.mz_accuracy, config.mz_units),
        axis=1,
    )

    if SPECTRAL_ANGLE_KEY not in input_df.columns:
        input_df['results'] = input_df.apply(
            lambda x : calculate_spectral_features(
                x,
                {
                    0: 0.0,
                    1: KNOWN_PTM_WEIGHTS['Oxidation (M)'],
                    2: KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
                },
                config.mz_accuracy,
                config.mz_units,
                None,
                '1',
                config.delta_method,
                minimal_features=True,
            ),
            axis=1,
        )
        input_df['spectralAngle'] = input_df['results'].apply(lambda x : x.get('spectralAngle'))

    titles = []
    for idx in range(input_df.shape[0]):
        seq = input_df[PEPTIDE_KEY].iloc[idx]
        scan_nr = input_df[SCAN_KEY].iloc[idx]
        charge = input_df[CHARGE_KEY].iloc[idx]

        if config.plot_spectra_source_split is not None:
            src = input_df[SOURCE_KEY].iloc[idx].split(config.plot_spectra_source_split)[-1]
        else:
            src = input_df[SOURCE_KEY].iloc[idx]
        spectral_angle = round(input_df[SPECTRAL_ANGLE_KEY].iloc[idx], 2)
        if 'Context' in input_df.columns:
            context = input_df['Context'].iloc[idx]
            titles.append(
                f'<b>Source</b> {src} <b>Scan</b> {scan_nr}<br><b>Peptide</b> {seq} ' +
                f'<b>Charge</b> {charge} <b>Spectral Angle</b> {spectral_angle}<br>' +
                f'<b>Context</b> {context}'
            )
        else:
            titles.append(
                f'<b>Source</b> {src} <b>Scan</b> {scan_nr}<br><b>Peptide</b> {seq} ' +
                f'<b>Charge</b> {charge} <b>Spectral Angle</b> {spectral_angle}<br>'
            )

    for group_idx in range(n_groups):

        start_idx = PLOTS_PER_PAGE*group_idx
        sub_df = input_df[input_df['group'] == group_idx]
        n_plots = sub_df.shape[0]
        n_plot_rows = 1 + (n_plots//PLOTS_PER_LINE)

        fig = make_subplots(
            rows=n_plot_rows,
            cols=PLOTS_PER_LINE,
            subplot_titles = titles[start_idx:start_idx+n_plots],
        )
        for idx in range(1, (n_plot_rows*PLOTS_PER_LINE)+1):
            fig.update_layout(
                {
                    f'xaxis{idx}':{'title_text': 'm/z'},
                    f'yaxis{idx}':{'title_text': 'L<sup>2</sup> Normalized Intensity'},
                }
            )

        plot_data = sub_df['plot_data'].tolist()
        for idx, (traces, annotations) in enumerate(plot_data):
            for trace in traces:
                fig.add_trace(
                    trace,
                    row=1 + (idx//PLOTS_PER_LINE),
                    col=1 + (idx%PLOTS_PER_LINE),
                )

            fig.layout['annotations'] += tuple(annotations)

        for idx in range(sub_df.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[0, 1500],
                    y=[0, 0],
                    mode='lines',
                    line={'width':0.5, 'color':'black'},
                ),
                row=1 + (idx//PLOTS_PER_LINE),
                col=1 + (idx%PLOTS_PER_LINE),
            )

        fig = update_fig_layout(fig, n_plot_rows)

        if not config.silent_execution:
            fig.show()

        pio.write_image(
            fig,
            f'{config.output_folder}/spectralPlots{group_idx}.pdf',
            engine='kaleido',
        )

    merger = PdfMerger()
    merger.append(
        f'{config.output_folder}/spectralPlots_legend.pdf'
    )
    for group_idx in range(n_groups):
        merger.append(
            f'{config.output_folder}/spectralPlots{group_idx}.pdf'
        )
    merger.write(f'{config.output_folder}/spectralPlots.pdf')
    merger.close()

    for group_idx in range(n_groups):
        os.remove(
            f'{config.output_folder}/spectralPlots{group_idx}.pdf'
        )
    os.remove(
        f'{config.output_folder}/spectralPlots_legend.pdf'
    )
