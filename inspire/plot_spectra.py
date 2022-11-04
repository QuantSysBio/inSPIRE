""" Functions for plotting spectra.
"""
import os

from PyPDF2 import PdfFileMerger
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from inspire.constants import (
    CHARGE_KEY,
    KNOWN_PTM_WEIGHTS,
    PEPTIDE_KEY,
    PROTON,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.input.mgf import process_mgf_file

from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.mz_match import get_ion_masses
from inspire.predict_spectra import predict_spectra
from inspire.spectral_features import calculate_spectral_features

PLOTS_PER_PAGE = 15
PLOTS_PER_LINE = 3


def convert_names_and_mzs(mod_seq, pred_names):
    """ Function to generate plotting ion names and mzs.

    Parameters
    ----------
    mod_seq : str
        The Prosit modified_sequence.
    pred_names : list of str
        A list of the names of the ions as read in from msp format.

    Returns
    -------
    pred_mzs : list of float
        A list of the theoretical mz values for the prosit predicted ions.
    plotting_names : list of str
        A list of the annotations for the prosit predicted ions.
    """
    if '(ox)' not in mod_seq and 'C' not in mod_seq:
        mods = None
        un_mod_seq = mod_seq.replace('(ox)', '')
    else:
        mods = '0.'
        un_mod_seq = ''
        while mod_seq:
            if len(mod_seq) > 1:
                if mod_seq[1] == '(':
                    un_mod_seq += 'M'
                    mods += '1'
                    mod_seq = mod_seq[5:]
                    continue
            un_mod_seq += mod_seq[0]
            if mod_seq[0] != 'C':
                mods += '0'
            else:
                mods += '2'
            mod_seq = mod_seq[1:]
        mods += '.0'
    masses, _ = get_ion_masses(
        un_mod_seq,
        {
            0: 0.0,
            1: KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            2: KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
        },
        modifications=mods
    )
    pred_mzs = []
    plotting_names = []
    for pred_ion in pred_names:
        ion_data = pred_ion.split('^')
        if '^' in pred_ion:
            charge = int(ion_data[1])
        else:
            charge = 1
        plotting_names.append(ion_data[0] + ('+' * charge))
        code = ion_data[0][0]
        idx = int(ion_data[0][1:])
        pred_mzs.append(
            (masses[code][idx-1] + (PROTON*charge))/charge
        )
    return pred_mzs, plotting_names

def get_unmatched(exp_mzs, exp_intes, matched_mzs, l2_norm):
    """ Function select and normalise the unmatched peaks from the experimental spectrum.

    Parameters
    ----------
    exp_mzs : list of float
        List of the mz values of the experimental spectrum.
    exp_intes : list of float
        List of the intensity values of the experimental spectrum.
    matched_mzs : list of float
        List of the mz values matched to the Prosit spectrum.
    l2_norm : float
        Either the l2 norm of the matched spectrum or the maximum intensity in the
        experimental spectrum if fewer than 5 peaks are matched.

    Returns
    -------
    unmatched_mzs : list of float
        List of the mz values of the experimental spectrum not matched
        to the Prosit predicted spectrum.
    exp_intes : list of float
        List of the intensity values of the experimental spectrum not
        matched to the Prosit predicted spectrum.
    """
    unmatched_mzs, unmatched_intes = [], []
    for idx, e_mz in enumerate(exp_mzs):
        if e_mz not in matched_mzs:
            unmatched_mzs.append(e_mz)
            unmatched_intes.append(exp_intes[idx]/l2_norm)

    return {
        'mzs': unmatched_mzs,
        'intensities': unmatched_intes,
    }

def experiment_match(exp_mzs, exp_intes, pred_mzs, plotting_names, mz_accuracy, mz_units):
    """ Function to match the mz values of the experimentally observed peaks to the
        Prosit predicted peaks.

    Parameters
    ----------
    exp_mzs : list of float
        The experimentally observed mz values.
    exp_intes : list of float
        The experimentally observed intensity values.
    pred_mzs : list of float
        The Prosit predicted mz values.

    Returns
    -------
    matched_mzs : list of float
        The matched mz values in the experimental spectrum.
    matched_intes : list of float
        The matched intensity values in the experimental spectrum.
    l2_norm : float
        The l2 norm of the matched intensities if more than 5 peaks are matched,
        otherwise the maximum intensity in the spectrum.
    matched_pred_mzs : list of float
        A list of the matched prosit predicted mz values.
    """
    matched_mzs, matched_intes = [], []
    matched_pred_mzs = []
    matched_names = []
    for match_name, pred_m in zip(plotting_names, pred_mzs):
        matched_ind = -1
        min_match = 100000
        for idx, exp_mz in enumerate(exp_mzs):
            if abs(exp_mz-pred_m) < min_match:
                min_match = abs(exp_mz-pred_m)
                matched_ind = idx

        if mz_units == 'ppm':
            mz_err = pred_m*mz_accuracy*(10**-6)
        else:
            mz_err = mz_accuracy

        if min_match < mz_err:
            matched_mzs.append(exp_mzs[matched_ind])
            matched_intes.append(exp_intes[matched_ind])
            matched_pred_mzs.append(pred_m)
            matched_names.append(match_name.split('+')[0])

    if len(matched_intes) > 5:
        l2_norm = np.linalg.norm(np.array(matched_intes), ord=2)
    else:
        l2_norm = np.max(exp_intes)

    matched_peaks = {
        'mzs': matched_mzs,
        'intensities': [x/l2_norm for x in matched_intes],
    }

    return matched_peaks, l2_norm, matched_pred_mzs, matched_names


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
    pred_intes = [-x for x in df_row['prositIons'].values()]

    pred_mzs, plotting_names = convert_names_and_mzs(
        df_row['modified_sequence'],
        list(df_row['prositIons'].keys())
    )

    matched_peaks, l2_norm, matched_p_mz, matched_names = experiment_match(
        df_row['mzs'], df_row['intensities'], pred_mzs, plotting_names, mz_accuracy, mz_units
    )
    unmatched_peaks = get_unmatched(
        df_row['mzs'], df_row['intensities'], matched_peaks['mzs'], l2_norm
    )

    annotations = []
    colours = []
    for idx, entry in enumerate(pred_mzs):
        if entry in matched_p_mz:
            colour = '#0095A8'
        else:
            colour = '#FF7043'
        colours.append(colour)

        if pred_intes[idx] < -0.05:
            annotations.append(
                {
                    'x': entry,
                    'ax': entry,
                    'y': pred_intes[idx]-0.05,
                    'ay': pred_intes[idx]-0.05,
                    'xref': f'x{index+1}',
                    'yref': f'y{index+1}',
                    'text': plotting_names[idx],
                    'font_size': 8,
                    'font_color': colour,
                    'showarrow': False,
                }
            )

    extra_traces = []
    annotations.append({
        'showarrow': False,
            'text': 'Experimental Spectrum',
            'x': 1300,
            'ax': 1300,
            'y': 1.5,
            'ay': 1.5,
            'font_size': 12,
            'font_family': "Arial, monospace",
            'xref': f'x{index+1}',
            'yref': f'y{index+1}',
            'align': 'left',
    })
    annotations.append({
        'showarrow': False,
            'text': 'Prosit Predicted Spectrum',
            'x': 1300,
            'ax': 1300,
            'y': -1.1,
            'ay': -1.1,
            'font_size': 12,
            'font_family': "Arial, monospace",
            'xref': f'x{index+1}',
            'yref': f'y{index+1}',
            'align': 'left',
    })
    for idx, residue in enumerate(peptide):
        annotations.append({
            'showarrow': False,
            'text': (residue),
            'x': 50 + (idx * 50),
            'ax': 50 + (idx * 50),
            'y': 1.25,
            'ay': 1.25,
            'font_size': 12,
            'font_family': "Arial, monospace",
            'xref': f'x{index+1}',
            'yref': f'y{index+1}',
            'align': 'left',
        })
        if idx > 0:
            min1_idx = idx-1
            rev_idx = len(peptide) - idx
            y_matched = False
            b_matched = False
            if f'b{idx}' in matched_names:
                annotations.append({
                    'showarrow': False,
                    'text': f'b{idx}',
                    'x': 40 + (min1_idx * 50),
                    'ax': 40 + (min1_idx * 50),
                    'y': 1.02,
                    'ay': 1.02,
                    'font_size': 10,
                    'font_family': "Arial, monospace",
                    'xref': f'x{index+1}',
                    'yref': f'y{index+1}',
                    'align': 'left',
                })
                b_matched = True
            if f'y{rev_idx}' in matched_names:
                annotations.append({
                    'showarrow': False,
                    'text': f'y{rev_idx}',
                    'x': 63 + (idx * 50),
                    'ax': 63 + (idx * 50),
                    'y': 1.49,
                    'ay': 1.49,
                    'font_size': 10,
                    'font_family': "Arial, monospace",
                    'xref': f'x{index+1}',
                    'yref': f'y{index+1}',
                    'align': 'left',
                })
                y_matched = True
        if idx > 0:
            if b_matched:
                extra_traces.append(
                    go.Scatter(
                        x=[50 + ((idx-1) * 50), 50 + ((idx-0.5) * 50)],
                        y=[1.1, 1.25],
                        mode='lines',
                        line_color='black',
                        line_width=0.5,
                    )
                )
            if y_matched:
                extra_traces.append(
                    go.Scatter(
                        x=[50 + ((idx-0.5) * 50), 50 + (idx * 50)],
                        y=[1.25, 1.4],
                        mode='lines',
                        line_color='black',
                        line_width=0.5,
                    )
                )
            if b_matched:
                extra_traces.append(
                    go.Scatter(
                        x=[40 + ((idx-1) * 50), 50 + ((idx-1) * 50)],
                        y=[1.1, 1.1],
                        mode='lines',
                        line_color='black',
                        line_width=0.5,
                    )
                )
            if y_matched:
                extra_traces.append(
                    go.Scatter(
                        x=[50 + ((idx) * 50), 60 + (idx * 50)],
                        y=[1.4, 1.4],
                        mode='lines',
                        line_color='black',
                        line_width=0.5,
                    )
                )

    traces = [
        go.Bar(
            x=pred_mzs,
            base='relative',
            alignmentgroup='predicted',
            y=pred_intes,
            name='Predicted',
            marker_color=colours,
            textfont={
                'size': 40,
                'color': 'black'
            },
            marker_line_width=0,
            width=4,
            textposition='outside',
        ),
        go.Bar(
            x=unmatched_peaks['mzs'],
            y=unmatched_peaks['intensities'],
            base='relative',
            alignmentgroup='experimental',
            name='Experimental Not Matched',
            marker_color='lightgrey',
            marker_line_width=0,
            width=4,
        ),
        go.Bar(
            x=matched_peaks['mzs'],
            y=matched_peaks['intensities'],
            base='relative',
            alignmentgroup='experimental',
            name='Experimental Matched',
            marker_color='black',
            marker_line_width=0,
            width=4,
        ),
    ] + extra_traces

    return traces, annotations

def fetch_scan_data(input_df, config, with_charge):
    """ Function to fetch the experimental scan data.

    Parameters
    ----------
    input_df : pd.DataFrame
        The DataFrame of PSMs whose spectra we wish to plot.
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.

    Returns
    -------
    total_scan_df : pd.DataFrame
        A DataFrame of the necessary scan data.
    """
    source_files = input_df[SOURCE_KEY].unique().tolist()
    scan_dfs = []
    if config.combined_scans_file is not None:
        scan_ids = input_df[SCAN_KEY].tolist()
        mgf_filename = f'{config.scans_folder}/{config.combined_scans_file}'
        scan_dfs = [process_mgf_file(
            mgf_filename,
            set(scan_ids),
            config.scan_title_format,
            config.source_files,
            with_charge=with_charge,
        )]
    else:
        for scan_file in source_files:
            scan_ids = input_df[input_df[SOURCE_KEY] == scan_file][SCAN_KEY].tolist()
            if config.scans_format == 'mzML':
                scan_df = process_mzml_file(
                    f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                    set(scan_ids),
                    with_charge=with_charge,
                )
            else:
                mgf_filename = f'{config.scans_folder}/{scan_file}.{config.scans_format}'
                scan_df = process_mgf_file(
                    mgf_filename,
                    set(scan_ids),
                    config.scan_title_format,
                    config.source_files,
                    with_charge=with_charge,
                )
            scan_dfs.append(scan_df)
    total_scan_df = pd.concat(scan_dfs)
    return total_scan_df

def convert_mod_seq_to_ptm_seq(mod_seq):
    """ Function to convert the modified prosit sequence to the PTM seq used elsewhere.

    Parameters
    ----------
    mod_seq : str
        The input sequence for Prosit.

    Returns
    -------
    ptm_seq : str
        The string of digits used to represent any PTMs present.
    """
    ptm_seq = '0.'
    while mod_seq:
        if len(mod_seq) == 1 or mod_seq[1] != '[':
            ptm_seq += '0'
            mod_seq = mod_seq[1:]
        elif mod_seq[1] == '[' and mod_seq[0] == 'C':
            ptm_seq += '2'
            mod_seq = mod_seq[8:]
        else:
            ptm_seq += '1'
            mod_seq = mod_seq[8:]

    ptm_seq += '.0'
    return ptm_seq

def plot_spectra(config):
    """ Function to generate pair plots of selected PSMs (experimental vs. Prosit
        predicted spectra.).

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.
    """
    input_df = pd.read_csv(f'{config.output_folder}/plotData.csv')

    get_charge_from_scan_file = not CHARGE_KEY in input_df.columns
    scan_df = fetch_scan_data(input_df, config, get_charge_from_scan_file)

    input_df = pd.merge(
        input_df,
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )
    n_groups = 1 + (input_df.shape[0] // PLOTS_PER_PAGE)

    input_df['modified_sequence'] = input_df['modifiedSequence'].apply(
        lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
    )
    input_df['precursor_charge'] = input_df[CHARGE_KEY]
    input_df['collision_energy'] = config.collision_energy
    input_df[['modified_sequence', 'precursor_charge', 'collision_energy']].to_csv(
        f'{config.output_folder}/plotInput.csv', index=False,
    )

    predict_spectra(config, pipeline='plotSpectra')
    prosit_df = msp_to_df(f'{config.output_folder}/plotPredictions.msp', 'prosit', None)
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
    input_df['plot_data'] = input_df.apply(
        lambda x : pair_plot(x, config.mz_accuracy, config.mz_units),
        axis=1,
    )

    if SPECTRAL_ANGLE_KEY not in input_df.columns:
        input_df['ptm_seq'] = input_df['modifiedSequence'].apply(
            convert_mod_seq_to_ptm_seq
        )
        input_df = input_df.apply(
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
                config.spectral_predictor,
                spectral_angle_only=True,
            ),
            axis=1,
        )

    titles = []
    for idx in range(input_df.shape[0]):
        seq = input_df[PEPTIDE_KEY].iloc[idx]
        scan_nr = input_df[SCAN_KEY].iloc[idx]
        src = input_df[SOURCE_KEY].iloc[idx]
        spectral_angle = round(input_df[SPECTRAL_ANGLE_KEY].iloc[idx], 2)
        titles.append(
            f'<b>Source</b> {src} <b>Scan</b> {scan_nr}<br><b>Peptide</b> {seq} ' +
            f'<b>Spectral Angle</b> {spectral_angle}<br>'
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

        fig.update_layout(
            width=2100,
            height=n_plot_rows*500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )


        fig.update_xaxes(
            showticklabels=True,
            range=[0, 1500],
            dtick=300,
            linecolor='black',
            linewidth=0.5,
            showgrid=False,
            ticks="outside",
        )

        fig.update_yaxes(
            showline=True,
            linewidth=0.5,
            linecolor='black',
            range=[-1.2, 1.6],
            tickvals = [-1.2+(i*0.4) for i in range(8)],
            ticktext = [round(abs(-1.2+(i*0.4)), 1) for i in range(8)],
            showgrid=False,
            ticks="outside",
        )

        fig.show()
        pio.write_image(
            fig,
            f'{config.output_folder}/spectralPlots{group_idx}.pdf',
            engine='kaleido',
        )

    merger = PdfFileMerger()
    for group_idx in range(n_groups):
        merger.append(
            f'{config.output_folder}/spectralPlots{group_idx}.pdf'
        )
    merger.write(f'{config.output_folder}/spectralPlots.pdf')
    for group_idx in range(n_groups):
        os.remove(
            f'{config.output_folder}/spectralPlots{group_idx}.pdf'
        )
