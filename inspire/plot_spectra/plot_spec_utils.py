""" Script for shared utilities when plotting MS2 spectra.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from inspire.constants import (
    ION_OFFSET,
    KNOWN_PTM_WEIGHTS,
    LOSS_NAMES,
    NEUTRAL_LOSSES,
    PROTON,
)
from inspire.mz_match import get_ion_masses, compute_potential_mws


LOSS_NAMES = ['',  '*', '&#xb0;']


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

    if len(matched_intes) > 2:
        l2_norm = np.linalg.norm(np.array(matched_intes), ord=2)
    else:
        l2_norm = np.max(exp_intes)

    matched_peaks = {
        'mzs': matched_mzs,
        'intensities': [x/l2_norm for x in matched_intes],
    }

    return matched_peaks, l2_norm, matched_pred_mzs, matched_names


def get_npp_ions(
        exp_mzs,
        exp_intes,
        matched_mzs,
        l2_norm,
        peptide,
        charge,
        ptm_id_weights,
        modifications,
        mz_accuracy,
        mz_units,
    ):
    """ Function to get the potential non-prosit-predicted ions.

    Parameters
    ----------
    exp_mzs : list of float
        A list of m/z values observed in the experimental spectrum.
    exp_intes : list of int
        A list of the intensities observed in the experimental spectrum.
    matched_mzs : list of float
        A list of the m/z values already matched to the Prosit predictions.
    l2_norm : float
        The L^2 norm of the Prosit matched ions in the experimental spectrum.
    peptide : string
        The peptide sequence of the precursor.
    charge : int
        The charge state of the peptide.
    ptm_id_weights : dict
        A dictionary mapping inSPIRE PTM IDs to their molecular weights.
    modifications : str
        The modifications of the peptide written as a string of inSPIRE IDs.
    mz_accuracy : float
        The accuracy of the mass spectrometer.
    mz_units : str
        Da or ppm

    Returns
    -------
    npp_ions : dict
        A dictionary of the m/z values, intensities, and names of non-prosit predicted ions.
    prec_ion : dict
        A dictionary of the m/z and intensity value of the precursor ion if observed in the MS2.
    """
    poss_mzs = []
    poss_intes = []
    poss_names = []
    prec_mz = []
    prec_inte = []
    all_mzs, names = generate_all_mzs(peptide, charge, ptm_id_weights, modifications)

    for idx, e_mz in enumerate(exp_mzs):
        if e_mz not in matched_mzs:
            if mz_units == 'ppm':
                mz_err = e_mz*mz_accuracy*(10**-6)
            else:
                mz_err = mz_accuracy

            for alt_idx, possible_mz in enumerate(all_mzs):
                if abs(e_mz-possible_mz) < mz_err:
                    if alt_idx == 0:
                        prec_mz.append(e_mz)
                        prec_inte.append(exp_intes[idx]/l2_norm)
                    poss_mzs.append(e_mz)
                    poss_intes.append(exp_intes[idx]/l2_norm)
                    poss_names.append(names[alt_idx])

    return {
        'mzs': poss_mzs,
        'intensities': poss_intes,
        'names': poss_names,
    }, {
        'mzs': prec_mz,
        'intensities': prec_inte,
    }

def generate_all_mzs(sequence, prec_charge, ptm_id_weights, modifications=None):
    """ Function to the get mz's for all the ions predicted by prosit.

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.

    Returns
    -------
    all_possible_ions : dict
        A dictionary of all the mzs of all possible b and y ions
        that could be produced.
    """
    sub_seq_mass, total_residue_mass = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=False,
        ptm_id_weights=ptm_id_weights
    )

    rev_sub_seq_mass, _ = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=True,
        ptm_id_weights=ptm_id_weights
    )

    # b- and y-ions for each of the charge options
    possible_ions = {}
    possible_ions['b'] = ION_OFFSET['b'] + sub_seq_mass
    possible_ions['y'] = ION_OFFSET['y'] + rev_sub_seq_mass
    possible_ions['a'] = ION_OFFSET['a'] + sub_seq_mass

    all_mzs = [(total_residue_mass+(prec_charge*PROTON))/prec_charge]
    names = ['precursor']

    for ion_type in 'bya':
        loss_list = [0.0] + NEUTRAL_LOSSES

        for frag_idx, weight in enumerate(possible_ions[ion_type]):
            for loss_idx, loss in enumerate(loss_list):
                for charge in range(1, prec_charge+1):
                    all_mzs.append(
                        ((weight-loss) + (PROTON*charge))/charge
                    )
                    charge_name = '+'*charge
                    names.append(
                        f'{ion_type}{frag_idx+1}{LOSS_NAMES[loss_idx]}{charge_name}'
                    )

    return all_mzs, names

def get_unmatched(exp_mzs, exp_intes, matched_mzs, npp_mzs, prec_mzs, l2_norm):
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
        if e_mz not in matched_mzs and e_mz not in npp_mzs and e_mz not in prec_mzs:
            unmatched_mzs.append(e_mz)
            unmatched_intes.append(exp_intes[idx]/l2_norm)

    return {
        'mzs': unmatched_mzs,
        'intensities': unmatched_intes,
    }


def get_plot_details(
        index, pred_mzs, matched_p_mz, pred_intes, plotting_names,
        peptide, matched_names, non_prosit_pred_peaks
    ):
    """ Function to get the traces, colours and annotations shown in the MS2 spectral
        plot.
    """
    pep_len = len(peptide)
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

    if pep_len < 20:
        annot_height = 1.5
    else:
        annot_height = 0.7

    extra_traces = []
    annotations.append({
        'showarrow': False,
            'text': 'Experimental Spectrum',
            'x': 1300,
            'ax': 1300,
            'y': annot_height,
            'ay': annot_height,
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
            npp_names = non_prosit_pred_peaks['names']
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
            for name in npp_names:
                if name == 'precursor':
                    continue
                frag_name = name.split('&')[0].split('*')[0].split('+')[0]
                frag_type = frag_name[0]
                frag_idx = int(frag_name[1:])
                if rev_idx == frag_idx:
                    if frag_type == 'y':
                        if not y_matched:
                            annotations.append({
                                'showarrow': False,
                                'text': name.split('+')[0],
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
                if idx == frag_idx:
                    if frag_type != 'y':
                        if not b_matched:
                            annotations.append({
                                'showarrow': False,
                                'text': name.split('+')[0],
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

    return annotations, colours, extra_traces

def create_traces(
        pred_mzs, pred_intes, unmatched_peaks, non_prosit_pred_peaks,
        prec_peak, matched_peaks, colours,
    ):
    """ Function to create the required bar traces for MS2 spectral plotting.
    """
    return [
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
            marker_color='#E8E8E8',
            marker_line_width=0,
            width=4,
        ),
        go.Bar(
            x=non_prosit_pred_peaks['mzs'],
            y=non_prosit_pred_peaks['intensities'],
            base='relative',
            alignmentgroup='experimental',
            name='Possible Ion Not Predicted by Prosit',
            marker_color='darkseagreen',
            marker_line_width=0,
            width=4,
        ),
        go.Bar(
            x=prec_peak['mzs'],
            y=prec_peak['intensities'],
            base='relative',
            alignmentgroup='experimental',
            name='Precursor',
            marker_color='plum',
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
    ]



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



def add_legend(output_folder, experiment_title, plot_type):
    """ Function for adding a legend to the start of the spectrolPlots.pdf file.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE outputs are written.
    experiment_title : str
        The title of the experiment.
    plot_type : str
        standard or isobar
    """
    legend_traces = [
        go.Scatter(
            x=[1.0],
            y=[7.5],
            marker={'size':22, 'color':'black'},
            text='    Experimental peak matched to a Prosit predicted peak.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
        go.Scatter(
            x=[1.0],
            y=[5.0],
            marker={'size':22, 'color':'darkseagreen'},
            text='    Possible ion unknown to Prosit.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
        go.Scatter(
            x=[1.0],
            y=[2.5],
            marker={'size':22, 'color':'plum'},
            text='    Precursor matched peak.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
        go.Scatter(
            x=[1.0],
            y=[0.0],
            marker={'size':22, 'color':'#E8E8E8'},
            text='    Experimental peak not matched to any potential ion.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
        go.Scatter(
            x=[1.0],
            y=[7.5],
            marker={'size':22, 'color':'#0095A8'},
            text='    Prosit predicted peak matched to experimental spectrum.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
        go.Scatter(
            x=[1.0],
            y=[5.0],
            marker={'size':22, 'color':'#FF7043'},
            text='    Prosit predicted peak not matched to experimental spectrum.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='markers+text',
        ),
    ]

    colour_key_fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=[
            '',
            'Experimental Spectrum Colour Code:',
            'Prosit Spectrum Colour Code:',
            'Additional Notes:'
        ],
    )
    for trace in legend_traces[:4]:
        colour_key_fig.add_trace(trace, row=2, col=1)
    for trace in legend_traces[4:]:
        colour_key_fig.add_trace(trace, row=3, col=1)

    colour_key_fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[7.5],
            text='&#xb0; indicates an ion with loss of H<sub>2</sub>O.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='text',
        ),
        row=4,
        col=1,
    )
    colour_key_fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[5.0],
            text='* indicates an ion with loss of NH<sub>3</sub>.',
            textposition='middle right',
            textfont_size=22,
            textfont_family='Helvetica',
            mode='text',
        ),
        row=4,
        col=1,
    )

    colour_key_fig.update_layout(
        width=2100,
        height=2500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )


    colour_key_fig.update_xaxes(
        showticklabels=True,
        range=[0, 4],
        dtick=300,
        linecolor='black',
        linewidth=0.5,
        showgrid=False,
        ticks="outside",
        zeroline=False, # thick line at x=0
        visible=False
    )

    colour_key_fig.update_yaxes(
        showline=True,
        linewidth=0.5,
        linecolor='black',
        range=[-2, 10],
        tickvals = [-1.2+(i*0.4) for i in range(8)],
        ticktext = [round(abs(-1.2+(i*0.4)), 1) for i in range(8)],
        showgrid=False,
        ticks="outside",
        zeroline=False, # thick line at x=0
        visible=False
    )

    colour_key_fig.update_annotations(
        {
            'font_size': 26,
            'font_family': "Helvetica",
        }
    )

    if plot_type == 'standard':
        title_text = f'<b>inSPIRE Spectral Plotting for {experiment_title}</b>'
    else:
        title_text = f'<b>inSPIRE Spectral Plotting for {experiment_title}</b>'

    colour_key_fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        title_font_size=30,
        title_font_family='Helvetica',
    )

    if plot_type == 'standard':
        out_path = f'{output_folder}/spectralPlots_legend.pdf'
    else:
        out_path = f'{output_folder}/spectralPlots_legend.pdf'

    pio.write_image(
        colour_key_fig,
        out_path,
        engine='kaleido',
    )


def update_fig_layout(fig, n_plot_rows):
    """ Function to update the figure layout.
    """
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

    return fig
