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
    ION_OFFSET,
    NEUTRAL_LOSSES,
    PEPTIDE_KEY,
    PROTON,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)

from inspire.input.msp import msp_to_df
from inspire.mz_match import get_ion_masses, compute_potential_mws
from inspire.predict_spectra import predict_spectra
from inspire.spectral_features import calculate_spectral_features
from inspire.utils import convert_mod_seq_to_ptm_seq, fetch_scan_data

PLOTS_PER_PAGE = 5
PLOTS_PER_LINE = 1
LOSS_NAMES = ['',  '*', '&#xb0;']

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
                        prec_mz.append(exp_mzs[idx])
                        prec_inte.append(exp_intes[idx]/l2_norm)
                    poss_mzs.append(exp_mzs[idx])
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


def isobar_pair_plot(df_row, mz_accuracy, mz_units, id_grp_name):
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
    if id_grp_name:
        index = (df_row['index'] * 2) + 1
    else:
        index = df_row['index'] * 2

    peptide = df_row[f'{id_grp_name}peptide']
    pep_len = len(peptide)
    pred_intes = [-x for x in df_row[f'{id_grp_name}prositIons'].values()]
    print(f'{id_grp_name}modified_sequence', df_row[f'{id_grp_name}modified_sequence'], df_row[f'{id_grp_name}peptide'])
    pred_mzs, plotting_names = convert_names_and_mzs(
        df_row[f'{id_grp_name}modified_sequence'],
        list(df_row[f'{id_grp_name}prositIons'].keys())
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
        df_row[f'{id_grp_name}ptm_seq'], mz_accuracy, mz_units
    )

    unmatched_peaks = get_unmatched(
        df_row['mzs'],
        df_row['intensities'],
        matched_peaks['mzs'],
        non_prosit_pred_peaks['mzs'],
        prec_peak['mzs'],
        l2_norm,
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

    if non_prosit_pred_peaks['intensities'] or prec_peak['intensities']:
        max_npp = max(non_prosit_pred_peaks['intensities'] + prec_peak['intensities'])
        # if max_npp > 1:
        #     unmatched_peaks['intensities'] = [x/max_npp for x in unmatched_peaks['intensities']]
        #     matched_peaks['intensities'] = [x/max_npp for x in matched_peaks['intensities']]
        #     non_prosit_pred_peaks['intensities'] = [
        #         x/max_npp for x in non_prosit_pred_peaks['intensities']
        #     ]
        #     prec_peak['intensities'] = [x/max_npp for x in prec_peak['intensities']]

    for npp_mz, npp_inte, npp_name in zip(
        non_prosit_pred_peaks['mzs'],
        non_prosit_pred_peaks['intensities'],
        non_prosit_pred_peaks['names'],
    ):
        colours.append(colour)

        if npp_inte > 0.4:
            if npp_name == 'precursor':
                font_colour = 'plum'
            else:
                font_colour = 'forestgreen'
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
            x=non_prosit_pred_peaks['mzs'],
            y=non_prosit_pred_peaks['intensities'],
            base='relative',
            alignmentgroup='experimental',
            name='Possible Ion Not Predicted by Prosit',
            marker_color='forestgreen',
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
    ] + extra_traces

    return traces, annotations


def plot_isobars(config):
    """ Function to generate pair plots of selected PSMs (experimental vs. Prosit
        predicted spectra.).

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.
    """
    add_legend(config.output_folder, config.experiment_title)

    input_df = pd.read_csv(f'{config.output_folder}/isobarData.csv')

    get_charge_from_scan_file = not CHARGE_KEY in input_df.columns
    scan_df = fetch_scan_data(input_df, config, get_charge_from_scan_file)

    input_df = pd.merge(
        input_df,
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )

    input_df['pepLen'] = input_df['peptide'].apply(len)
    input_df = input_df[(input_df['pepLen'] > 6) & (input_df['pepLen'] < 31)]
    input_df = input_df[input_df['charge'] < 7]

    n_groups = (input_df.shape[0] // PLOTS_PER_PAGE)
    if input_df.shape[0] % PLOTS_PER_PAGE:
        n_groups += 1

    for idx, id_grp_name in enumerate(['', 'isobar']):
        input_df[f'{id_grp_name}modified_sequence'] = input_df[f'{id_grp_name}modifiedSequence'].apply(
            lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
        )
        input_df['precursor_charge'] = input_df[CHARGE_KEY]

        if 'collisionEnergy' in input_df.columns:
            input_df = input_df.rename(columns={'collisionEnergy': 'collision_energy'})
        else:
            input_df['collision_energy'] = config.collision_energy

        input_df[[f'{id_grp_name}modified_sequence', 'precursor_charge', 'collision_energy']].rename(
            columns={
                f'{id_grp_name}modified_sequence': 'modified_sequence',
            }
        ).to_csv(
            f'{config.output_folder}/plot{id_grp_name}Input.csv', index=False,
        )

        predict_spectra(config, pipeline=f'plot{id_grp_name}Spectra')
        prosit_df = msp_to_df(
            f'{config.output_folder}/plot{id_grp_name}Predictions.msp', 'prosit', None,
        ).rename(columns={'modified_sequence': f'{id_grp_name}modified_sequence'})
        prosit_df = prosit_df.drop_duplicates(subset=[f'{id_grp_name}modified_sequence', CHARGE_KEY])
        prosit_df = prosit_df.rename(columns={
                'prositIons': f'{id_grp_name}prositIons', 'modified_sequence': f'{id_grp_name}modified_sequence'}
        )


        input_df = pd.merge(
            input_df,
            prosit_df[[CHARGE_KEY, f'{id_grp_name}prositIons', f'{id_grp_name}modified_sequence']],
            how='inner',
            on=[f'{id_grp_name}modified_sequence', CHARGE_KEY]
        )
        print(input_df.columns)

    input_df = input_df.sort_values(by='deltaRT')
    input_df = input_df.reset_index(drop=True)
    input_df['group'] = input_df.index // PLOTS_PER_PAGE
    input_df['index'] = input_df.index % PLOTS_PER_PAGE
    for idx, id_grp_name in enumerate(['', 'isobar']):
        input_df[f'{id_grp_name}ptm_seq'] = input_df[f'{id_grp_name}modifiedSequence'].apply(
            convert_mod_seq_to_ptm_seq
        )
        input_df[f'{id_grp_name}plot_data'] = input_df.apply(
            lambda x : isobar_pair_plot(x, config.mz_accuracy, config.mz_units, id_grp_name),
            axis=1,
        )

    if SPECTRAL_ANGLE_KEY not in input_df.columns:
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
                minimal_features=True,
            ),
            axis=1,
        )

    titles = []
    for idx in range(input_df.shape[0]):
        seq = input_df[PEPTIDE_KEY].iloc[idx]
        scan_nr = input_df[SCAN_KEY].iloc[idx]
        charge = input_df[CHARGE_KEY].iloc[idx]
        isobar_seq = input_df['isobarpeptide'].iloc[idx]
        isobar_irt = round(input_df['isobardeltaRT'].iloc[idx],2)
        isobar_sa = round(input_df['isobarspectralAngle'].iloc[idx],2)
        isobar_spear = round(input_df['isobarspearmanR'].iloc[idx],2)
        spear = round(input_df['spearmanR'].iloc[idx],2)
        src = input_df[SOURCE_KEY].iloc[idx].split('UnLabeled_')[-1]
        spectral_angle = round(input_df[SPECTRAL_ANGLE_KEY].iloc[idx], 2)
        delta_rt = round(input_df['deltaRT'].iloc[idx],2)
        if isobar_irt > 4.25:
            isobar_irt = f'<span style="color:red">{isobar_irt}</span>'
        if delta_rt > 4.25:
            delta_rt = f'<span style="color:red">{delta_rt}</span>'
        titles.append(
            f'<b>Source</b> {src} <b>Scan</b> {scan_nr}<br><b>Peptide</b> {seq} ' +
            f'<b>Charge</b> {charge} <b>Spectral Angle</b> {spectral_angle}<br>' +
            f'<b>Spearman Correlation</b>  {spear} <b>iRT Error:</b> {delta_rt}'
        )
        titles.append(
            f'<b>Source</b> {src} <b>Scan</b> {scan_nr}<br><b>Peptide</b> {isobar_seq} ' +
            f'<b>Charge</b> {charge} <b>Spectral Angle</b> {isobar_sa}<br>' +
            f'<b>Spearman Correlation</b> {isobar_spear} <b>iRT Error:</b> {isobar_irt}'
        )


    for group_idx in range(n_groups):

        start_idx = PLOTS_PER_PAGE*2*group_idx
        sub_df = input_df[input_df['group'] == group_idx]
        n_plots = sub_df.shape[0]*2

        fig = make_subplots(
            rows=PLOTS_PER_PAGE,
            cols=2,
            subplot_titles = titles[start_idx:start_idx+n_plots],
        )
        for idx in range(1, (PLOTS_PER_PAGE*2)+1):
            fig.update_layout(
                {
                    f'xaxis{idx}':{'title_text': 'm/z'},
                    f'yaxis{idx}':{'title_text': 'L<sup>2</sup> Normalized Intensity'},
                }
            )

        plot_data = sub_df['plot_data'].tolist()
        for idx, (traces, annotations) in enumerate(plot_data):
            print(idx)
            for trace in traces:
                fig.add_trace(
                    trace,
                    row=idx+1,
                    col=1,
                )

            fig.layout['annotations'] += tuple(annotations)
        
        plot_data = sub_df['isobarplot_data'].tolist()
        for idx, (traces, annotations) in enumerate(plot_data):
            for trace in traces:
                fig.add_trace(
                    trace,
                    row=idx+1,
                    col=2,
                )

            fig.layout['annotations'] += tuple(annotations)

        for idx in range(sub_df.shape[0]):
            for col in [1, 2]:
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1500],
                        y=[0, 0],
                        mode='lines',
                        line={'width':0.5, 'color':'black'},
                    ),
                    row=1 + idx,
                    col=col,
                )

        fig.update_layout(
            width=1400,
            height=PLOTS_PER_PAGE*500,
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

        # if not config.silent_execution:
        #     fig.show()

        pio.write_image(
            fig,
            f'{config.output_folder}/isobarPlots{group_idx}.pdf',
            engine='kaleido',
        )

    merger = PdfFileMerger()
    merger.append(
        f'{config.output_folder}/isobarPlots_legend.pdf'
    )
    for group_idx in range(n_groups):
        merger.append(
            f'{config.output_folder}/isobarPlots{group_idx}.pdf'
        )
    merger.write(f'{config.output_folder}/isobarPlots.pdf')
    for group_idx in range(n_groups):
        os.remove(
            f'{config.output_folder}/isobarPlots{group_idx}.pdf'
        )
    os.remove(
        f'{config.output_folder}/isobarPlots_legend.pdf'
    )


def add_legend(output_folder, experiment_title):
    """ Function for adding a legend to the start of the spectrolPlots.pdf file.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE outputs are written.
    experiment_title : str
        The title of the experiment.
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
            marker={'size':22, 'color':'forestgreen'},
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
            marker={'size':22, 'color':'lightgrey'},
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
        subplot_titles=['', 'Experimental Spectrum Colour Code:', 'Prosit Spectrum Colour Code:', 'Additional Notes:'],
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
        linecolor='forestgreen',
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

    colour_key_fig.update_layout(
        title_text=f'<b>inSPIRE isobaric peptide comparisons for {experiment_title}</b>',
        title_x=0.5,
        title_font_size=30,
        title_font_family='Helvetica',
    )

    pio.write_image(
        colour_key_fig,
        f'{output_folder}/isobarPlots_legend.pdf',
        engine='kaleido',
    )
