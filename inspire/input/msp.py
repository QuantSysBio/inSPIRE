""" Functions for reading in Prosit predicted spectra in msp format.
"""
import re

import numpy as np
import polars as pl

from inspire.constants import (
    CHARGE_KEY,
    MS2PIP_NAME_MAPPINGS,
    OXIDATION_PREFIX,
    OXIDATION_PREFIX_LEN,
    PROSIT_IONS_KEY,
    PROSIT_INTES_KEY,
    PROSIT_SEQ_KEY,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
)

def msp_process_sequence_and_charge(line):
    """ Function to extract the name and charge of a sample from
        the relevant line of an msp file.

    Parameters
    ----------
    line : str
        The latest line read from the msp file.

    Returns
    -------
    sequence : str
        The peptide sequence.
    charge : int
        The charge of the sequence.
    """
    regex_match = re.match(r'Name: (.*?)/(\d+)\n', line)
    sequence = regex_match.group(1)
    charge = int(regex_match.group(2))
    return sequence, charge

def process_intensities(intensities):
    """ Function to L2 normalise intensity array.

    Parameters
    ----------
    intensities : list of float
        A list of the intensities of peaks found in the spectrum.

    Returns
    -------
    normed_intensities : list of float
        A list of normalised intensities.
    """
    l2_norm = np.linalg.norm(np.array(intensities), ord=2)
    normed_intensities = [z/l2_norm for z in intensities]
    return normed_intensities

def msp_process_peaks(line, msp_file):
    """ Function to extract the ms2 spectrum of a sample from
        the relevant lines of an msp file.

    Parameters
    ----------
    line : str
        The latest line read from the msp file.
    msp_file : file
        The full msp file.

    Returns
    -------
    ions : list of str
        A list of the ions predicted
    normed_intensities : list of float
        A list of ions normed intensities.
    """
    regex_match = re.match(r'Num peaks: (\d+)\n', line)
    n_peaks = int(regex_match.group(1))

    ions = []
    intensities = []
    for _ in range(n_peaks):
        line = msp_file.readline()
        peak_data = line.strip().split('\t')
        intensity = float(peak_data[1])
        regex_match = re.match(r'(.*?)/(.*?)ppm', peak_data[2].strip('"'))
        ion = regex_match.group(1).strip(')').strip('(')
        if intensity > 0:
            ions.append(ion)
            intensities.append(intensity)

    normed_intensities = process_intensities(intensities)

    return ions, normed_intensities

def ms2pip_process_peaks(line, msp_file):
    """ Function to extract the ms2 spectrum of a sample from
        the relevant lines of an msp file.

    Parameters
    ----------
    line : str
        The latest line read from the msp file.
    msp_file : file
        The full msp file.

    Returns
    -------
    normed_intensities : dict
        A dictionary of ion names mapped to their normed intensity.
    """
    regex_match = re.match(r'Num peaks: (\d+)\n', line)
    n_peaks = int(regex_match.group(1)) - 1

    ions = []
    intensities = []
    for _ in range(n_peaks):
        line = msp_file.readline()

        peak_data = line.strip().split('\t')

        if len(peak_data) == 1:
            break
        intensity = float(peak_data[1])
        ion = peak_data[2].strip('"')
        if intensity > 0:
            ions.append(ion)
            intensities.append(intensity)

    normed_intensities = process_intensities(intensities)

    return ions, normed_intensities

def get_ms2pip_mods(line, sequence, mod_id_mappings):
    """ Function to fetch the modifications from an MS2PIP modification file.

    Parameters
    ----------
    line : str
        A line from the msp file.
    sequence : str
        The peptide sequence.
    mod_id_mappings : dict
        Dictionary mapping modification names to their single digit code.

    Returns
    -------
    ptm_seq : str
        The PTM sequence that encodes all modifications in the data.
    """
    regex_match = re.match(
        r'Comment: Mods=(.*?) Parent=(.*?)',
        line,
    )
    modifications = regex_match.group(1)
    mod_list = modifications.split('/')[1:]

    if not mod_list:
        return None
    ptm_seq = '0'*(len(sequence)+2)

    for entry in mod_list:
        ptm_data = entry.split(',')
        ptm_name = ptm_data[2]
        ptm_loc = int(ptm_data[0])
        ptm_loc += 1
        ptm_seq = ptm_seq[:ptm_loc] + str(mod_id_mappings[ptm_name]) + ptm_seq[ptm_loc+1:]

    ptm_seq = ptm_seq[0] + '.' + ptm_seq[1:-1] + '.' + ptm_seq[-1]

    return ptm_seq

def process_prosit_comment(line, sequence):
    """ Function to extract data the comments on a sample from
        the relevant line of an msp file of Prosit output.

    Parameters
    ----------
    line : str
        The latest comment line read from the msp file.
    sequence : str
        The latest peptide sequence read from the msp file.

    Returns
    -------
    sequence : str
        The sequence with any modifications added.
    irt : float
        The iRT value predicted by Prosit.
    collision_energy : int
        The collision energy setting used by Prosit.
    """
    regex_match = re.match(
        r'(.*?) Collision_energy=(.*?) Mods=(.*?) ModString=(.*?)//(.*?)/(.*?)',
        line
    )
    collision_energy = int(float(regex_match.group(2)))
    mods = regex_match.group(5)
    if 'iRT=' in line:
        irt = float(line.split('iRT=')[-1].split(' ')[0])
    else:
        irt = None

    if mods:
        mods_list = mods.split('; ')
        mod_seq = ''
        previous_ind = 0
        for mod in mods_list:
            if mod.startswith(OXIDATION_PREFIX):
                pos = int(mod[OXIDATION_PREFIX_LEN:]) - 1
                mod_seq += sequence[previous_ind:pos]
                mod_seq += "(ox)"
                previous_ind = pos
        mod_seq += sequence[previous_ind:]
        return mod_seq, irt, collision_energy

    return sequence, irt, collision_energy

def msp_to_df(msp_filename, msp_format, mods_df):
    """ Function to process an msp file and extract relevant information
        for training into csv format (tab separated).

    Parameters
    ----------
    msp_filename : str
        The location where the msp file is written.
    msp_format : str
        Either prosit or ms2pip. The predictor which wrote the msp file.
    mods_df : pd.DataFrame
        Small DataFrame detailing the PTMs considered.

    Returns
    -------
    ion_df : pd.DataFrame
        The DataFrame with the spectra found in the msp file.
    """
    if msp_format == 'ms2pip':
        mods_df['ms2pipName'] = mods_df[PTM_NAME_KEY].apply(
            lambda x : MS2PIP_NAME_MAPPINGS[x]
        )
        mod_id_mappings = dict(zip(mods_df['ms2pipName'].tolist(), mods_df[PTM_ID_KEY].tolist(), ))

    with open(msp_filename, 'r', encoding='UTF-8') as msp_file:
        peptides = []
        charges = []
        ion_intensities = []
        modified_sequences = []
        irts = []
        collision_energies = []
        ptm_seqs = []
        ion_names = []
        named_ions = None
        bufsize = 65536
        while True:
            lines = msp_file.readlines(bufsize)
            if not lines:
                break
            for line in lines:
                if line.startswith('Name: '):
                    if named_ions is not None:
                        normed_intensities = process_intensities(intensities)
                        ion_intensities.append(normed_intensities)
                        ion_names.append(named_ions)
                        peptides.append(sequence)
                        charges.append(charge)
                        modified_sequences.append(modified_sequence)
                        irts.append(irt)
                        collision_energies.append(col_e)

                    sequence, charge = msp_process_sequence_and_charge(line)
                    named_ions = []
                    intensities = []
                    active = False
                if line.startswith('Comment: '):
                    modified_sequence, irt, col_e = process_prosit_comment(line, sequence)

                if line.startswith('Num peaks: '):
                    active = True
                    continue
                if active:
                    peak_data = line.strip().split('\t')
                    intensity = float(peak_data[1])
                    regex_match = re.match(r'(.*?)/(.*?)ppm', peak_data[2].strip('"'))
                    ion = regex_match.group(1).strip(')').strip('(')
                    if intensity > 0:
                        named_ions.append(ion)
                        intensities.append(intensity)


    if named_ions is not None:
        normed_intensities = process_intensities(intensities)
        ion_intensities.append(normed_intensities)
        ion_names.append(named_ions)
        peptides.append(sequence)
        charges.append(charge)
        modified_sequences.append(modified_sequence)
        irts.append(irt)
        collision_energies.append(col_e)

    df_data = {
        CHARGE_KEY: charges,
        PROSIT_IONS_KEY: ion_names,
        PROSIT_INTES_KEY: ion_intensities,
    }
    if msp_format == 'prosit':
        df_data[PROSIT_SEQ_KEY] = modified_sequences
        df_data['iRT'] = irts
        df_data['collisionEnergy'] = collision_energies
    else:
        df_data['peptide'] = modified_sequences
        df_data[PTM_SEQ_KEY] = ptm_seqs

    msp_df = pl.DataFrame(df_data)

    return msp_df
