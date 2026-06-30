""" Suite of function for preparing input to external tools, specifically
    Prosit and netMHCpan input.
"""

import os

import pandas as pd

from inspire.constants import (
    CHARGE_KEY,
    ENDC_TEXT,
    MS2PIP_NAME_MAPPINGS,
    OKCYAN_TEXT,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
)
from inspire.input.search_results import generic_read_df
from inspire.utils import get_ox_flag, permute_ptms, permute_seq

def create_prosit_mod_seq(sequence, modifications, ox_marker):
    """ Function to add any required oxidation flags to the peptide sequence.

    Parameters
    ----------
    sequence : str
        The original unmodified sequence.
    modifications : str or None
        The ptms associated with this sequence.
    ox_marker : int
        The ptm marker that indicates oxidation.

    Returns
    -------
    mod_seq : str
        The sequence with modifications added.
    """
    if (
        modifications is None or
        not modifications or
        modifications == 'nan' or
        not isinstance(modifications, str)
    ):
        return sequence

    ptms_list = modifications.split(".")
    mods_list = [int(mod) for mod in ptms_list[1]]
    if ox_marker in mods_list:
        mod_seq = ""
        previous_ind = 0
        for idx, mod in enumerate(mods_list):
            if mod == ox_marker:
                mod_seq += sequence[previous_ind:idx+1]
                mod_seq += '(ox)'
                previous_ind = idx+1
        mod_seq += sequence[previous_ind:]
        return mod_seq
    return sequence

def write_prosit_input_df(
        search_df,
        mods_df,
        config,
        collision_energy,
        filename,
        overwrite=True,
    ):
    """ Function to write prosit input sequences.

    Parameters
    ----------
    search_df : pd.DataFrame
        A DataFrame containing search results.
    mods_df : pd.DataFrame
        A DataFrame containing variable ptm data.
    output_folder : str
        The folder where output will be written.
    collision_energy : int
        The collision energy setting of the mass spectrometer.
    """
    # Create modified sequence.
    ox_flag = get_ox_flag(mods_df)
    search_df['modified_sequence'] = search_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
        lambda x : create_prosit_mod_seq(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], ox_flag),
        axis=1
    )

    # Write sequences in Prosit's input format.
    prosit_df = search_df[['modified_sequence', CHARGE_KEY]].rename(
        columns={CHARGE_KEY: 'precursor_charge'}
    )
    prosit_df['collision_energy'] = collision_energy
    prosit_df = prosit_df.drop_duplicates()
    if overwrite:
        prosit_df.to_csv(
            f'{config.output_folder}/{filename}.csv',
            index=False,
        )
    else:
        prosit_df.to_csv(
            f'{config.output_folder}/{filename}.csv',
            index=False,
            mode='a',
            header=False,
        )

    if config.delta_method == 'bruteForce' and filename == 'prositInput':
        prosit_df['mSeq'] = prosit_df['modified_sequence'].apply(
            lambda x : x.replace('M(ox)', 'm')
        )
        prosit_df['permSeqs'] = prosit_df['mSeq'].apply(permute_seq)
        prosit_df = prosit_df[prosit_df["permSeqs"].astype(bool)]
        prosit_df = prosit_df.explode('permSeqs')

        prosit_df = prosit_df.drop_duplicates(
            subset=['permSeqs', 'precursor_charge']
        )
        prosit_df['modified_sequence'] = prosit_df['permSeqs'].apply(
            lambda x : x.replace('m', 'M(ox)')
        )
        prosit_df[['modified_sequence', 'precursor_charge', 'collision_energy']].to_csv(
            f'{config.output_folder}/deltaInput.csv',
            index=False,
        )


def prepare_for_spectral_prediction(config):
    """ Function to prepare sequences for Prosit input.

    Parameters
    ----------
    configuration : inspire.config.Config
        The configuration settings for the pipeline.
    """
    target_df, mods_df = generic_read_df(config)

    if config.spectral_predictor == 'prosit':
        write_prosit_input_df(
            target_df,
            mods_df,
            config,
            config.collision_energy,
            'prositInput'
        )
        print(
            OKCYAN_TEXT +
            '\tFormatted Prosit input written.' +
            ENDC_TEXT
        )
    else:
        write_ms2pip_input_df(
            target_df,
            mods_df,
            config.output_folder,
            'ms2pipInput',
            config.delta_method,
        )
        print(
            OKCYAN_TEXT +
            '\tFormatted MS2PIP input written.' +
            ENDC_TEXT
        )

def get_ms2pip_mods(ptm_seq, mod_id_mappings):
    """ Function to format PTMs for MS2PIP input.

    Parameters
    ----------
    ptm_seq : str or NaN
        The PTM sequence formatted by inSPIRE.
    mod_id_mappings : dict
        A dictionary mapping PTM IDs to the weight of the PTM.

    Return
    ------
    ms2pip_mods : str
        The PTMs observed formatted for MS2PIP input.
    """
    if not isinstance(ptm_seq, str):
        return '-'
    ptm_seq = ptm_seq.replace('.', '')
    mods = []
    for idx, char in enumerate(ptm_seq):
        if char != '0':
            mods.append(f'{idx}|{mod_id_mappings[int(char)]}')
    if mods:
        return '|'.join(mods)
    return '-'

def write_ms2pip_input_df(target_df, mods_df, output_folder, output_name, delta_method):
    """ Function to write input sequences for ms2pip.

    Parameters
    ----------
    target_df : pd.DataFrame
        A DataFrame of PSMs for which we require MS2PIP predictions.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs in the data.
    output_folder : str
        The folder where all inSPIRE output is written.
    output_name : str
        The name used in the MS2PIP input file.
    delta_method : str
        The method used to calculate Prosit Delta values.
    """
    mods_df['ms2pipName'] = mods_df[PTM_NAME_KEY].apply(
        lambda x : MS2PIP_NAME_MAPPINGS[x]
    )
    mod_id_mappings = dict(zip(mods_df[PTM_ID_KEY].tolist(), mods_df['ms2pipName'].tolist()))

    target_df['spec_id'] = target_df.index
    target_df['spec_id'] = target_df['spec_id'].apply(lambda x : f'peptide_{x}')
    target_df['modifications'] = target_df[PTM_SEQ_KEY].apply(
        lambda x : get_ms2pip_mods(x, mod_id_mappings)
    )
    ms2pip_input_df = target_df[['spec_id', 'modifications', 'peptide', 'charge']]
    ms2pip_input_df = ms2pip_input_df.drop_duplicates(
        subset=['modifications', 'peptide', 'charge']
    )

    ms2pip_input_df.to_csv(
        f'{output_folder}/{output_name}.preprec', sep=' ', index=False,
    )

    if delta_method == 'bruteForce':
        target_df['permSeqs'] = target_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
            lambda x : list(
                zip(permute_seq(x[PEPTIDE_KEY]), permute_ptms(x[PEPTIDE_KEY], x[PTM_SEQ_KEY]))
            ),
            axis=1
        )
        target_df = target_df[target_df["permSeqs"].astype(bool)]
        target_df = target_df.explode('permSeqs')

        target_df[[PEPTIDE_KEY, PTM_SEQ_KEY]] = pd.DataFrame(
            target_df['permSeqs'].tolist(), index=target_df.index
        )
        target_df['modifications'] = target_df[PTM_SEQ_KEY].apply(
            lambda x : get_ms2pip_mods(x, mod_id_mappings)
        )

        target_df = target_df.drop_duplicates(
            subset=['peptide', 'modifications', 'charge']
        )
        target_df = target_df.reset_index(drop=True)
        target_df['spec_id'] = target_df.index

        target_df[['spec_id', 'modifications', 'peptide', 'charge']].to_csv(
            f'{output_folder}/deltaInput.preprec',
            sep=' ',
            index=False,
        )

def prepare_for_mhcpan(config):
    """ Function to prepare sequences for NetMHCpan input.

    Parameters
    ----------
    configuration : inspire.config.Config
        The configuration settings for the pipeline.
    """
    if config.use_binding_affinity not in ('asValidation', 'asFeature'):
        return

    target_df, _ = generic_read_df(config)

    peptide_lengths = target_df[PEPTIDE_KEY].apply(len)
    unique_pep_lens = peptide_lengths.unique().tolist()

    if not os.path.exists(f'{config.output_folder}/mhcpan'):
        os.makedirs(f'{config.output_folder}/mhcpan')

    for length in unique_pep_lens:
        len_df = target_df[peptide_lengths == length]
        peptides = len_df[[PEPTIDE_KEY]].drop_duplicates()
        peptides.to_csv(
            f'{config.output_folder}/mhcpan/inputLen{length}.txt',
            header=False,
            index=False,
        )
    print(
        OKCYAN_TEXT +
        '\tFormatted NetMHCpan input written.' +
        ENDC_TEXT
    )
