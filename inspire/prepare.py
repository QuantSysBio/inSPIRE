""" Suite of function for preparing input to external tools, specifically
    Prosit and netMHCpan input.
"""

import os

from inspire.constants import (
    CHARGE_KEY,
    ENDC_TEXT,
    OKCYAN_TEXT,
    PEPTIDE_KEY,
    PTM_SEQ_KEY,
)
from inspire.input.search_results import generic_read_df
from inspire.utils import get_ox_flag

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
        output_folder,
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
            f'{output_folder}/{filename}.csv',
            index=False,
        )
    else:
        prosit_df.to_csv(
            f'{output_folder}/{filename}.csv',
            index=False,
            mode='a',
            header=False,
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
            config.output_folder,
            config.collision_energy,
            'prositInput'
        )
        print(
            OKCYAN_TEXT +
            '\tFormatted Prosit input written.' +
            ENDC_TEXT
        )
    else:
        print(
            OKCYAN_TEXT +
            '\tFormatted MS2PIP input written.' +
            ENDC_TEXT
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
