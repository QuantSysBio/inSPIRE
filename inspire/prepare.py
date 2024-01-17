""" Suite of function for preparing input to external tools, specifically
    Prosit and netMHCpan input.
"""

import os

import polars as pl

from inspire.constants import (
    CHARGE_KEY,
    ENDC_TEXT,
    MS2PIP_NAME_MAPPINGS,
    OKCYAN_TEXT,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    SEQ_LEN_KEY,
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
    search_df = search_df.with_columns(
        pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).apply(
            lambda x : create_prosit_mod_seq(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], ox_flag),
            skip_nulls=False,
        ).alias('modified_sequence')
    )

    # Write sequences in Prosit's input format.
    prosit_df = search_df.select('modified_sequence', CHARGE_KEY).rename(
        {CHARGE_KEY: 'precursor_charge'}
    )
    if isinstance(collision_energy, list):
        prosit_df = prosit_df.unique()
        prosit_df['collision_energy'] = [collision_energy for _ in range(prosit_df.shape[0])]
        prosit_df = prosit_df.explode('collision_energy')
        if overwrite:
            prosit_df.write_csv(
                f'{config.output_folder}/{filename}.csv',
            )
        else:
            with open(f'{config.output_folder}/{filename}.csv', mode='ab') as out_file:
                prosit_df.write_csv(
                    out_file,
                    has_header=False,
                )
    else:
        prosit_df = prosit_df.with_columns(
            pl.lit(collision_energy).alias('collision_energy')
        )
        prosit_df = prosit_df.unique()
        if overwrite:
            prosit_df.write_csv(
                f'{config.output_folder}/{filename}.csv',
            )
        else:
            with open(f'{config.output_folder}/{filename}.csv', mode='ab') as out_file:
                prosit_df.write_csv(
                    out_file,
                    has_header=False,
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

def write_ms2pip_input_df(target_df, mods_df, output_folder, output_name):
    """ Function to write input sequences for ms2pip.

    Parameters
    ----------
    target_df : pl.DataFrame
        A DataFrame of PSMs for which we require MS2PIP predictions.
    mods_df : pl.DataFrame
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

    target_df = target_df.with_row_count(name='spec_id')
    target_df = target_df.with_columns(
        pl.col('spec_id').apply(lambda x : f'peptide_{x}'),
        pl.col(PTM_SEQ_KEY).apply(
            lambda x : get_ms2pip_mods(x, mod_id_mappings)
        ).alias('modifications'),
    )

    ms2pip_input_df = target_df.select(['spec_id', 'modifications', 'peptide', 'charge'])
    ms2pip_input_df = ms2pip_input_df.unique(
        subset=['modifications', 'peptide', 'charge']
    )

    ms2pip_input_df.write_csv(
        f'{output_folder}/{output_name}.preprec', separator=' ',
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

    unique_pep_lens = target_df[SEQ_LEN_KEY].unique().to_list()

    if not os.path.exists(f'{config.output_folder}/mhcpan'):
        os.makedirs(f'{config.output_folder}/mhcpan')

    for length in unique_pep_lens:
        if length > config.ba_pred_limit:
            continue
        len_df = target_df.filter(pl.col(SEQ_LEN_KEY) == length)
        peptides = len_df.select(PEPTIDE_KEY).unique()
        peptides = peptides.with_row_count('id').with_columns(
            (pl.col('id')//10_000).alias('batch')
        )
        split_peptides = peptides.partition_by('batch')
        for idx, pep_group_df in enumerate(split_peptides):
            pep_group_df.select('peptide').write_csv(
                f'{config.output_folder}/mhcpan/inputLen{length}_{idx}.txt',
                has_header=False,
            )

    print(
        OKCYAN_TEXT +
        '\tFormatted NetMHCpan input written.' +
        ENDC_TEXT
    )
