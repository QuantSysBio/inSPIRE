""" Function for generating Prosit or MS2PIP spectral predictions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from inspire.prosit import (
    load_model,
    get_sequence_integer,
    get_precursor_charge_onehot,
    prosit_predict,
    write_msp_file,
)


def predict_spectra(config, pipeline='core'):
    """ Function to generate spectral predictions for a set of peptide sequences
        using either Prosit or MS2PIP and write to msp format.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object which manages the experiment.
    pipeline : str (default=core)
        The pipeline which is being run.
    """
    if config.spectral_predictor == 'ms2pip':
        raise ValueError('inSPIRE 2.0 does not support running MS2PIP natively.')

    home = str(Path.home())
    d_irt = load_model(
        f'{home}/inSPIRE_models/models/irt_config.yml',
        f'{home}/inSPIRE_models/models/irt_model.json',
        f'{home}/inSPIRE_models/models/weight_66_0.00796.hdf5',
    )
    d_spectra = load_model(
        f'{home}/inSPIRE_models/models/spectra_config.yml',
        f'{home}/inSPIRE_models/models/spectra_model.json',
        f'{home}/inSPIRE_models/models/weight_163_0.11385.hdf5',
    )

    if pipeline =='core':
        input_file = f'{config.output_folder}/prositInput.csv'
        out_file = f'{config.output_folder}/prositPredictions.msp'
    elif pipeline == 'calibrate':
        input_file = f'{config.output_folder}/calibrationInput.csv'
        out_file = f'{config.output_folder}/calibrationPredictions.msp'
    elif pipeline == 'spectralAngle':
        input_file = f'{config.output_folder}/saInput.csv'
        out_file = f'{config.output_folder}/saPredictions.msp'
    elif pipeline == 'validation':
        input_file = f'{config.output_folder}/validationInput.csv'
        out_file = f'{config.output_folder}/validationPredictions.msp'
    else:
        input_file = f'{config.output_folder}/plotInput.csv'
        out_file = f'{config.output_folder}/plotPredictions.msp'

    for chunk_idx, input_df in enumerate(pd.read_csv(input_file, chunksize=200_000)):
        input_df = input_df.reset_index(drop=True)
        prosit_input = {
            'collision_energy_aligned_normed': (
                np.expand_dims(
                    np.array(input_df['collision_energy']).astype(float), axis=1
                ) / 100.0
            ),
            'sequence_integer': get_sequence_integer(input_df['modified_sequence']),
            'precursor_charge_onehot': get_precursor_charge_onehot(
                input_df['precursor_charge']
            ),
        }

        prosit_data = prosit_predict(prosit_input, d_irt)
        final_result = prosit_predict(prosit_data, d_spectra)

        write_msp_file(input_df, final_result, out_file, chunk_idx)
        del prosit_input
        del prosit_data
