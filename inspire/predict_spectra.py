""" Function for generating Prosit or MS2PIP spectral predictions.
"""
from pathlib import Path
from ms2pip.ms2pipC import MS2PIP

from inspire.prosit import (
    load_model, process_csv_file, prosit_predict, write_msp_file
)

PARAMS = {
    "ms2pip": {
        "ptm": [
            'Oxidation,15.994915,opt,M',
            'Carbamidomethyl,57.021464,opt,C',
            'Acetyl,42.010565,opt,N-term',
            'Deamidation,0.984016,opt,N',
            'Phospho,79.966331,opt,S',
        ],
        "frag_error": 0.02,
        "out": "msp",
        "sptm": [], "gptm": [],
    }
}

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
        specific_params = PARAMS
        specific_params['ms2pip']['frag_method'] = config.ms2pip_model

        ms2pip = MS2PIP(
            f'{config.output_folder}/ms2pipInput.preprec',
            params=PARAMS
        )
        ms2pip.run()
        if config.delta_method == 'bruteForce':
            ms2pip = MS2PIP(
                f'{config.output_folder}/deltaInput.preprec',
                params=PARAMS
            )
            ms2pip.run()
    else:
        home = str(Path.home())
        d_irt = load_model(
            f'{home}/inSPIRE_models/models/irt_config.yml',
            f'{home}/inSPIRE_models/models/irt_model.yml',
            f'{home}/inSPIRE_models/models/weight_66_0.00796.hdf5',
        )
        d_spectra = load_model(
            f'{home}/inSPIRE_models/models/spectra_config.yml',
            f'{home}/inSPIRE_models/models/spectra_model.yml',
            f'{home}/inSPIRE_models/models/weight_163_0.11385.hdf5',
        )

        if pipeline =='core':
            input_file = f'{config.output_folder}/prositInput.csv'
            out_file = f'{config.output_folder}/prositPredictions.msp'
        elif pipeline == 'calibrate':
            input_file = f'{config.output_folder}/calibrationInput.csv'
            out_file = f'{config.output_folder}/calibrationPredictions.msp'
        else:
            input_file = f'{config.output_folder}/plotInput.csv'
            out_file = f'{config.output_folder}/plotPredictions.msp'
        input_df, prosit_input = process_csv_file(input_file)

        prosit_data = prosit_predict(prosit_input, d_irt)
        final_result = prosit_predict(prosit_data, d_spectra)

        write_msp_file(input_df, final_result, out_file)

        if config.delta_method == 'bruteForce' and pipeline == 'core':
            input_df, prosit_input = process_csv_file(f'{config.output_folder}/deltaInput.csv')

            prosit_data = prosit_predict(prosit_input, d_irt)
            final_result = prosit_predict(prosit_data, d_spectra)

            write_msp_file(
                input_df,
                final_result,
                f'{config.output_folder}/deltaPredictions.msp',
            )
