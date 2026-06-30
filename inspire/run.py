""" Main Script from which the whole program runs.
"""
from argparse import ArgumentParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # pylint: disable=wrong-import-position

import pandas as pd
import tensorflow as tf

from inspire.calibration import calibrate
from inspire.config import Config
from inspire.constants import ENDC_TEXT, OKGREEN_TEXT
from inspire.download import download_data, download_models
from inspire.plot_spectra import plot_spectra
from inspire.predict_spectra import predict_spectra
from inspire.prepare import prepare_for_spectral_prediction, prepare_for_mhcpan
from inspire.feature_creation import create_features
from inspire.feature_selection import select_features
from inspire.rescore import final_rescoring
from inspire.report import generate_report
import inspire

print(f'\n---> Running inSPIRE version {inspire.__version__} <---\n')
pd.options.mode.chained_assignment = None
tf.config.set_visible_devices([], 'GPU')

PIPELINE_OPTIONS = [
    'calibrate',
    'core',
    'downloadExample',
    'prepare',
    'spectralPrepare',
    'panPrepare',
    'predictSpectra',
    'rescore',
    'featureGeneration',
    'featureSelection',
    'featureSelection+',
    'finalRescoring',
    'queryTable',
    'generateReport',
    'plotSpectra',
]

def get_arguments():
    """ Function to collect command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = ArgumentParser(description='inSPIRE Pipeline for MS Search Results.')

    parser.add_argument(
        '--config_file',
        help='Config file to be read from.'
    )

    parser.add_argument(
        '--pipeline',
        choices=PIPELINE_OPTIONS,
        required=True,
        help='What pipeline do you want to run?',
    )

    return parser.parse_args()

def main():
    """ Function to orchestrate running of the whole ininspire package.
    """
    args = get_arguments()

    if args.pipeline == 'downloadExample':
        download_data()
    else:
        config = Config(args.config_file)
        config.validate()
        print(
            OKGREEN_TEXT +
            'Checking for required inSPIRE models...' +
            ENDC_TEXT
        )
        download_models(force_reload=config.force_reload)

    if args.pipeline == 'calibrate':
        print(
            OKGREEN_TEXT +
            'Running CE Calibration...' +
            ENDC_TEXT
        )
        calibrate(config)

    if args.pipeline in ('spectralPrepare', 'prepare', 'core'):
        print(
            OKGREEN_TEXT +
            'Creating Formatted Spectral Prediction Input...' +
            ENDC_TEXT
        )
        prepare_for_spectral_prediction(config)

    if args.pipeline in ('panPrepare', 'prepare', 'core'):
        if config.use_binding_affinity is not None:
            print(
                OKGREEN_TEXT +
                'Creating Formatted NetMHCpan Input...' +
                ENDC_TEXT
            )
        prepare_for_mhcpan(config)

    if args.pipeline in ('predictSpectra', 'core'):
        print(
            OKGREEN_TEXT +
            'Predicting Spectra...' +
            ENDC_TEXT
        )
        predict_spectra(config, 'core')

    if args.pipeline in ('featureGeneration', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Generating Features for Percolator Input...' +
            ENDC_TEXT
        )
        create_features(config)

    if args.pipeline in ('featureSelection', 'featureSelection+', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Optimising Feature Set...' +
            ENDC_TEXT
        )
        select_features(config)

    if args.pipeline in ('finalRescoring', 'featureSelection+', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Running Finalised Rescoring...' +
            ENDC_TEXT
        )
        final_rescoring(config)

    if args.pipeline in ('generateReport', 'featureSelection+', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Generating inSPIRE Performance Report...' +
            ENDC_TEXT
        )
        generate_report(config)

    if args.pipeline == 'plotSpectra':
        print(
            OKGREEN_TEXT +
            'Plotting Spectra...' +
            ENDC_TEXT
        )
        plot_spectra(config)

    print(
        OKGREEN_TEXT +
        f'inSPIRE Pipeline "{args.pipeline}" Complete!' +
        ENDC_TEXT
    )

if __name__ == '__main__':
    main()
