""" Main Script from which the whole program runs.
"""
from argparse import ArgumentParser
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # pylint: disable=wrong-import-position

import pandas as pd
from scipy.stats import ConstantInputWarning
import tensorflow as tf

from inspire.calibration import calibrate
from inspire.config import Config
from inspire.convert import convert_raw_to_mgf
from inspire.constants import ENDC_TEXT, OKGREEN_TEXT
from inspire.download import download_data, download_models, download_utils
from inspire.epitope.extract_candidates import extract_epitope_candidates
from inspire.execute_msfragger import execute_msfragger
from inspire.get_spectral_angle import get_spectral_angle
from inspire.input.search_results import generic_read_df
from inspire.plot_spectra.plot_isobars import plot_isobars
from inspire.plot_spectra.plot_spectra import plot_spectra
from inspire.predict_binding import predict_binding
from inspire.predict_spectra import predict_spectra
from inspire.prepare import prepare_for_spectral_prediction, prepare_for_mhcpan
from inspire.feature_creation import create_features
from inspire.feature_selection import select_features
from inspire.quant.execute import quantify_identifications
from inspire.quant.normalise import normalise_intensities
from inspire.quant.de_analysis import de_analysis
from inspire.quant.report_template import create_quant_report
from inspire.rescore import final_rescoring
from inspire.report import generate_report
from inspire.utils import fetch_collision_energy
from inspire.validate import validate_spliced
import inspire

pd.options.mode.chained_assignment = None
tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings("ignore", category=ConstantInputWarning)


PIPELINE_OPTIONS = [
    'calibrate',
    'core',
    'convert',
    'downloadExample',
    'format',
    'fragger',
    'extractCandidates',
    'featureGeneration',
    'featureSelection',
    'featureSelection+',
    'finalRescoring',
    'generateReport',
    'panPrepare',
    'plotIsobars',
    'plotSpectra',
    'predictBinding',
    'predictSpectra',
    'prepare',
    'quantify',
    'rescore',
    'spectralAngle',
    'spectralPrepare',
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

def run_inspire(pipeline=None, config_file=None):
    """ Function to orchestrate running of the whole ininspire package.
    """
    print(f'\n---> Running inSPIRE version {inspire.__version__} <---\n')
    if pipeline is None:
        args = get_arguments()
        config_file = args.config_file
        pipeline = args.pipeline

    if pipeline == 'downloadExample':
        download_data()
    else:
        config = Config(config_file)
        config.validate()
        print(
            OKGREEN_TEXT +
            'Checking for required inSPIRE models...' +
            ENDC_TEXT
        )
        download_models(force_reload=config.force_reload)
        download_utils(force_reload=config.force_reload)

    if pipeline == 'calibrate' or (
        config.collision_energy is None and
        not os.path.exists(f'{config.output_folder}/collisionEnergyStats.csv')
        and pipeline not in ('convert', 'fragger')
    ):
        print(
            OKGREEN_TEXT +
            'Running CE Calibration...' +
            ENDC_TEXT
        )
        calibrate(config)

    if config.collision_energy is None and pipeline not in ('convert', 'fragger'):
        config.collision_energy = fetch_collision_energy(config.output_folder)

    if pipeline == 'convert':
        print(
            OKGREEN_TEXT +
            'Creating Formatted Spectral Prediction Input...' +
            ENDC_TEXT
        )
        convert_raw_to_mgf(config)

    if pipeline == 'fragger':
        print(
            OKGREEN_TEXT +
            'Executing MSFragger with default inSPIRE settings...' +
            ENDC_TEXT
        )
        execute_msfragger(config)

    if pipeline == 'format':
        print(
            OKGREEN_TEXT +
            'Formatting search results for inSPIRE input...' +
            ENDC_TEXT
        )
        _ = generic_read_df(config)

    if pipeline in ('spectralPrepare', 'prepare', 'core'):
        print(
            OKGREEN_TEXT +
            'Creating Formatted Spectral Prediction Input...' +
            ENDC_TEXT
        )
        prepare_for_spectral_prediction(config)

    if pipeline in ('panPrepare', 'prepare', 'core'):
        if config.use_binding_affinity is not None:
            print(
                OKGREEN_TEXT +
                'Creating Formatted NetMHCpan Input...' +
                ENDC_TEXT
            )
        prepare_for_mhcpan(config)

    if pipeline in ('predictSpectra', 'core'):
        print(
            OKGREEN_TEXT +
            'Predicting Spectra...' +
            ENDC_TEXT
        )
        predict_spectra(config, 'core')

    if pipeline in ('predictBinding', 'core'):
        if config.use_binding_affinity is not None:
            print(
                OKGREEN_TEXT +
                'Predicting NetMHCpan Binding Affinity...' +
                ENDC_TEXT
            )
            predict_binding(config)

    if pipeline in ('featureGeneration', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Generating Features for Percolator Input...' +
            ENDC_TEXT
        )
        create_features(config)

    if pipeline in ('featureSelection', 'featureSelection+', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Optimising Feature Set...' +
            ENDC_TEXT
        )
        select_features(config)

    if pipeline in ('finalRescoring', 'featureSelection+', 'rescore', 'core'):
        print(
            OKGREEN_TEXT +
            'Running Finalised Rescoring...' +
            ENDC_TEXT
        )
        final_rescoring(config)

    if (
        pipeline in ('validate', 'featureSelection+', 'rescore', 'core', 'calibrate+core')
        and config.accession_format == 'invitroSPI'
    ):
        print(
            OKGREEN_TEXT +
            'Validating spliced assignments...' +
            ENDC_TEXT
        )
        validate_spliced(config)

    if (
        pipeline in ('featureSelection+', 'rescore', 'core') and not config.silent_execution
    ) or pipeline == 'generateReport':
        print(
            OKGREEN_TEXT +
            'Generating inSPIRE Performance Report...' +
            ENDC_TEXT
        )
        generate_report(config)

    if (
        pipeline in ('validate', 'featureSelection+', 'rescore', 'core', 'calibrate+core')
        and config.use_accession_stratum
    ):
        print(
            OKGREEN_TEXT +
            'Validating spliced assignments...' +
            ENDC_TEXT
        )
        validate_spliced(config)

    if pipeline == 'spectralAngle':
        print(
            OKGREEN_TEXT +
            'Calculating Spectral Angles...' +
            ENDC_TEXT
        )
        get_spectral_angle(config)

    if pipeline == 'quantify':
        print(
            OKGREEN_TEXT +
            'Running quantification via skyline docker...' +
            ENDC_TEXT
        )
        quantify_identifications(config)
        normalise_intensities(config)
        de_analysis(config)
        create_quant_report(config)


    if pipeline == 'extractCandidates':
        print(
            OKGREEN_TEXT +
            'Extracting Potential Epitope Candidates...' +
            ENDC_TEXT
        )
        extract_epitope_candidates(config)

    if pipeline == 'plotSpectra':
        print(
            OKGREEN_TEXT +
            'Plotting Spectra...' +
            ENDC_TEXT
        )
        plot_spectra(config)

    if pipeline == 'plotIsobars':
        print(
            OKGREEN_TEXT +
            'Plotting Isobars...' +
            ENDC_TEXT
        )
        plot_isobars(config)

    print(
        OKGREEN_TEXT +
        f'inSPIRE Pipeline "{pipeline}" Complete!' +
        ENDC_TEXT
    )

if __name__ == '__main__':
    run_inspire()
