""" Function for downloading the models required for inSPIRE execution.
"""
import os
from pathlib import Path
from urllib.request import urlretrieve
import shutil
import tarfile
import zipfile

from inspire.constants import (
    ENDC_TEXT,
    FIGSHARE_EXAMPLE_PATH,
    FIGSHARE_EXTERNAL_UTILS_PATH,
    FIGSHARE_PATH,
    FIGSHARE_PISCES_MODELS,
    OKCYAN_TEXT,
    THERMO_PARSER_PATH,
)


def download_thermo_raw_file_parser():
    """ Function to download the ThermoRawFileParser.
    """
    home = str(Path.home())
    if os.path.isdir(f'{home}/inSPIRE_models/ThermoRawFileParser'):
        print(
            OKCYAN_TEXT + '\tThermoRawFileParser already downloaded.' + ENDC_TEXT
        )
    else:
        os.mkdir(f'{home}/inSPIRE_models/ThermoRawFileParser')
        print(
            OKCYAN_TEXT + '\tDownloading ThermoRawFileParser...' + ENDC_TEXT
        )
        urlretrieve(THERMO_PARSER_PATH, f'{home}/inSPIRE_models/ThermoRawFileParser/parser.zip')
        print(
            OKCYAN_TEXT + '\tExtracting ThermoRawFileParser...' + ENDC_TEXT
        )
        with zipfile.ZipFile(f'{home}/inSPIRE_models/ThermoRawFileParser/parser.zip') as zip_ref:
            zip_ref.extractall(f'{home}/inSPIRE_models/ThermoRawFileParser')
        print(
            OKCYAN_TEXT + '\tThermoParserReady ready.' + ENDC_TEXT
        )


def download_pisces_models(force_reload=False):
    """ Function to download the required models for inSPIRE execution from
        figshare.

    Parameters
    ----------
    force_reload : bool (default=False)
        Flag indicating whether to remove the existing inSPIRE_models folder
        and redownload all models.
    """
    home = str(Path.home())
    pisces_model_path = f'{home}/inSPIRE_models/pisces_models'

    if force_reload:
        os.rmdir(f'{pisces_model_path}')

    download=True
    if os.path.isfile(f'{pisces_model_path}/version.txt'):
        with open(
            f'{pisces_model_path}/version.txt', 'r', encoding='UTF-8',
        ) as version_file:
            version = version_file.read().strip()
            if version == '>=3.0':
                print(
                    OKCYAN_TEXT + '\tModels already downloaded.' + ENDC_TEXT
                )
                download=False

    if download:
        if not os.path.isdir(f'{pisces_model_path}'):
            os.mkdir(f'{pisces_model_path}')
        print(
            OKCYAN_TEXT + '\tDownloading PISCES models...' + ENDC_TEXT
        )
        urlretrieve(FIGSHARE_PISCES_MODELS, f'{home}/inSPIRE_models/pisces_models.zip')
        print(
            OKCYAN_TEXT + '\tExtracting Models...' + ENDC_TEXT
        )
        with zipfile.ZipFile(f'{home}/inSPIRE_models/pisces_models.zip') as zip_ref:
            zip_ref.extractall(f'{home}/inSPIRE_models')

        for mode in os.listdir(pisces_model_path):
            for method in os.listdir(f'{pisces_model_path}/{mode}'):
                for model_idx in os.listdir(f'{pisces_model_path}/{mode}/{method}'):
                    for model in os.listdir(f'{pisces_model_path}/{mode}/{method}/{model_idx}'):
                        if not model.startswith('clf'):
                            true_name = 'clf' + model.split('_clf')[-1]
                            os.rename(
                                f'{pisces_model_path}/{mode}/{method}/{model_idx}/{model}',
                                f'{pisces_model_path}/{mode}/{method}/{model_idx}/{true_name}'
                            )

        os.rename(f'{home}/inSPIRE_models/version.txt', f'{pisces_model_path}/version.txt')
        os.remove(f'{home}/inSPIRE_models/pisces_models.zip')
        print(
            OKCYAN_TEXT + '\tPISCES Models ready.' + ENDC_TEXT
        )


def download_models(force_reload=False):
    """ Function to download the required models for inSPIRE execution from
        figshare.

    Parameters
    ----------
    force_reload : bool (default=False)
        Flag indicating whether to remove the existing inSPIRE_models folder
        and redownload all models.
    """
    home = str(Path.home())
    if force_reload:
        os.rmdir(f'{home}/inSPIRE_models/models')
    

    download=True
    if os.path.isfile(f'{home}/inSPIRE_models/models/version.txt'):
        with open(
            f'{home}/inSPIRE_models/models/version.txt', 'r', encoding='UTF-8',
        ) as version_file:
            version = version_file.read().strip()
            if version == '>=3.0':
                print(
                    OKCYAN_TEXT + '\tModels already downloaded.' + ENDC_TEXT
                )
                download=False

    if download:
        if not os.path.isdir(f'{home}/inSPIRE_models'):
            os.mkdir(f'{home}/inSPIRE_models')
        print(
            OKCYAN_TEXT + '\tDownloading models...' + ENDC_TEXT
        )
        urlretrieve(FIGSHARE_PATH, f'{home}/inSPIRE_models/models.zip')
        print(
            OKCYAN_TEXT + '\tExtracting Models...' + ENDC_TEXT
        )
        with zipfile.ZipFile(f'{home}/inSPIRE_models/models.zip') as zip_ref:
            zip_ref.extractall(f'{home}/inSPIRE_models/models')

        os.remove(f'{home}/inSPIRE_models/models.zip')
        print(
            OKCYAN_TEXT + '\tModels ready.' + ENDC_TEXT
        )

def download_utils(force_reload=False):
    """ Function to download the required models for inSPIRE execution from
        figshare.

    Parameters
    ----------
    force_reload : bool (default=False)
        Flag indicating whether to remove the existing inSPIRE_models folder
        and redownload all models.
    """
    home = str(Path.home())
    if force_reload:
        shutil.rmtree(f'{home}/inSPIRE_models/utilities')

    download=True
    if os.path.isfile(f'{home}/inSPIRE_models/utilities/version.txt'):
        with open(
            f'{home}/inSPIRE_models/utilities/version.txt', 'r', encoding='UTF-8',
        ) as version_file:
            version = version_file.read().strip()
            if version == '>=3.0':
                print(
                    OKCYAN_TEXT + '\tUtils already downloaded.' + ENDC_TEXT
                )
                download=False

    if download:
        if os.path.isdir(f'{home}/inSPIRE_models/utilities'):
            shutil.rmtree(f'{home}/inSPIRE_models/utilities')

        os.mkdir(f'{home}/inSPIRE_models/utilities')
        print(
            OKCYAN_TEXT + '\tDownloading external utilities...' + ENDC_TEXT
        )
        urlretrieve(FIGSHARE_EXTERNAL_UTILS_PATH, f'{home}/inSPIRE_models/utilities/utils.zip')
        print(
            OKCYAN_TEXT + '\tExtracting utils...' + ENDC_TEXT
        )
        with zipfile.ZipFile(f'{home}/inSPIRE_models/utilities/utils.zip') as zip_ref:
            zip_ref.extractall(f'{home}/inSPIRE_models/utilities')
        print(
            OKCYAN_TEXT + '\tUtils ready.' + ENDC_TEXT
        )


def download_data():
    """ Function to download the example dataset from Figshare

    Parameters
    ----------
    force_reload : bool (default=False)
        Flag indicating whether to remove the existing inSPIRE_models folder
        and redownload all models.
    """

    if os.path.isdir('example'):
        print(
            OKCYAN_TEXT + '\tExample data already downloaded.' + ENDC_TEXT
        )
    else:
        print(
            OKCYAN_TEXT + '\tDownloading data...' + ENDC_TEXT
        )
        urlretrieve(FIGSHARE_EXAMPLE_PATH, filename=f'{os.getcwd()}/example.tar.gz')
        print(
            OKCYAN_TEXT + '\tExtracting Data...' + ENDC_TEXT
        )
        with tarfile.open('example.tar.gz', "r:gz") as tar:
            tar.extractall()
        print(
            OKCYAN_TEXT + '\tDataset ready.' + ENDC_TEXT
        )
