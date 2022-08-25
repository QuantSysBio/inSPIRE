""" Function for downloading the models required for inSPIRE execution.
"""
import os
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

from inspire.constants import ENDC_TEXT, FIGSHARE_PATH, OKCYAN_TEXT


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
        os.rmdir(f'{home}/inSPIRE_models')

    if os.path.isdir(f'{home}/inSPIRE_models'):
        print(
            OKCYAN_TEXT + '\tModels already downloaded.' + ENDC_TEXT
        )
    else:
        os.mkdir(f'{home}/inSPIRE_models')
        print(
            OKCYAN_TEXT + '\tDownloading models...' + ENDC_TEXT
        )
        urlretrieve(FIGSHARE_PATH, f'{home}/inSPIRE_models/models.zip')
        print(
            OKCYAN_TEXT + '\tExtracting Models...' + ENDC_TEXT
        )
        zip_ref = zipfile.ZipFile(f'{home}/inSPIRE_models/models.zip')
        zip_ref.extractall(f'{home}/inSPIRE_models/models')
        zip_ref.close()
        print(
            OKCYAN_TEXT + '\tModels ready.' + ENDC_TEXT
        )
