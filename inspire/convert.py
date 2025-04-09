""" Functions for RAW file conversion.
"""
from multiprocessing import Pool
import os
from pathlib import Path
import platform

from inspire.constants import ENDC_TEXT, OKCYAN_TEXT
from inspire.download import download_thermo_raw_file_parser

def convert_raw_to_mgf(config):
    """ Function to convert RAW files by calling the ThermoRawFileParser.

    Parameters
    ----------
    config : inspire.config.Config
        Config object which runs the experiment.
    """
    scans_folder = config.scans_folder
    raw_files = [
        scan_f for scan_f in os.listdir(scans_folder) if scan_f.lower().endswith('.raw')
    ]
    print(
        OKCYAN_TEXT + f'\tFound {len(raw_files)} RAW files to be converted.' + ENDC_TEXT
    )

    home = str(Path.home())
    if not os.path.isdir(f'{home}/inSPIRE_models/ThermoRawFileParser'):
        download_thermo_raw_file_parser()
    thermo_path = f'{home}/inSPIRE_models/ThermoRawFileParser/ThermoRawFileParser.exe'

    prefix = ''
    if platform.system() != 'Windows':
        prefix = 'mono '

    convert_commands = [
        f'{prefix}{thermo_path} -o={scans_folder} -f=0 -i={scans_folder}/{raw_file} -l 4'
        for raw_file in raw_files if not os.path.exists(
            f'{scans_folder}/{raw_file.replace(".raw", ".mgf")}'
        )
    ]

    with Pool(processes=config.n_cores) as pool:
        pool.map(os.system, convert_commands)
