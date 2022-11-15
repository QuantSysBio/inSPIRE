""" Definition of Config class.
"""
import os
from pathlib import Path

import yaml

from inspire.utils import read_distiller_log

ALL_CONFIG_KEYS = [
    'accessionFormat',
    'collisionEnergy',
    'combinedScansFile',
    'deltaMethod',
    'distillerLog',
    'dropUnknownPTMs',
    'excludeFeatures',
    'experimentTitle',
    'falseDiscoveryRate',
    'fixedModifications',
    'filterCysteine',
    'forceReload',
    'includeFeatures',
    'ms2pipModel',
    'mzAccuracy',
    'mzUnits',
    'nCores',
    'outputFolder',
    'proteome',
    'reduce',
    'rescoreMethod',
    'reuseInput',
    'scansFolder',
    'scansFormat',
    'scanTitleFormat',
    'searchResults',
    'searchEngine',
    'sourceFileName',
    'spectralPredictor',
    'useBindingAffinity',
    'useAccessionStrata',
]

class Config:
    """ Holder for configuration of the inspire pipeline.
    """
    def __init__(self, config_file):
        with open(config_file, 'r', encoding='UTF-8') as stream:
            config_dict = yaml.safe_load(stream)
        for config_key in config_dict:
            if config_key not in ALL_CONFIG_KEYS:
                raise ValueError(f'Unrecognised key {config_key} found in config file.')
        self._load_data(config_dict)
        self._clean_file_paths()

    def _clean_file_paths(self):
        """ Function to clean the file paths given to inspire.
        """
        home = str(Path.home())

        if isinstance(self.search_results, str):
            self.search_results = self.search_results.replace('~', home).replace(
                '%USERPROFILE%', home
            )
        else:
            self.search_results = [
                x.replace('~', home).replace('%USERPROFILE%', home) for x in self.search_results
            ]

        self.scans_folder = self.scans_folder.replace('~', home).replace('%USERPROFILE%', home)
        if self.scans_folder.endswith('/'):
            self.scans_folder = self.scans_folder[:-1]
        self.output_folder = self.output_folder.replace('~', home).replace('%USERPROFILE%', home)
        if self.output_folder.endswith('/'):
            self.output_folder = self.output_folder[:-1]

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


    def _load_data(self, config_dict):
        """ Function to load data.
        """
        # Required:
        self.experiment_title = config_dict['experimentTitle']
        self.search_results = config_dict['searchResults']
        self.search_engine = config_dict['searchEngine']
        self.output_folder = config_dict['outputFolder']
        self.scans_folder = config_dict['scansFolder']
        self.scans_format = config_dict['scansFormat']

        # Recommended:
        self.n_cores = config_dict.get('nCores', 1)
        self.collision_energy = config_dict.get('collisionEnergy', 32)
        self.mz_accuracy = config_dict.get('mzAccuracy', 0.02)
        self.mz_units = config_dict.get('mzUnits', 'Da')
        self.fixed_modifications = config_dict.get('fixedModifications', None)
        self.spectral_predictor = config_dict.get('spectralPredictor', 'prosit')
        if self.spectral_predictor == 'prosit':
            self.delta_method = config_dict.get('deltaMethod', 'predictor')
        else:
            self.delta_method = config_dict.get('deltaMethod', 'ignore')
        self.reuse_input = config_dict.get('reuseInput', False)

        # Optional
        self.fdr = config_dict.get('falseDiscoveryRate', 0.01)
        self.max_for_selection = -1
        self.exclude_features = config_dict.get('excludeFeatures', [])
        self.include_features = config_dict.get('includeFeatures', None)
        self.reduce = config_dict.get('reduce', False)
        self.rescore_method = config_dict.get('rescoreMethod', 'mokapot')

        self.filter_c = config_dict.get('filterCysteine', True)
        if self.spectral_predictor == 'prosit':
            self.drop_unknown_mods = config_dict.get('dropUnknownPTMs', True)
        else:
            self.drop_unknown_mods = config_dict.get('dropUnknownPTMs', False)

        self.force_reload=config_dict.get('forceReload', False)

        # Mascot files
        self.combined_scans_file = config_dict.get('combinedScansFile', None)
        self.scan_title_format = config_dict.get('scanTitleFormat', None)
        self.source_files = None
        if 'distillerLog' in config_dict:
            self.source_files = read_distiller_log(config_dict['distillerLog'])
        self.source_filename = config_dict.get('sourceFileName', None)

        # NetMhcPan
        self.use_binding_affinity = config_dict.get('useBindingAffinity', None)

        # MS2PIP Model
        self.ms2pip_model = config_dict.get('ms2pipModel', None)


        # Accession Groups
        self.accession_groups = config_dict.get('accessionGroups')
        self.accession_hierarchy = config_dict.get('accessionHierarchy')
        self.accession_format = config_dict.get('accessionFormat')
        if self.accession_format == 'invitroSPI' and self.accession_hierarchy is None:
            self.accession_hierarchy = ['nonspliced', 'cisspliced', 'transspliced']

        self.use_accession_stratum = config_dict.get('useAccessionStrata', False)
        self.proteome = config_dict.get('proteome')
        self.raw_file_groups = config_dict.get('rawFileGroupings')

    def __str__(self):
        print_string = f'inSPIRE Settings for Experiment {self.experiment_title}:<br>'
        print_string += f"""<table style="width:40%">
        <tr>
            <th>Config</th>
            <th>Setting</th>
        </tr>
        <tr>
            <td>searchEngine</td>
            <td>{self.search_engine}</td>
        </tr>
        <tr>
            <td>scansFormat</td>
            <td>{self.scans_format}</td>
        </tr>
        <tr>
            <td>spectralPredictor</td>
            <td>{self.spectral_predictor}</td>
        </tr>
        <tr>
            <td>deltaMethod</td>
            <td>{self.delta_method}</td>
        </tr>
        <tr>
            <td>rescoreMethod</td>
            <td>{self.rescore_method}</td>
        </tr>
        <tr>
            <td>searchResults</td>
            <td>{self.search_results}</td>
        </tr>
        <tr>
            <td>scansFolder</td>
            <td>{self.scans_folder}</td>
        </tr>
        <tr>
            <td>outputFolder</td>
            <td>{self.output_folder}</td>
        </tr>
        """
        if self.spectral_predictor == 'prosit':
            print_string += f"""
                <tr>
                    <td>collisionEnergy</td>
                    <td>{self.collision_energy}</td>
                </tr>
            """
        else:
            print_string += f"""
                <tr>
                    <td>ms2pipModel</td>
                    <td>{self.ms2pip_model}</td>
                </tr>
            """
        print_string += f"""
            <tr>
                <td>mzAccuracy</td>
                <td>{self.mz_accuracy}</td>
            </tr>
            <tr>
                <td>mzUnits</td>
                <td>{self.mz_units}</td>
            </tr>
            <tr>
                <td>fixedModifications</td>
                <td>{self.fixed_modifications}</td>
            </tr>
            <tr>
                <td>forceReload</td>
                <td>{self.force_reload}</td>
            </tr>
            <tr>
                <td>falseDiscoveryRate</td>
                <td>{self.fdr}</td>
            </tr>
            <tr>
                <td>excludeFeatures</td>
                <td>{self.exclude_features}</td>
            </tr>
            <tr>
                <td>includeFeatures</td>
                <td>{self.include_features}</td>
            </tr>
            <tr>
                <td>reduce</td>
                <td>{self.reduce}</td>
            </tr>
            <tr>
                <td>filterCysteine</td>
                <td>{self.filter_c}</td>
            </tr>
            <tr>
                <td>dropUnknownPTMs</td>
                <td>{self.drop_unknown_mods}</td>
            </tr>
            <tr>
                <td>useBindingAffinity</td>
                <td>{self.use_binding_affinity}</td>
            </tr>
        """

        print_string += '</table>'
        # if isinstance(self.search_results, str):
        #     print_string += f'Search Results File:\t{self.search_results}\n'
        # else:
        #     print_string += f'Search Results Files:\t{", ".join(self.search_results)}\n'
        # print_string += f'Search Engine:\t{self.search_engine}\n'
        # print_string += f'MS/MS Scans Folder:\t{self.scans_folder}\n'
        # print_string += f'Scans Format:\t{self.scans_format}\n'
        # print_string += f'Output Folder:\t{self.output_folder}\n'

        # print_string += f'Collision Energy:\t{self.collision_energy}\n'
        # print_string += f'M/Z Accuracy on Mass Spectrometer:\t{self.mz_accuracy}\n'

        # print_string += f'Maximum Samples for Feature Selection:\t{self.max_for_selection}\n'
        # print_string += f'FDR:\t{self.fdr}\n'
        # print_string += f'Rescore Method:\t{self.rescore_method}\n'
        # print_string += f'Results Reduced to Maximum Engine Score:\t{self.reduce}\n'
        # if self.exclude_features:
        #     print_string += f'Features Excluded:\t{", ".join(self.exclude_features)}\n'

        # if self.use_binding_affinity is not None:
        #     print_string += f'Binding Affinity Used:\t{self.use_binding_affinity}\n'

        # if self.scan_title_format == 'mascotDistiller':
        #     print_string += 'Mascot Distiller used.\n'
        #     print_string += f'Combined Scans File:\t{self.combined_scans_file}\n'
        #     print_string += f'Original Source Files:\t{", ".join(self.source_files)}\n'

        return print_string

    def validate(self):
        """ Function to validate config settings.
        """
        if self.search_engine not in ('mascot', 'maxquant', 'peaks'):
            raise ValueError(
                f'Unsupported Search Engine: "{self.search_engine}". Supported ' +
                'engines are "mascot", "peaks", and "maxquant".'
            )

        if self.scans_format not in ('mgf', 'mzML'):
            raise ValueError(
                f'Unsupported Scans Format: "{self.scans_format}". Supported ' +
                'formats are "mgf" and "mzML".'
            )

        if self.mz_units not in ('Da', 'ppm'):
            raise ValueError(
                f'Unsupported mz unit: "{self.mz_units}". Supported units are :Da" and "ppm".'
            )

        if self.rescore_method not in ('mokapot', 'percolator', 'percolatorSeparate'):
            raise ValueError(
                f'Unsupported Rescore Method: "{self.rescore_method}". Supported ' +
                'methods are "mokapot", "percolatorSeparate", and "percolator".'
            )

        if self.spectral_predictor not in ('prosit', 'ms2pip'):
            raise ValueError(
                f'Unsupported Spectral Predictor: "{self.spectral_predictor}". Supported ' +
                'methods are "prosit" and "ms2pip".'
            )

        if self.spectral_predictor == 'ms2pip' and self.ms2pip_model is None:
            raise ValueError(
                'You must specify an ms2pipModel when using the ms2pip spectral predictor.'
            )
