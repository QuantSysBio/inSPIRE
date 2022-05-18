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
    'digest',
    'distillerLog',
    'excludeFeatures',
    'excludeRawFiles',
    'experimentTitle',
    'falseDiscoveryRate',
    'fixedModifications',
    'filterCysteine',
    'featureFoldVarianceLimit',
    'groundTruth',
    'groundTruthAccessionStratumKey',
    'groundTruthScanKey',
    'groundTruthSeqKey',
    'groundTruthSourceKey',
    'includeFeatures',
    'mzAccuracy',
    'outputFolder',
    'proteome',
    'reduce',
    'rescoreMethod',
    'maxForFeatureSelection',
    'scansFolder',
    'scansFormat',
    'scanTitleFormat',
    'searchResults',
    'searchEngine',
    'spectralPredictor',
    'useBindingAffinity',
    'useAccessionStrata',
    'dropUnknownPTMs',
    'rawFileGroupings',
    'additionalContextFeatures',
    'saveSpectra',
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

        if self.query_table is not None:
            self.query_table = self.query_table.replace('~', home).replace('%USERPROFILE%', home)
            if self.query_table.endswith('/'):
                self.query_table = self.query_table[:-1]

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
        self.digest = config_dict['digest']

        # Recommended:
        self.collision_energy = config_dict.get('collisionEnergy', 32)
        self.mz_accuracy = config_dict.get('mzAccuracy', 0.02)
        self.fixed_modifications = config_dict.get('fixedModifications', None)
        self.spectral_predictor = config_dict.get('spectralPredictor', 'prosit')

        # Optional
        self.save_spectra = config_dict.get('saveSpectra', False)
        self.fdr = config_dict.get('falseDiscoveryRate', 0.01)
        self.max_for_selection = config_dict.get('maxForFeatureSelection', 1)
        self.exclude_raw_files = config_dict.get('excludeRawFiles', [])
        self.exclude_features = config_dict.get('excludeFeatures', None)
        self.include_features = config_dict.get('includeFeatures', None)
        self.reduce = config_dict.get('reduce', False)
        self.rescore_method = config_dict.get('rescoreMethod', 'mokapot')
        self.fold_variance_limit = config_dict.get('featureFoldVarianceLimit', 2000.0)
        self.filter_c = config_dict.get('filterCysteine', False)
        self.drop_unknown_mods = config_dict.get('dropUnknownPTMs', False)
        self.additional_context_feats = config_dict.get('additionalContextFeatures', None)

        # Mascot files
        self.combined_scans_file = config_dict.get('combinedScansFile', None)
        self.scan_title_format = config_dict.get('scanTitleFormat', None)
        self.source_files = None
        if 'distillerLog' in config_dict:
            self.source_files = read_distiller_log(config_dict['distillerLog'])

        # NetMhcPan
        self.use_binding_affinity = config_dict.get('useBindingAffinity', None)

        # Ground Truth
        self.query_table = config_dict.get('groundTruth', None)
        self.qt_seq_key = config_dict.get('groundTruthSeqKey')
        self.qt_source_key = config_dict.get('groundTruthSourceKey')
        self.qt_scan_key = config_dict.get('groundTruthScanKey')
        self.qt_accession_stratum_key = config_dict.get('groundTruthAccessionStratumKey')

        # Accession Groups
        self.accession_groups = config_dict.get('accessionGroups')
        self.accession_hierarchy = config_dict.get('accessionHierarchy')
        self.accession_format = config_dict.get('accessionFormat')
        if self.accession_format == 'invitroSPI' and self.accession_hierarchy is None:
            self.accession_hierarchy = ['nonspliced', 'cisspliced', 'transspliced']
        self.model_per_acc_grp = config_dict.get('modelPerAccessionGroup', False)
        if self.model_per_acc_grp and self.model_per_acc_grp in self.accession_hierarchy:
            self.model_per_acc_grp = [
                self.accession_hierarchy.index(x) for x in self.model_per_acc_grp
            ]
        self.use_accession_stratum = config_dict.get('useAccessionStrata', False)
        self.proteome = config_dict.get('proteome')
        self.raw_file_groups = config_dict.get('rawFileGroupings')

    def __str__(self):
        print_string = f'inSPIRE Settings for Experiment {self.experiment_title}:\n\n'
        if isinstance(self.search_results, str):
            print_string += f'Search Results File:\t{self.search_results}\n'
        else:
            print_string += f'Search Results Files:\t{", ".join(self.search_results)}\n'
        print_string += f'Search Engine:\t{self.search_engine}\n'
        print_string += f'MS/MS Scans Folder:\t{self.scans_folder}\n'
        print_string += f'Scans Format:\t{self.scans_format}\n'
        print_string += f'Output Folder:\t{self.output_folder}\n'

        print_string += f'Collision Energy:\t{self.collision_energy}\n'
        print_string += f'M/Z Accuracy on Mass Spectrometer:\t{self.mz_accuracy}\n'

        print_string += f'Maximum Samples for Feature Selection:\t{self.max_for_selection}\n'
        print_string += f'FDR:\t{self.fdr}\n'
        print_string += f'Rescore Method:\t{self.rescore_method}\n'
        print_string += f'Results Reduced to Maximum Engine Score:\t{self.reduce}\n'
        print_string += f'Limit on Feature Variance across Folds:\t{self.fold_variance_limit}\n'
        if self.exclude_features:
            print_string += f'Features Excluded:\t{", ".join(self.exclude_features)}\n'

        if self.use_binding_affinity is not None:
            print_string += f'Binding Affinity Used:\t{self.use_binding_affinity}\n'

        if self.scan_title_format == 'mascotDistiller':
            print_string += 'Mascot Distiller used.\n'
            print_string += f'Combined Scans File:\t{self.combined_scans_file}\n'
            print_string += f'Original Source Files:\t{", ".join(self.source_files)}\n'

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

        if self.rescore_method not in ('mokapot', 'percolator'):
            raise ValueError(
                f'Unsupported Rescore Method: "{self.search_engine}". Supported ' +
                'methods are "mokapot" and "percolator".'
            )

        if self.spectral_predictor not in ('prosit', 'ms2pip'):
            raise ValueError(
                f'Unsupported Spectral Predictor: "{self.spectral_predictor}". Supported ' +
                'methods are "prosit" and "ms2pip".'
            )

        if self.query_table is not None:
            if self.qt_scan_key is None:
                raise ValueError(
                    'Value for groundTruth provided without groundTruthScanKey'
                )

            if self.qt_source_key is None:
                raise ValueError(
                    'Value for groundTruth provided without groundTruthSourceKey'
                )

            if self.qt_seq_key is None:
                raise ValueError(
                    'Value for groundTruth provided without groundTruthSeqKey'
                )

            if self.qt_seq_key is None:
                raise ValueError(
                    'Value for groundTruth provided without groundTruthAccessionStratumKey'
                )
