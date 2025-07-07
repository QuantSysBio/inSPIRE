""" Definition of Config class.
"""
import glob
import os
from pathlib import Path
import platform

import yaml

from inspire.utils import read_distiller_log

ALL_CONFIG_KEYS = [
    'accessionFlags',
    'accessionFormat',
    'accessionKeys',
    'additionalPsms',
    'alleles',
    'baPredictionLimit',
    'calibratePerFile',
    'collisionEnergy',
    'contaminantData',
    'combinedScansFile',
    'controlFlags',
    'deNovoFilterResults',
    'deltaMethod',
    'distillerLog',
    'dropUnknown',
    'dropUnknownPTMs',
    'epitopeCandidateCutOff',
    'epitopeCutLevel',
    'excludeFeatures',
    'engineScoreCut',
    'enzyme',
    'experimentTitle',
    'falseDiscoveryRate',
    'fixedModifications',
    'filterCysteine',
    'forceReload',
    'forPisces',
    'fraggerMemory',
    'fraggerMods',
    'fraggerDbSplits',
    'fraggerPath',
    'fraggerParams',
    'fraggerUseContams',
    'hostOnlyResults',
    'hostProteome',
    'includeFeatures',
    'inferProteins',
    'mapContaminants',
    'ms2pipModel',
    'ms1Accuracy',
    'mzAccuracy',
    'mzRange',
    'mzUnits',
    'nCores',
    'netMHCpan',
    'outputFolder',
    'pathogenProteome',
    'piscesDnEngine',
    'plotSpectraSourceSplit',
    'proteinDecoyKey',
    'proteome',
    'quantCutOff',
    'rtFitLoc',
    'reduce',
    'remapToProteome',
    'replaceIL',
    'rescoreCommand',
    'rescoreMethod',
    'resultsExport',
    'reuseInput',
    'panDocker',
    'scansFolder',
    'scansFormat',
    'scanTitleFormat',
    'searchResults',
    'searchEngine',
    'sourceFileName',
    'silentExecution',
    'skylineConfig',
    'skylineReportTemplate',
    'skylineRunner',
    'skylineIdpCutOff',
    'skylinePrecursorCountFilter',
    'skylineRatioCutOff',
    'spectralAngleDfs',
    'spectralPredictor',
    'useAccessionStrata',
    'useBindingAffinity',
    'useIrtDelta',
    'useMinimalFeatures',
    'technicalReplicates',
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
        elif isinstance(self.search_results, list):
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

        if not os.path.exists(f'{self.output_folder}/quant'):
            os.makedirs(f'{self.output_folder}/quant')

        if not os.path.exists(f'{self.output_folder}/PEPSeek'):
            os.makedirs(f'{self.output_folder}/PEPSeek')

        if not os.path.exists(f'{self.output_folder}/img'):
            os.makedirs(f'{self.output_folder}/img')

    def _load_data(self, config_dict):
        """ Function to load data.
        """
        home = str(Path.home())

        # Required:
        self.experiment_title = config_dict['experimentTitle']
        self.output_folder = config_dict['outputFolder']
        search_res = config_dict.get('searchResults')
        if isinstance(search_res, list):
            self.search_results = []
            for result_group in search_res:
                self.search_results.extend(glob.glob(result_group))
        else:
            self.search_results = search_res
        if (
            self.search_results is None and
            os.path.exists(f'{self.output_folder}/fragger_searches.txt')
        ):
            with open(
            f'{self.output_folder}/fragger_searches.txt',
            mode='r',
            encoding='UTF-8',
        ) as search_file:
                self.search_results = [line.rstrip() for line in search_file]
        self.search_engine = config_dict['searchEngine']
        self.scans_folder = config_dict['scansFolder']
        self.scans_format = config_dict['scansFormat']

        # Recommended:
        self.map_contaminants = config_dict.get('mapContaminants', 'standard')
        self.results_export = config_dict.get('resultsExport', 'psm')
        self.use_irt_diff = config_dict.get('useIrtDelta', True)
        self.proteome = config_dict.get('proteome')
        if self.proteome is None:
            self.infer_proteins = config_dict.get('inferProteins', False)
        else:
            self.infer_proteins = config_dict.get('inferProteins', True)
        self.n_cores = config_dict.get('nCores', 1)
        self.collision_energy = config_dict.get('collisionEnergy', None)
        self.ms1_accuracy = config_dict.get('ms1Accuracy', 5)
        self.mz_accuracy = config_dict.get('mzAccuracy', 0.02)
        self.mz_units = config_dict.get('mzUnits', 'Da')
        self.enzyme = config_dict.get('enzyme', None)
        self.replace_il = config_dict.get('replaceIL', False)
        self.fixed_modifications = config_dict.get('fixedModifications', None)
        self.spectral_predictor = config_dict.get('spectralPredictor', 'prosit')
        if self.spectral_predictor == 'prosit':
            self.delta_method = config_dict.get('deltaMethod', 'predictor')
        else:
            self.delta_method = config_dict.get('deltaMethod', 'ignore')
        self.reuse_input = config_dict.get('reuseInput', False)
        self.for_pisces = config_dict.get('forPisces', False)
        self.pisces_dn_method = config_dict.get('piscesDnEngine')
        self.minimal_features = config_dict.get('useMinimalFeatures', self.for_pisces)
        self.additional_psms = config_dict.get('additionalPsms')

        # MSFragger
        self.fragger_mods = config_dict.get('fraggerMods', 'standard')
        self.fragger_use_contams = config_dict.get('fraggerUseContams', True)
        self.fragger_memory = config_dict.get('fraggerMemory', 60)
        self.fragger_db_splits = config_dict.get('fraggerDbSplits', 4)
        self.fragger_path = config_dict.get('fraggerPath')
        self.fragger_params = config_dict.get(
            'fraggerParams',
            f'{home}/inSPIRE_models/utilities/fragger_template.params',
        )

        # Quantification
        self.quantification_cut_off = config_dict.get('quantCutOff', 0.01)
        self.technical_replicates = config_dict.get('technicalReplicates')
        self.skyline_runner = config_dict.get('skylineRunner')
        self.skyline_config_file = config_dict.get(
            'skylineConfig', f'{home}/inSPIRE_models/utilities/skyline_config.sky'
        )
        self.skyline_precursor_filter = config_dict.get('skylinePrecursorCountFilter', 3)
        self.skyline_report_template = config_dict.get(
            'skylineReportTemplate', f'{home}/inSPIRE_models/utilities/skyline_report_template.skyr'
        )
        self.skyline_idp_cut_off = config_dict.get('skylineIdpCutOff', 0.5)
        self.skyline_bg_ratio_cut_off = config_dict.get('skylineRatioCutOff', 0.8)

        # Protein Inference
        self.decoy_protein_flag = config_dict.get('proteinDecoyKey', 'rev_')

        # Optional
        self.drop_unknown = config_dict.get('dropUnknown', False)
        self.calibrate_per_file = config_dict.get('calibratePerFile', True)
        self.remap_to_proteome = config_dict.get('remapToProteome', False)
        self.fdr = config_dict.get('falseDiscoveryRate', 0.01)
        self.exclude_features = config_dict.get('excludeFeatures', [])
        self.include_features = config_dict.get('includeFeatures', None)
        self.reduce = config_dict.get('reduce', False)
        self.rescore_method = config_dict.get('rescoreMethod', 'percolator')
        self.rescore_command = config_dict.get('rescoreCommand')
        if self.rescore_command is None:
            if self.rescore_method.startswith('percolator'):
                self.rescore_command = 'percolator'
            else:
                self.rescore_command = 'mokapot'

        self.sa_query_dfs = config_dict.get('spectralAngleDfs', None)
        self.silent_execution = config_dict.get('silentExecution', False)
        self.plot_spectra_source_split = config_dict.get('plotSpectraSourceSplit')

        self.rt_fit_loc = config_dict.get('rtFitLoc', None)

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
        self.pan_docker = config_dict.get(
            'panDocker',
            platform.system() == 'Windows',
        )
        self.use_binding_affinity = config_dict.get('useBindingAffinity', None)
        self.pan_command = config_dict.get('netMHCpan')
        self.ba_pred_limit = config_dict.get('baPredictionLimit', 15)

        # MS2PIP Model
        self.ms2pip_model = config_dict.get('ms2pipModel', None)


        # Accession Groups
        self.accession_hierarchy = config_dict.get('accessionKeys')
        self.accession_flags = config_dict.get('accessionFlags')
        self.accession_format = config_dict.get('accessionFormat')
        if self.accession_format == 'invitroSPI' and self.accession_hierarchy is None:
            self.accession_hierarchy = ['nonspliced', 'spliced']

        self.use_accession_stratum = config_dict.get('useAccessionStrata', False)

        # Epitope validation
        self.host_proteome = config_dict.get('hostProteome')
        self.pathogen_proteome = config_dict.get('pathogenProteome')
        self.control_flags = config_dict.get('controlFlags')
        self.alleles = config_dict.get('alleles')
        self.epitope_candidate_cut_off = config_dict.get('epitopeCandidateCutOff', 0.1)
        self.epitope_cut_level = config_dict.get('epitopeCutLevel', 'psm')
        self.host_only_results = config_dict.get('hostOnlyResults')
        self.engine_score_cut = config_dict.get('engineScoreCut')
        self.epitope_length_cut_off = config_dict.get('epitopeLengthCutOff', 15)

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

        return print_string

    def validate(self):
        """ Function to validate config settings.
        """
        if self.search_engine not in (
            'mascot', 'maxquant', 'peaks', 'peaksDeNovo', 'msfragger', 'psms', 'casanovo'
        ):
            raise ValueError(
                f'Unsupported Search Engine: "{self.search_engine}". Supported ' +
                'engines are "mascot", "peaks", "peaksDeNovo", "psms", "msfragger", and "maxquant".'
            )

        if self.scans_format not in ('mgf', 'mzML', 'mzML_rt'):
            raise ValueError(
                f'Unsupported Scans Format: "{self.scans_format}". Supported ' +
                'formats are "mgf", "mzML", and "mzML_rt.'
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

        if self.epitope_cut_level not in ('psm', 'peptide'):
            raise ValueError(
                'Unsupported epitope filtering level {self.epitope_cut_level}, must be' +
                'psm or peptide.'
            )
