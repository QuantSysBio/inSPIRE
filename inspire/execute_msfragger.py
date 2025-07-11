""" Functions for executing MSFragger Searches from within inSPIRE.
"""
import os
from pathlib import Path
import shutil
import sys


UNSPECIFIC_CLEAVAGE = (
    'search_enzyme_name_1 = nonspecific\n'+
    'search_enzyme_cut_1 = -\n' +
    'search_enzyme_nocut_1 = \n' +
    'search_enzyme_sense_1 = C\n' +
    'allowed_missed_cleavage_1 = 2\n'
)

TRYPTIC_CLEAVAGE = (
    'search_enzyme_name_1 = trypsin\n' +
    'search_enzyme_cut_1 = KR\n' +
    'search_enzyme_nocut_1 = \n' +
    'search_enzyme_sense_1 = C\n' +
    'allowed_missed_cleavage_1 = 2\n'
)

MODIFICATIONS_STANDARD = (
    'variable_mod_01 = 15.9949 M 3\n'
)

MODIFICATIONS_EXTENDED = (
    'variable_mod_01 = 15.9949 M 3\n' +
    'variable_mod_02 = 57.0215 C 3\n' +
    'variable_mod_03 = 42.0106 [^ 1\n' +
    'variable_mod_04 = 0.9840 NQ 3\n' +
    'variable_mod_05 = 119.0041 C 3\n'
)

CARBA_FIXED = 'add_C_cysteine = 57.0215'
NO_CARBA_FIXED = 'add_C_cysteine = 0.0'


def get_proteins(protein_file, all_ids, all_proteins):
    """ Function to retrieve protein IDs and names from a file.

    Parameters
    ----------
    protein_file : str
        Path to a protein fasta file.
    all_ids : list of str
        A list of the IDs of the proteins in the file.
    all_proteins : list of str
        A list of the sequences of the proteins in the file.

    Returns
    -------
    all_ids : list of str
        A list of the IDs of the proteins in the file.
    all_proteins : list of str
        A list of the sequences of the proteins in the file.
    """
    with open(protein_file, mode='r', encoding='UTF-8') as fasta_f:
        protein = ''
        while (line := fasta_f.readline()):
            if line.startswith('>'):
                all_ids.append(line[1:].strip('\n'))
                if protein:
                    all_proteins.append(protein)
                    protein = ''
            else:
                protein += line.strip('\n')
        all_proteins.append(protein)

    return all_ids, all_proteins


def write_search_proteome(proteome, output_folder, contams_db):
    """ Write new proteome file with decoys.

    Parameters
    ----------
    proteome : str
        Path to a proteome fasta file.
    output_folder : str
        Path to the inSPIRE output folder.
    contams_db : str
        Path to a contaminants fasta file to be appended to the search.
    """
    all_ids = []
    all_proteins = []

    all_ids, all_proteins = get_proteins(proteome, all_ids, all_proteins)
    if contams_db is not None:
        all_ids, all_proteins = get_proteins(contams_db, all_ids, all_proteins)

    # Copy original proteome file.
    shutil.copyfile(proteome, f'{output_folder}/search_proteome.fasta')
    if contams_db is not None:
        with open(
            f'{output_folder}/search_proteome.fasta',
            mode='a+',
            encoding='UTF-8',
        ) as search_file:
            with open(contams_db, mode='r', encoding='UTF-8') as cont_file:
                search_file.write(cont_file.read())

    # Append decoys to copy of original proteome.
    with open(
        f'{output_folder}/search_proteome.fasta',
        mode='a',
        encoding='UTF-8',
    ) as out_file:
        for prot_id, prot_seq in zip(all_ids, all_proteins):
            if not prot_id.startswith('rev_'):
                rev_sequence = prot_seq[::-1]
                out_file.write(
                    f'>rev_{prot_id}\n{rev_sequence}\n'
                )


def write_fragger_params(config, fragger_params_template):
    """ Functiont to write MSFragger parameters.

    Parameters
    ----------
    config : inSPIRE.config.Config
        Config object for the experiment.
    fragger_params_template : str
        Path to the template for MSFragger parameters.
    """
    if config.mz_units == 'Da':
        ms2_units = 0
    else:
        ms2_units = 1

    if config.enzyme == 'trypsin':
        cleavage = TRYPTIC_CLEAVAGE
    else:
        cleavage = UNSPECIFIC_CLEAVAGE

    if config.fragger_mods == 'standard':
        modifications = MODIFICATIONS_STANDARD
        carba_fixed = CARBA_FIXED
    else:
        modifications = MODIFICATIONS_EXTENDED
        carba_fixed = NO_CARBA_FIXED

    with open(
        fragger_params_template,
        mode='r',
        encoding='UTF-8',
    ) as frag_template_file:
        fragger_params = frag_template_file.read().format(
            search_database=f'{config.output_folder}/search_proteome.fasta',
            ncpus=config.n_cores,
            precursor_tolerance=config.ms1_accuracy,
            fragament_tolerance=config.mz_accuracy,
            fragment_units=ms2_units,
            top_n_candidates=10,
            cleavage_parameters=cleavage,
            fragger_modifications=modifications,
            carba_fixed=carba_fixed,
        )

    with open(
        f'{config.output_folder}/fragger.params',
        mode='w',
        encoding='UTF-8',
    ) as params_file:
        params_file.write(fragger_params)


def clean_up_fragger(config):
    """ Function to clean up files after MSFragger execution.

    Parameters
    ----------
    config : inSPIRE.config.Config
        Config object for the experiment.
    """
    with open(
        f'{config.output_folder}/fragger_searches.txt',
        mode='w',
        encoding='UTF-8',
    ) as frag_out:
        for s_file in os.listdir(config.scans_folder):
            if s_file.endswith('.pepXML'):
                frag_out.write(f'{config.scans_folder}/{s_file}\n')

    # Remove intermediate files
    if config.scans_format == 'mgf':
        mzml_files = [
            f'{config.scans_folder}/{scan_f}' for scan_f in os.listdir(config.scans_folder)
            if scan_f.lower().endswith('mzML')
        ]
        for mzml_file in mzml_files:
            os.remove(mzml_file)

def execute_msfragger(config):
    """ Function to run an MSFragger search with inSPIRE default settings.

    Parameters
    ----------
    config : inSPIRE.config.Config
        Config object for the experiment.
    """
    home = str(Path.home())
    fragger_params_template = config.fragger_params
    if config.fragger_use_contams:
        contams_db = (
            f'{home}/inSPIRE_models/utilities/contaminants_20120713.fasta'
        )
    else:
        contams_db = None
    frag_pipe_script_path = (
        f'{home}/inSPIRE_models/utilities/fragpipe_ms_fragger_script.py'
    )
    raw_files_for_fragger = ' '.join([
        f'{config.scans_folder}/{scan_f}' for scan_f in os.listdir(config.scans_folder)
        if scan_f.lower().endswith('.raw')
    ])

    write_search_proteome(config.proteome, config.output_folder, contams_db)
    write_fragger_params(config, fragger_params_template)
    experiment_title = config.experiment_title.replace(' ', '_')

    os.system(
        f'{sys.executable} {frag_pipe_script_path} {config.fragger_db_splits} ' +
        f' "java -Xmx{config.fragger_memory}g -jar" ' +
        f' {config.fragger_path} {config.output_folder}/fragger.params {experiment_title} ' +
        f' {raw_files_for_fragger} > {config.output_folder}/fragger.log '
    )

    clean_up_fragger(config)
