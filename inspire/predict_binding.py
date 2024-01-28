""" Functions for predicting binding affinity using NetMHCpan
"""
from multiprocessing import Pool
import os
from pathlib import Path
import platform

def predict_binding(config):
    """ Function to run binding affinity prediction using NetMHCpan.
    """
    home = str(Path.home())
    singularity_image = f'{home}/inSPIRE_models/utils/singularity_image.sif'

    input_files = [
        in_file for in_file in os.listdir(
            f'{config.output_folder}/mhcpan/'
        ) if in_file.startswith(
            'inputLen'
        )
    ]
    function_args = []
    for allele in config.alleles:
        for input_file in input_files:
            pep_len = int(input_file.split('inputLen')[-1].split('_')[0].split('.')[0])
            if pep_len > config.ba_pred_limit:
                continue
            output_file = input_file.replace(
                f'inputLen{pep_len}', f'output_{pep_len}_{allele}'
            )
            if config.run_singularity:
                function_args.append(
                    'singularity run --bind ' +
                    f'{config.pan_command},{config.output_folder}:/root/netMHCpan-4.1' +
                    f'{singularity_image} -BA -inptype 1 -a {allele} -l {pep_len} -p -f ' +
                    f'/root/netMHCpan-4.1/mhcpan/{input_file} > ' +
                    f'{config.output_folder}/mhcpan/{output_file}'
                )
            else:
                function_args.append(
                    f'{config.pan_command} -BA -inptype 1 -a {allele} -l {pep_len} -p -f ' +
                    f'{config.output_folder}/mhcpan/{input_file} > ' +
                    f'{config.output_folder}/mhcpan/{output_file}'
                )
                


    with Pool(processes=config.n_cores) as pool:
        pool.map(os.system, function_args)
