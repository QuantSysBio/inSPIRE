""" Functions for predicting binding affinity using NetMHCpan
"""
from multiprocessing import Pool
import os

import docker


def predict_binding(config):
    """ Function to run binding affinity prediction using NetMHCpan.
    """
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
            if config.pan_docker:
                pan_command = config.pan_command
                if pan_command.endswith('/netMHCpan'):
                    pan_command = pan_command[:-10]
                function_args.append(
                    f'docker run -v {os.path.abspath(pan_command)}:/root/netMHCpan-4.1 ' +
                    f'-v {os.path.abspath(config.output_folder)}:/root/output -e PAN_ARGS=' +
                    f'"-BA -inptype 1 -a {allele} -l {pep_len} -p ' +
                    f'-f /root/output/mhcpan/{input_file}" johncormican/basic-netmhcpan ' +
                    f'> {os.path.abspath(config.output_folder)}/mhcpan/{output_file}\n'
                )
            else:
                function_args.append(
                    f'{config.pan_command} -BA -inptype 1 -a {allele} -l {pep_len} -p -f ' +
                    f'{config.output_folder}/mhcpan/{input_file} > ' +
                    f'{config.output_folder}/mhcpan/{output_file}'
                )

    # print(function_args)
    with Pool(processes=config.n_cores) as pool:
        pool.map(os.system, function_args)
