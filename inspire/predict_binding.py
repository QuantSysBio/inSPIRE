""" Functions for predicting binding affinity using NetMHCpan
"""
from multiprocessing import Pool
import os

def predict_binding(config):
    """ Function to run binding affinity prediction using NetMHCpan.
    """

    # Run first command so docker image only pulled once.
    if config.pan_docker:
        os.system('docker image pull johncormican/basic-pan-execution')
        alleles_string = ','.join(config.alleles)
        pan_command = config.pan_command
        if pan_command.endswith('/netMHCpan'):
            pan_command = pan_command[:-10]

        os.system(
            f'docker run --rm -v {os.path.abspath(pan_command)}:/net/sund-nas.win.dtu.dk' +
            '/storage/services/www/packages/netMHCpan/4.1/netMHCpan-4.1 ' +
            f'-v {os.path.abspath(config.output_folder)}:/root/output -e ALLELES="{alleles_string}"' +
            f' -e PRED_LIMIT={config.ba_pred_limit} -e N_CORES={config.n_cores} ' +
            'johncormican/basic-pan-execution '
        )

        return

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
            function_args.append(
                f'{config.pan_command} -BA -inptype 1 -a {allele} -l {pep_len} -p -f ' +
                f'{config.output_folder}/mhcpan/{input_file} > ' +
                f'{config.output_folder}/mhcpan/{output_file}'
            )

    with Pool(processes=config.n_cores) as pool:
        pool.map(os.system, function_args)
