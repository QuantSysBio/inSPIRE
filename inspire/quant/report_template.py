""" Functions for generating the html report at the end of the inSPIRE
    pipeline.
"""
import os
import webbrowser

import pandas as pd

from inspire.constants import ENDC_TEXT, OKCYAN_TEXT

def create_quant_report(config):
    """ Function to create the final html report and open it in the brower.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the whole pipeline.
    figures : dict
        A dictionary containing all of the required plots.
    """
    out_path = os.path.abspath(config.output_folder)

    base_table = """<table style="width:40%">
        <tr>
            <th>Original Name</th>
            <th>Renamed</th>
            <th>Sample</th>
        </tr>
    """
    meta_df = pd.read_csv(
        f'{config.output_folder}/quant/metadata.csv'
    )
    if 'sample' not in meta_df.columns:
        meta_df['sample'] = meta_df['renamed'].str.replace('File', 'Sample')

    for _, df_row in meta_df.iterrows():
        base_table += f'''
            <tr>
                <td>{df_row['source']}</td>
                <td>{df_row['renamed']}</td>
                <td>{df_row['sample']}</td>
            </tr>
        '''

    with open(f'{out_path}/img/norm_correlation.svg', mode='r', encoding='UTF-8') as in_f:
        norm_corr = in_f.read()

    with open(f'{out_path}/img/raw_correlation.svg', mode='r', encoding='UTF-8') as in_f:
        raw_corr = in_f.read()

    with open(f'{out_path}/img/quant_clustermap.svg', mode='r', encoding='UTF-8') as in_f:
        quant_clustermap = in_f.read()

    with open(f'{out_path}/img/quant_pca.svg', mode='r', encoding='UTF-8') as in_f:
        quant_pca = in_f.read()

    with open(f'{out_path}/img/quant_distro.svg', mode='r', encoding='UTF-8') as in_f:
        quant_distro = in_f.read()


    base_table += '</table>'

    html_string = ('''
    <html>
        <head>
            <link 
                rel="stylesheet"
                href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css"
            >
            <style>
                body{
                    font-family: Helvetica;
                    margin:0 100;
                    padding-bottom: 50px;
                    background:whitesmoke;
                }
                h2{
                    color: firebrick;
                    font-family: Helvetica;
                }
                h3{
                    color: firebrick;
                    font-family: Helvetica;
                }
                h2:after
                {
                    content:' ';
                    display: block;
                    border:0.25px solid #696969;
                    position: absolute;
                    width: 60%;
                    margin-top: 2px;
                    left: 20%;
                }
                table {
                    font-family: Helvetica;
                    width: 60%;
                    border: 2px solid #696969;
                }
                th, td {
                    border: 1px solid #696969;
                    padding: 2px;
                }
            </style>
        </head>
        <body>
            <center>
            <h2>inSPIRE Quantification Report for ''' + config.experiment_title + '''</h2>
            </center>
            <h3>
                Raw Files Analysed
            </h3>
            <p>
                ''' + base_table + '''
            </p>
            <h3>
                Correlation between Files:
            </h3>
            <p>
                The figure shows the correlation between raw intensities. You should see particularly
                high correlation if you have technical replicates.
            </p>
        ''' + raw_corr +
        '''
                <br><br>
                <p>
                    The figure shows the correlation between normalised intensities. As with the
                    non-normalised intensities you should see highest correlation between
                    technical replicates. You may also see some increase in correlation across
                    files after normalisation.
                </p>
        ''' + norm_corr +
        '''
                <h3>
                    Distributions across files:
                </h3>
                <p>
                    This figure shows the distributions of abundances as quantified
                    by Skyline before and after normalisation. While the raw intensities
                    may be higher or lower for individual files, the normalised
                    intensities should all be centered around the same point.
                </p>
        ''' + quant_distro +
        '''
                <h3>
                    Clustering over files:
                </h3>
                <p>
                    The figure the raw files after principal component analysis of normalised
                    peptide intensities. We should see technical replicates clustering
                    together. Furthermore, we hope to see separation between infected and
                    control files in the case of inSPIRE-Pathogen.
                </p>
        ''' + quant_pca + '''
                <br><br>
                <p>
                    The figure shows a clustering heatmap over normalised intensities.
                </p>
        ''' + quant_clustermap
    )
    if os.path.exists(f'{out_path}/img/peptide_volcano.svg'):
        with open(f'{out_path}/img/peptide_volcano.svg', mode='r', encoding='UTF-8') as in_f:
            pep_volcano = in_f.read()
        html_string += (
            '''
                    <br><br>
                    <h3>
                        Volcano Plots of Up and Downregulated Peptides
                    </h3>
                    <p>
                        This plot shows up and downregulation on the peptide level. These results are
                        also available in csv format in the inSPIRE quantitative outputs.
                    </p>
            ''' + pep_volcano
        )

    output_path = f'{config.output_folder}/quant/inspire-quant-report.html'
    with open(output_path, 'w', encoding='UTF-8') as output_file:
        output_file.write(html_string)

    print(
        OKCYAN_TEXT +
        '\tReport generated.' +
        ENDC_TEXT
    )

    if not config.silent_execution:
        webbrowser.open(
            'file://' + os.path.realpath(output_path),
            new=2
        )
