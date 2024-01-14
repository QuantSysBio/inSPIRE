""" Functions for generating the html report at the end of the inSPIRE
    pipeline.
"""
import os
import webbrowser

import pandas as pd

from inspire.constants import ENDC_TEXT, OKCYAN_TEXT

def create_epitope_report(config):
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
            <th>Peptide</th>
            <th>Proteins</th>
        </tr>
    """
    ep_df = pd.read_csv(
        f'{config.output_folder}/epitope/potentialEpitopeCandidates.csv'
    )

    for _, df_row in ep_df.iterrows():
        base_table += f'''
            <tr>
                <td>{df_row['peptide']}</td>
                <td>{df_row['protein']}</td>
            </tr>
        '''

    with open(f'{out_path}/img/epitope_bar_plot.svg', mode='r', encoding='UTF-8') as in_f:
        bar_plot = in_f.read().strip('\n')

    with open(f'{out_path}/img/epitope_metrics.svg', mode='r', encoding='UTF-8') as in_f:
        swarm_plot = in_f.read()

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
            <h2>inSPIRE Epitope Candidate Report for ''' + config.experiment_title + '''</h2>
            </center>
            <h3>
                Epitope Candidates Found
            </h3>
            <p>
                Below is a full table of the pathogen peptides identified and their associated
                proteins.
            </p>
            <p>
                ''' + base_table + '''
            </p>
            <h3>
                Shared and inSPIRE Only Peptides:
            </h3>
            <p>
                This figure compares the number of peptides identified by both the
                search engine and inSPIRE against peptides identified by inSPIRE alone.
            </p>
            <br>
             ''' + bar_plot +
    '''
            <h3>
                MS2 Quality Metrics
            </h3>
            <p>
                This figure shows distribution of spectral angle, engine score, and
                retention time prediction error on the pathogen peptide identifications.
                For spectral angle and retention time prediction error we should see a
                similar distribution. On engine score, the inSPIRE only peptides likely
                score lower.
            </p>
            <br>
    ''' + swarm_plot
    )
    if os.path.exists(f'{out_path}/img/epitope_pca.svg'):
        with open(f'{out_path}/img/epitope_pca.svg', mode='r', encoding='UTF-8') as in_f:
            ep_pca = in_f.read().strip('\n')
        html_string += (
            '''
                    <h3>
                        Quantitative Data:
                    </h3>
                    <p>
                        Dimensionality reduction applied to quantiative data from the pathogen
                        peptide and a random seleciton of host peptides below. Ideally we should
                        see some clustering effects with pathogen peptides mostly close together.
                        This is likely not a perfect clustering as there is noise in the
                        label free quantification, however it can be a useful indicator.
                    </p>
            ''' + ep_pca
        )

    if os.path.exists(f'{out_path}/img/epitope_affinity_cluster.svg'):
        with open(
            f'{out_path}/img/epitope_affinity_cluster.svg',
            mode='r',
            encoding='UTF-8',
        ) as in_f:
            aff_clust = in_f.read().strip('\n')
        html_string += (
            '''
                <h3>
                    Binding Affinity Predictions:
                </h3>
                <p>
                    The predicted binding affinities for the identified peptides. This can
                    help see you the distribution of predicted binders for each allele in
                    your host cell.
                </p>
            ''' + aff_clust
        )

    html_string += (
    '''
        <h3>
            MS2 Spectral Plots:
        </h3>
        <p>
            These are the MS2 spectra based on which the peptides were assigned. Inspection
            of the spectra can be informative and increase your confidence in peptides
            identified.
        </p>
        <embed src="
    ''' + f'{config.output_folder}/epitope/spectralPlots.pdf" width=1000 height=2000>'
    )


    output_path = f'{config.output_folder}/epitope/inspire-epitope-report.html'
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
