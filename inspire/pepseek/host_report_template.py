""" Functions for generating the html report at the end of the PEPSeek
    pipeline.
"""
import os
import webbrowser

from inspire.constants import ENDC_TEXT, OKCYAN_TEXT

def create_host_report(config):
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
    with open(f'{out_path}/img/PEPSeek_host_bar_plot.svg', mode='r', encoding='UTF-8') as in_f:
        bar_plot = in_f.read().strip('\n')

    with open(f'{out_path}/img/PEPSeek_host_metrics.svg', mode='r', encoding='UTF-8') as in_f:
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
            <h2>PEPSeek Host Peptide Report for ''' + config.experiment_title + '''</h2>
            </center>
            <p>
                Note quantitative information on host peptide changes upon infection can be
                found in the quantitative report.
            </p>
            <h3>
                Host Peptide Candidates Found
            </h3>
            <p>
                This figure compares the number of peptides identified by both the
                search engine and PEPSeek against peptides identified by PEPSeek alone.
            </p>
            <br>
             ''' + bar_plot +
    '''
            <h3>
                MS2 Quality Metrics
            </h3>
            <p>
                This figure shows distribution of spectral angle, engine score, and
                retention time prediction error on the host peptide identifications.
                For spectral angle and retention time prediction error we should see a
                similar distribution. On engine score, the PEPSeek only peptides likely
                score lower.
            </p>
            <br>
    ''' + swarm_plot
    )
    if os.path.exists(f'{out_path}/img/logo_comp_plots.svg'):
        with open(f'{out_path}/img/logo_comp_plots.svg', mode='r', encoding='UTF-8') as in_f:
            logo_plot = in_f.read().strip('\n')
        html_string += (
            '''
                    <h3>
                        Pathogen vs. Host Amino Acid Frequency:
                    </h3>
                    <p>
                        This chart shows the JS divergence between amino acid frequencies in host
                        and pathogen peptides. Amino acids on the positive y-axis are
                        overrepresented in the pathogen peptides compared to the host,
                        while amino acids on the negative y-axis
                        are overrepresented in the host peptides compared to the pathogen.
                        This plot can provide insight into the differing characteristics
                        of pathogen and host peptides.
                    </p>
            ''' + logo_plot
        )

    if os.path.exists(f'{out_path}/img/PEPSeek_affinity_cluster.svg'):
        with open(
            f'{out_path}/img/PEPSeek_host_affinity_cluster.svg',
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
                    The predicted binding affinities for the identified host peptides. This can
                    help see you the distribution of predicted binders for each allele in
                    your host cell. Colour is based on NetMHCpan predicted binding affinty
                    (percentage rank).
                </p>
            ''' + aff_clust
        )

    output_path = f'{config.output_folder}/PEPSeek/pepseek-host-report.html'
    with open(output_path, 'w', encoding='UTF-8') as output_file:
        output_file.write(html_string)

    print(
        OKCYAN_TEXT +
        '\tPEPSeek host report generated.' +
        ENDC_TEXT
    )

    if not config.silent_execution:
        webbrowser.open(
            'file://' + os.path.realpath(output_path),
            new=2
        )
