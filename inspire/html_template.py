""" Functions for generating the html report at the end of the inSPIRE
    pipeline.
"""
import os
import webbrowser

from inspire.constants import ENDC_TEXT, OKCYAN_TEXT

def create_html_report(config, figures):
    """ Function to create the final html report and open it in the brower.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the whole pipeline.
    figures : dict
        A dictionary containing all of the required plots.
    """
    config_str = str(config)
    html_string = ('''
    <html>
        <head>
            <link 
                rel="stylesheet"
                href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css"
            >
            <style>
                body{
                    margin:0 100;
                    padding-bottom: 50px;
                    background:whitesmoke;
                }
                h2{
                    color: firebrick;
                }
                h3{
                    color: firebrick;
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
            <h2>inSPIRE Report for ''' + config.experiment_title + '''</h2>
            </center>
            <h3>
                inSPIRE Settings Used
            </h3>
            <p>
                ''' + config_str + '''
            </p>
            <h3>
                Selected Features and Importance
            </h3>
            <p>
                The table below shows the importance of the final feature set used by
                percolator. Strong positive values (highlighted in green) may indicate that
                higher feature values are more common among target PSMs while strongly negative
                values (highlighted in red) may indicate that higher feature values are more
                common among decoy PSMs. However, it is also possible that a feature like
                searchEngineScore ends up with a negative coefficient simply because it
                is so strongly correlated to a more powerful feature like deltaScore which
                has a strong positive coefficent.
            </p>
            <center>
    ''' + figures['table'] +
    '''
            </center>
            <h3>
                Feature Distributions
            </h3>
            <p>
                These Violin Plots show the distributions of the three most heavily positive
                and heavily negative weighted features for accepted and rejected PSMs.
            </p>
            <center>
    ''' + figures['violin_fig'])

    html_string += (
        '''
                </center>
                <h3>
                    inSPIRE Performance: Number of PSMs Identified
                </h3>
                <p>
                    This shows the number of PSMs discovered by inSPIRE compared to the original
                    search engine for q-value cut offs between 0.01 and 0.1.
                </p>
                <center>
        ''' + figures['psms_fig']
    )

    if config.use_binding_affinity in ['asValidation', 'asFeature']:
        html_string += ('''
                </center>
                <h3>
                    inSPIRE Performance: Percentage Binders Identified
                </h3>
                <p>
                    This shows the percentage of HLA-I binders as predicted by NetMHCpan among
                    the PSMs identified by inSPIRE compared to the original search engine
                    results. Note that this plot is more meaningful if you have set
                    useBindingAffinty to asValidation, as if it is set to asFeature it is
                    unsurprising that inSPIRE would produce a higher percentage of HLA binders.
                </p>
                <center>
        ''' + figures['binders_fig']
    )

    html_string += ('''
            </center>
        </body>
    </html>
    ''')

    output_path = f'{config.output_folder}/inspire-report.html'
    with open(output_path, 'w', encoding='UTF-8') as output_file:
        output_file.write(html_string)

    print(
        OKCYAN_TEXT +
        '\tReport generated.' +
        ENDC_TEXT
    )

    webbrowser.open(
        'file://' + os.path.realpath(output_path),
        new=2
    )
