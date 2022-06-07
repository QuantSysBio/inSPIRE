# inSPIRE
in silico Spectral Predictor Informed REscoring

## Set Up

Note, inSPIRE requires conda already installed. Once, these are installed follow these steps:

1) To start with create a new conda environment with python version 3.8:

```
conda create --name inspire python=3.8
conda activate inspire
```

2) You will then need to install the inspire package:

```
python setup.py install
```

4) To check your installation, run the following command (it is normal for this call to hang for a few seconds on first execution)

```
inspire -h
```

Once you have successfully installed inSPIRE you must run it specifying your pipeline and a config file.

## Running a small example.

The example folder provides a simple example that you should be able to run immediately. To run the "prepare" pipeline to produce formatted a formatted csv file for Prosit input call:

```
inspire --pipeline prepare --config_file example/config.yml
```

This creates an input file for Prosit at example/output/prositInput.csv. To simplify this example we have already provided the prositPredictions.msp file needed in that folder. This allows you to move immediately to running the rescoring pipeline with:

```
inspire --pipeline rescore --config_file example/config.yml
```

This will run the rescoring with mokapot rather than percolator as it is easier to install and produce a report upon completion (this should take 2-3 minutes).

### Plotting Spectra

To use the inSPIRE plotting utility you will need to further install plotly-orca:

```
conda install -c plotly plotly-orca 
```

Then to to see example pair plots comparing the experimental data to prosit predictions call:

```
inspire --pipeline plotSpectra --config_file example/config.yml
```

This plots the PSMs specified in example/output/plotData.csv and save the plots to example/output/spectralPlots.pdf. To plot a different PSM simply copy the relevant line from example/output/finalAssignments.csv into example/output/finalAssignments.csv and rerun the plotSpectra pipeline.

## inSPIRE Pipelines.

inSPIRE has 2 core pipelines, prepare and rescore. Individual parts of these pipelines may be called separately.

### inspire --pipeline prepare

Calling --pipeline prepare will produce a file "prositInput.csv" in the output folder. If mhcpan is being used a folder called "mhcpan" which contains input files for mhcpan.

Prosit predictions can then either collected from https://www.proteomicsdb.org/prosit/ or if you have a GPU available you can install it yourself from https://github.com/kusterlab/prosit. In either case export the results in msp format.

Once you have Prosit msp output you should put it in the output folder and name the file "prositPredictions.msp".

If you are using NetMHCpan, the prepare pipeline will have created an mhcpan folder in the output folder with all unique peptides of each length saved as "inputLen{length}.txt". These can then be input to NetMHCpan (see https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1). These predictions should be put in the output folder as "outputLen{length}preds.txt".

### inspire --pipeline rescore

Calling --pipeline rescore will then use the predictions to rescore all PSMs.

## inSPIRE Config File

The other argument required for execution is --config_file "path/to/configFile.yml". yaml is a relatively simple format for entering many configs. The yaml file should take the form:

```
---
key1: value1
key2: value2
...
```

The keys needed are detailed below.

### Required Configs
These are the minimal configs required to run inSPIRE.

| Key   | Description   |
|-------|---------------|
| experimentTitle  | A title for the experiment.  |
| searchResults    | The file path to the results of your ms search or a list of file paths if using multiple search results. |
| searchEngine     | The search engine used (maxquant, mascot, or peaks). |
| outputFolder     | Specify an output folder location which inSPIRE should write to. |
| scansFolder      | Specify a folder containing the experimental spectra files in mgf or mzML format. |
| scansFormat      | Specify the format of the spectra file (must be either mgf or mzML). |

### Optional Settings (Experiment Specifications, Recommended to Check)
The following setting are set by default but you should check that they are valid for your experimental set up.

| Key   | Description   |
|-------|---------------|
| collisionEnergy  | The mass spectrometer collision energy setting (default=32). |
| mzAccuracy       | The mz accuracy of the mass spectrometer in Daltons (default=0.02). |
| rescoreMethod       | inSPIRE supports either "mokapot" or "percolator" (default=mokapot). |

### Additional Options

These setting are optional and are completely dependent on user preference.

| Key   | Description   |
|-------|---------------|
| falseDiscoveryRate  | This is the false discovery rate Percolator optimises for (default=0.01). |
| excludeFeatures       | This specifies any features which you wish to exclude from rescoring (default=empty list). |
| reduce       | By default inSPIRE uses only the highest scoring hit per scan (and accession group if specified). If you set reduce to False this will consider all hits (default=True). |

### Additional Configs for Mascot Distiller

If you have a combined mgf file from Mascot Distiller, you must add the following configs.

| Key   | Description   |
|-------|---------------|
| scanTitleFormat  | If Distiller used, set this argument to mascotDistiller.  |
| distillerLog  | If Distiller used, set this argument to the path to the "table_peptide_int.txt" file from Distiller.  |
| combinedScansFile | Set this to the combined mgf file from Mascot Distiller and put the mgf file in folder specified by scansFolder. |

### NetMHCpan Configs

NetMHCpan predicts the binding affinity of a peptide for various HLA molecules. inSPIRE can use this as a validation of its predictions or as a feature for rescoring.

| Key   | Description   |
|-------|---------------|
| useBindingAffinity  | Set to asValidation if you want to check the percentage binders in your standard inSPIRE identifications. Set to asFeature if you want to use predicted binding affinity as a rescoring feature. |

### Ground Truth Datasets

Ground Truth Datasets may be useful in assessing the accuracy of your identification method. In this case, peptides with prior labels are measured via MS/MS. Some of these peptides are then inserted into the standard proteome and precision and recall your peptide identification method may be estimated.

To use ground truth datasets with inSPIRE, add the following parameters to the config file.

| Key   | Description   |
|-------|---------------|
| groundTruth  | The file path to your ground truth dataset matching scans to true peptide label.  |
| groundTruthSeqKey    | The name of the column containing the true peptide in the ground truth dataset. |
| groundTruthSourceKey     | The name of the column containing the source file in the ground truth dataset. |
| groundTruthScanKey  | The name of the column containing the scan number in the ground truth dataset. |
| groundTruthAccessionGroupKey*       | The name of the column containing the accession group in the ground truth dataset. |

*Optional argument need only if the labelled peptides represent different accession groups.
