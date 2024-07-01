# Exporting PEAKS search results for inSPIRE

This provides a quick guide on exporting PEAKS search results for inSPIRE.

## Before the search

The key setting before you run your PEAKS search is to Estimate FDR with decoy-fusion:

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_estimate_fdr.png" alt="drawing" width="600"/>

## After the search

Upon completion of the PEAKS search, it is important to update the export thresholds from the default -10lgP threshold of 15

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_default.png" alt="drawing" width="600"/>

to the correct threshold of 0:

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_correct.png" alt="drawing" width="600"/>


One should also ensure that decoy peptides are exported by select the Preferences menu:

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_preferences.png" alt="drawing" width="600"/>

and clicking Show Decoy Hits under the Display Options menu:

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_show_decoy.png" alt="drawing" width="600"/>

When exporting search results, the required table for inSPIRE is DB search psm.csv which can be exported under the Text Formats section:

<img src="https://raw.githubusercontent.com/QuantSysBio/inSPIRE/master/img/peaks_export_tables.png" alt="drawing" width="600"/>
