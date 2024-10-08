# Sisk_DWI_Adversity_sCCA
### Code for DWI/Adversity sCCA project

**DWI_Extract_ICV_ScannerSite** extracts intracranial volume from Freesurfer files and pulls site participant was scanned at from .json files.

**DWI_Create_Dataset** pulls data from various sources, recodes and transforms to prepare for CCA analysis.

**DWI_Motion_Exclusion** pulls motion metrics from QSIPrep output and removes outliers.

**DWI_Create_Jobfiles_Modeling** prepares DSQ batch submission.

**DWI_RI_PMDModels_Iteration** is the code for a single iteration of model fitting.

**DWI_RI_Evaluate_PMD** aggregates model results across iterations and analyzes them.

The **requirements.txt** file lists the package versions used in the present analysis.
