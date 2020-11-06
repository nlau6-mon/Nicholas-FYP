# Nicholas-FYP - Mineral Classification Using SWIR Hyperspectral Data

This is a repository to host my Final Year Project on hyperspectral mineral classification.

Included as part of this repository is the ipynb notebook obtained from Colab

Please also refer to important information below.

&nbsp;
&nbsp;
### Colab Notebook Direct Link
For visualization purposes the Colab notebook link is included below:

https://colab.research.google.com/drive/1LsugVl_VTvdXOHbV0-QqnBzKt5UVbODe?authuser=1#scrollTo=eBanU_33tf_4

&nbsp;
&nbsp;
### Usage Notes
abcdefgh

&nbsp;
&nbsp;

## TRAINING DATA NOTE:

The training data is too large to host on github and hence has been provided as a link to a google drive location, with a subsquent link provided for the labels. All Other Files can be found as part of this repository

Training Data - https://drive.google.com/file/d/1cDZizT5889xOdjsxBHBE8l7wZHtqB_Ar/view?usp=sharing

Training Label - https://drive.google.com/file/d/1Je42yMgTv7RVdhMpC8VyoU6Kv_80BZy9/view?usp=sharing

&nbsp;

### Data Files
Train_Data_Sep20 = Training Data collected from Scientific Data (Acces via Google Drive link)

Train_Label_Sep20 = Training Labels collected from Scientific Data (Access via Google Drive link)

210_0004_NoFilter_Reduced = Unlabelled Borehole Data. Comes as 2D and has had bands reduced to 246

BU_Data = Data collected from the Planteray Data System (and also includes Brown University data included for the PDS project). Has had bands reduced to 246

labels_BU = Associated labels with above BU/PDS Dataset

CSIRO_data = Data collected from CSIRO. Has had bands reduced to 246

labels_CSIRO = Associated labels with above CSIRO Dataset

JPL_data = Data collected from the Jet Propulsion Labratory. Has had bands reduced to 246

labels_JPL = Associated labels with above JPL dataset

USGS_data_update = Data collected from the USGS. Has had bands reduced to 246

labels_USGS_update = Associated labels with above USGS dataset

Reducedband_Data_Drillhole = Data collected from NCVL drillhole. Has had bands reduced to 246

Label_Data_Drillhole = Associated labels with above NCVL data

SAM_Array = Array for Spectral Angle Mapper. Consists of 30 USGS Spectras

SAM_Labels = Associated labels from above SAM_Array

SAM_Array_Test = Array for Testing Spectral Angle Mapper. Consists of USGS, PDS, JPL and CSIRO data minus the 30 USGS reference spectra

SAM_Labels_Test = Associated labels from above SAM_Array_Test

&nbsp;
&nbsp;
### Unfitted Data
This section consists of all 799 test spectras which have not had their bands altered. It Has not been used in the final project and its inclusion is present if bands want to be manipulated to a different fitting

&nbsp;
&nbsp;
### Functions Used Folder
This folder consists of various functions used to process and vizualise data. Most these functions were not used beyond August as the project moved to Colab.
