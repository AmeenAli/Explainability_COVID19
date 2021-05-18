# Explainability_COVID19
Official Code Implementation of the paper : Explainability Guided Multi-Site Covid-19 CT Classification
<p align="center">
  <img  src="https://raw.githubusercontent.com/AmeenAli/Explainability_COVID19/main/images/1.png?token=ABU4KO77VHNAIZ6BNB5IYWLAVT2DC">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/AmeenAli/Explainability_COVID19/main/images/2.png?token=ABU4KO7Z6EDS7XFREHHU4LLAVT2GY">
</p>

# Datasets :
We employ three publicly available COVID-19 CT datasets:
1. SARS-CoV-2 dataset .
2. COVID-CT dataset .
3. COVIDx-CT .

The first two datasets can be downloaded using the following google drive link :
<br>
https://drive.google.com/file/d/1JBp9RH9-yBEdtkNYDi6wWL79o62JD5Td/view
<br>
The third dataset can be downloaded through :
<br>
https://github.com/haydengunraj/COVIDNet-CT/blob/master/docs/dataset.md
<br>

# Training

The base training command takes the following form :

python main.py --b BATCHSIZE --lr LEARNINGRATE --dataset DATASET --mixmethod MIXMETHOD .
<br>
if you want to use SnapMix is the main Augmentation tool, then you have to set MIXMETHOD = snapmix, otherwise MIXMETHOD=baseline.
<br>
<b>
 For reproducing the results of tables 1 & 2, please run the following command :
  
 For reproducing the results of tables 3 , please run the following command :
