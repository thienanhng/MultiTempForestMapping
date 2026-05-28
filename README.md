# MultiTempForestMapping

This repository corresponds to the paper [Multi-temporal forest monitoring in the Swiss Alps with knowledge-guided deep learning, Thiên-Anh Nguyen, Marc Rußwurm, Gaston Lenczner, Devis Tuia, Remote sensing of environment, 2024](https://www.sciencedirect.com/science/article/pii/S0034425724001202)

The goal is to segment forest cover from time series of aerial images and a Digital Elevation Model (DEM), with forest cover annotations available for the most recent images only. 

# Method

The model is first pre-trained in a fully supervised manner using the most recent images, the DEM and their corresponding forest cover labels:

<img width="590" height="83" alt="overall_model_diagram_pretrain_white_background" src="https://github.com/user-attachments/assets/cf6af165-5c0f-4052-85b5-2d25aacd531b" />


The pre-trained feature extractor is then combined with a temporal module (a customized GRU) in a multi-temporal framework. A temporal loss comparing subsequent predictions is used in combination with the supervised segmentation loss to train this model:

<img width="590" height="489.17" alt="overall_model_diagram_white_background" src="https://github.com/user-attachments/assets/338ee1bd-b456-4874-bd54-62a7c34cb57e" />

The temporal loss is based on common sense knowledge about forest loss and gain dynamics (see paper for mathematical formulation). The following figure illustrates how different scenarios of pairs subsequent forest cover predictions (green areas) affect the loss value:

<img width="1942" height="1424" alt="lca_intuition_white_background" src="https://github.com/user-attachments/assets/c7a23b46-efa5-4fe9-bfd5-e2abc0b280d2" />


# Dataset
**Input data**
- SwissImage aerial imagery 
  - [1946](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-hist-1946) (not open access to this day)
  - [1947-1997](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-hist) (not open access to this day)
  - [1998-current](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10) (not fully open access to this day)
- [SwissALTI3D digital elevation model](https://www.swisstopo.admin.ch/en/height-model-swissalti3d)

**Training labels**: rasterized labels ([download](https://drive.google.com/file/d/14ut2kYcFPirWk-iPICXvP7mQqSrihTAp/view?usp=sharing)) extracted from [SwissTLM3D](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d) (2022-03 release)

**Evaluation labels**: manually annotated tiles from random locations and dates (1946 to 2020) ([download](https://drive.google.com/file/d/1BOsuv77L9bJnJMQWAD01yjeJMcVAu6VZ/view?usp=sharing)) 

# Trained models

Download trained models [here](https://drive.google.com/file/d/1RPdPxcpn2PXkCBMC60G0bONQlx1xgRfA/view?usp=sharing)

# Running the code

You can reproduce the training experiments by running [launch_nontemp_array.py](launch_nontemp_array.py) (pre-training) and [launch_temp_array.py](launch_temp_array.py) (multi-temporal finetuning).

To perform inference with a specific model you have trained or downloaded, run [infer.py](infer.py) after specifying the correct model name and path [line 285](https://github.com/thienanhng/MultiTempForestMapping/blob/bbaf79fbfe07a93fc4d11f55315346ec265c76fe/infer.py#L285).  

# Results

All the scripts used to obtain metrics and figure showed in the paper are in the .[analysis/](analysis/) directory.

[Download results (forest cover segmentation maps)](https://drive.google.com/drive/folders/1isRYaBt6GJT0NkXpsEWCSQtV8wgbDgEy?usp=sharing)

[View results with Google Earth Engine](https://temp-forest-mapping.projects.earthengine.app/view/multitempforestmap)
<img width="1157" height="787" alt="gee_screenshot" src="https://github.com/user-attachments/assets/686c99d1-bda6-4735-bc89-60d0ed89a52c" />



