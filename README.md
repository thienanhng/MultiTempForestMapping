# MultiTempForestMapping
Paper: [Multi-temporal forest monitoring in the Swiss Alps with knowledge-guided deep learning, Thiên-Anh Nguyen, Marc Rußwurm, Gaston Lenczner, Devis Tuia, Remote sensing of environment, 2024](https://www.sciencedirect.com/science/article/pii/S0034425724001202)

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

# Results

[Download results](https://drive.google.com/drive/folders/1isRYaBt6GJT0NkXpsEWCSQtV8wgbDgEy?usp=sharing)

[View results with Google Earth Engine](https://temp-forest-mapping.projects.earthengine.app/view/multitempforestmap)


