#  Forest Cover Type Prediction with Machine Learning
### This project implements a machine learning script to predict forest cover types based on cartographic and environmental data using the [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype) from the UCI Machine Learning Repository.

---
## Purpose
#### The goal of this project is to train and evaluate a machine learning model capable of predicting forest cover types based on geospatial and environmental features such as elevation, slope, aspect, distances to hydrology and roads, and soil classification.

Although physical and environmental attributes of land areas are widely available through GIS and remote sensing, **up-to-date forest cover labels are not always available — especially in remote or infrequently surveyed regions**. This project's goal is to fill that gap by training a supervised model that can predict cover types using readily available geospatial layers.

### Supporting References
- *"Forest remote sensing data worldwide, especially acquired from sensors on satellites, are abundant and generally easily obtainable. In sharp contrast, there is a scarcity of ground-based data, including labels or annotations."*  
[OpenForest: A data catalogue for machine learning in forest monitoring(2023)](https://openforestdata.org/)

- *"Despite the growing availability of satellite imagery, obtaining reliable and comprehensive ground-truth labels remains a significant challenge, particularly in remote or inaccessible regions, limiting the accuracy and applicabilityTechnologies of supervised land cover classification models."*  
[Data Scarcity in Remote Sensing: Challenges for Supervised Land Cover Mapping (2023)](https://doi.org/10.1016/j.rse.2023.113456)

- *"Land cover characterization and change detection requires an effective classification algorithm which is in turn dependent on adequate training. Gathering comprehensive training data for the whole of the Earth's land surface is clearly very challenging logistically."*  
[Global characterization and monitoring of forest cover using Landsat data: opportunities and challenges (2012)](https://www.sciencedirect.com/science/article/pii/S0034425712002255)



**Note**: The project is based on features and data from the Roosevelt National Forest of northern Colorado, but can be adapted to any territory if similar features are available (even with less features than the original project). 

---

## Dataset Source
**Dataset Name**: Covertype  
**Source**: UCI Machine Learning Repository  
**Original Paper**: Blackard, J.A. and Dean, D.J. (1999)  
**Link**: [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype)  
**Note**: The project is based on features and data from the Roosevelt National Forest of northern Colorado, but it can be adapted to any territory if similar features are available (even with fewer features).

The dataset contains **581,012 instances** and **54 features**, including:
- **Quantitative features**: elevation, aspect, slope, distances to water/roads/fire, and hillshade at various times.
- **Categorical features**: 
  - 4 One-hot-encoded wilderness area indicators
  - 40 One-hot-encoded soil type indicators derived from the **STATSGO soil map** by the USGS
- **Target**: Forest cover type (1–7), including species like spruce/fir, lodgepole pine, ponderosa pine, and others.

---

## Problem Statement
Although GIS and remote sensing data can provide physical and environmental attributes of land areas, **forest cover labels are often missing or outdated**. This project aims to address this gap by training a supervised model capable of predicting cover types using easily available GIS layers, with the final goal to predict the missing cover labels.

---

## Applications
- **Forest management and planning** in areas lacking recent survey data.
- **Ecological modeling** to simulate how land cover may change under environmental shifts.
- **Wildfire risk modeling**, where vegetation types influence spread potential.

---

## Possible Extensions
- **Integration with satellite imagery** (e.g., Sentinel-2 or Landsat) to enhance prediction with visual features.
- **Explainability**: Use SHAP values or feature importance to understand which variables most influence predictions.
- **Geospatial deployment**: Apply the trained model over large-scale raster datasets using tools like GDAL or Rasterio.
- **Real-time classification** using edge devices (e.g., drones or IoT sensors with GIS inputs).

---

## Tools Used
- Python (scikit-learn)
- Pandas, NumPy, Matplotlib, Seaborn
- Git + GitHub
