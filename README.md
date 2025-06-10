# Forest Cover Type Prediction with Machine Learning
### This project implements a Machine Learning script to predict forests' cover types (*what species of trees thrive in a certain area*) based on cartographic and environmental data using the [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype) from the UCI Machine Learning Repository.
##### Note: *If the link to the UCI's dataset doesn't work, the reason could be dued to the removal of the dataset from their website.* The script will work anyway since i uploaded the entire dataset in a CSV file.

---
## Purpose
#### The goal of this project is to train and evaluate a machine learning model capable of predicting forests' cover types based on geospatial and environmental features such as elevation, slope, aspect, distances to hydrology and roads, and soil classification.

forAlthough physical and environmental attributes of land areas are widely available through GIS and remote sensing, **up-to-date forest cover labels are not always available — especially in remote or infrequently surveyed regions**. This project's goal is to fill that gap by training a supervised model that can predict forests' types using available geospatial data.

#### Supporting References
- *“Forest remote sensing data worldwide, especially acquired from sensors on satellites, are abundant and generally easily obtainable. In sharp contrast, there is a scarcity of ground-based data, including labels or annotations.”*  
[OpenForest: A data catalogue for machine learning in forest monitoring(2025)](https://www.cambridge.org/core/journals/environmental-data-science/article/openforest-a-data-catalog-for-machine-learning-in-forest-monitoring/F62FBEADFF8E3A10C6EDA789D7D180C6)

- *“Reliable data annotation requires field-surveys. Therefore, full segmentation maps are expensive to produce, and training data is often sparse, point-like, and limited to areas accessible by foot.”*
[Habitat classification from satellite observations with sparse annotations (2022)](https://arxiv.org/abs/2209.12995?utm_source=chatgpt.com)



---

## Dataset Source
**Dataset Name**: Covertype  
**Source**: UCI Machine Learning Repository  
**Original Paper**: Blackard, J.A. and Dean, D.J. (1999)  
**Link**: [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype)  
**Note**: The project is based on features and data taken in the **Roosevelt National Forest** of northern Colorado, but can be adapted to any territory if similar features are available (even using less features than the original dataset). 

The dataset contains **581,012 instances** and **54 features**, including:
- **Quantitative features**: elevation, aspect, slope, distances to water/roads/fire, and hillshade at various times.
- **Categorical features**: 
  - 4 One-hot-encoded wilderness area indicators
  - 40 One-hot-encoded soil type indicators derived from the **STATSGO soil map** by the USGS
- **Target**: Forest cover type (1–7), the trees species are Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz.
---

## Problem Statement
Although GIS and remote sensing data can provide physical and environmental attributes of land areas, **forest cover labels are often missing or outdated**. This project aims to address this gap by training a supervised model capable of predicting cover types using easily available GIS layers, with the final goal to predict the missing cover labels.

---
## Usage
This project implement a preliminary split to decide the number of samples you want to work with (based on your computational resources).
- **FULL DATASET**: For the whole 581,012 samples dataset just type `max`.
- **CUSTOMIZED DATASET'S SIZE**: Input the desired number of samples you want to work with.

### Hyperparameter Tuning Workflow

The training pipeline uses a `GridSearchCV` for tuning the and finding the best hyperparameters for each model.

### Fitting the best model (final evaluation)
**Choose the model** with the highest accuracy and insert it in `best_model = ` (*line 216*) inluding the ideal parameteres found with the `GridSearchCV`.

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
