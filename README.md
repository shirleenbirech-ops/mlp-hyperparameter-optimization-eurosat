# Systematic hyperparameter optimisation and regularisation analysis of an MLP applied to the EuroSAT image dataset


## Project Overview

This project investigates how key MLP hyperparameters influence classification performance on a multi-class remote-sensing dataset. Rather than relying on automated grid search, hyperparameters are tuned manually and evaluated independently to analyse their individual impact on convergence behaviour, generalisation, and model stability.

The study focuses on both performance metrics and internal model properties, including loss dynamics and parameter norms, to better understand the effect of regularisation and optimisation choices.

## Methods & Techniques

- Image preprocessing and downsampling to 32×32 resolution

- Feature standardisation with controlled variance scaling

- Supervised learning using MLPClassifier

- Manual hyperparameter tuning:

    - Hidden layer configurations
    
    - Activation functions
    
    - Optimisers (Adam, SGD, L-BFGS)
    
    - Learning rate initialisation
    
    - Batch size
    
    - L2 regularisation strength (alpha)
    
    - Early stopping and epoch-wise performance tracking
    
    - Stratified vs non-stratified cross-validation hypothesis testing
    
    - Unsupervised dimensionality reduction using Locally Linear Embedding (LLE)

## Repository Structure
- `src/` – Core MLP implementation and tuning logic  
- `notebooks/` – Experiments and visualisations  
- `data/` – Dataset instructions (dataset not included)  
- `results/` – Generated figures and outputs  

## Dataset

This project uses the EuroSAT RGB dataset.

The dataset is not included in this repository.
Download it from:
https://github.com/phelber/EuroSAT

Place the extracted EuroSAT_RGB/ folder inside the data/ directory before running the code.
