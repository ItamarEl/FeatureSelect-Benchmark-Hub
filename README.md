# FeatureSelect-Benchmark-Hub

## Overview
This repository contains all necessary materials for conducting and reproducing the feature selection (FS) experiments described in our study, "Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking." It includes a collection of 102 datasets, a robust set of FS algorithms, and scripts for running these algorithms under different configurations.

## Repository Structure
- **Dataset Repository/**:
  - **Standard Difficulty Data Collections/**: Datasets categorized as Easy/Medium, organized by their source repository.
  - **Challenging Data Collections/**: Hard datasets, including Peel-Processed datasets and Unaltered Data Sources.
    - `D_Train_source_Pre_Peeling.mat`: Training dataset pre-peeling.
    - `D_Test.mat`: Corresponding test dataset.
    - `D_Train_Peeling.mat`: Training dataset post-peeling, presenting more challenging scenarios.

- **Code/**:
  - `Main.py`: Main script to drive FS experiments.
  - `Config.py`: Configuration settings for experiments.
  - `Utilities.py`: Supporting functions for the experiments.
  - `Algorithms/`: Contains individual scripts for each of the 24 FS algorithms.
  - `Algorithms_Hyperparameters_FS.xlsx`: Spreadsheet with editable hyperparameters for each algorithm.

- **requirements.txt**: Contains all necessary Python packages.

## Prerequisites
To run the scripts in this repository, you will need Python 3.x and the packages listed in `requirements.txt`.

## Setup
1. Clone this repository:
