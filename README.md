# FeatureSelect-Benchmark-Hub

## Overview
This repository contains all necessary materials for conducting and reproducing the feature selection (FS) experiments described in our study, "Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking." It includes a collection of 102 datasets, a robust set of FS algorithms, and scripts for running these algorithms under different configurations.
<!-- 
### Current Availability
- **Dataset Repository:** From the Standard Difficulty Data Collections, we are currently sharing datasets from the `scikit-feature` repository. For Challenging Data Collections, we are providing access to Unaltered Data Sources.
- **Algorithms:** Out of 24 algorithms, 10 are currently shared in this repository. The full suite will be released progressively as the project evolves and after the completion of the submission process for our associated research.
-->
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
To run the scripts in this repository, you will need Python 3.11 and the packages listed in `requirements.txt`.

## Getting Started

These instructions will guide you through setting up your local environment to run the feature selection experiments.

### Prerequisites

Before you begin, ensure you have Python installed on your machine. You can download Python [here](https://www.python.org/downloads/). This project requires Python 3.11.6

### Setup

1. **Clone the repository:**
   To get started with the FeatureSelect-Benchmark-Hub, you will first need to clone the repository to your local machine. You can do this by opening a terminal and running the following command:

   ```bash
   git clone https://github.com/yourusername/FeatureSelect-Benchmark-Hub.git  
   Replace `yourusername` with your GitHub username or the username of the repository owner. This command downloads all the files from the GitHub repository to your local system in a folder named `FeatureSelect-Benchmark-Hub`.

2. **Navigate to the repository directory:**
Change your directory to the repository you just cloned:

3. **Install dependencies:**
This project depends on several Python libraries. Install them using pip with the provided requirements file: `requirements.txt`

### Running the Experiments

Once you have set up your environment, you can run the main script to begin the experiments: main.py

This command starts the feature selection processes as configured in your project. Make sure to check the `config.py` for any settings you might need to adjust according to your specific needs.

### Additional Resources

- **Google Colab:** If you prefer to use cloud resources, check the Google Colab notebook provided in the repository. It allows you to run the experiments without any local setup, using Google's servers.

For any issues during the setup or execution of the experiments, please check the GitHub issues page or create a new issue to get help.

### Arduino Specific Instructions

If you have any Arduino specific steps, configurations, or setups, you can outline those here to help users integrate or test with Arduino platforms (this section assumes relevance based on the mention of "arduino").

### Troubleshooting

For troubleshooting the setup or execution, please consult the GitHub issues page or open a new issue to get assistance.


## Contributing
We welcome contributions to this project. Please fork the repository and submit a pull request with your improvements.


## Contact
For questions or feedback, please file an issue through the GitHub issue tracker.


