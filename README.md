# DS4002 - SFW North American Camera Trap Imaging Analysis

A machine learning project that trains a convolutional neural network (CNN) to identify North American animal species from camera trap images. The goal is to achieve over 90% classification accuracy using a large-scale labeled dataset.

## Repository Overview

This repository contains all files and documentation for DS 4002 Project 3. The project focuses on building an image classification model that can automatically detect and classify animals captured in camera trap images across multiple U.S. locations.

We use a large dataset containing approximately 3.7 million images across 28 species and apply deep learning techniques using PyTorch. The project includes exploratory data analysis (EDA), preprocessing, model training, and evaluation. Due to the size of the dataset and computational requirements, portions of the project were completed using the University of Virginia High Performance Computing Cluster.

## 1. Software and Platform

### Software Used
- Python  
- Jupyter Notebook  
- PyTorch  
- Torchvision  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

### Required Python Packages
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- torch  
- torchvision  
- tqdm  
- pillow  

### Platform

This project was developed and tested on:

- Windows  
- Linux (UVA HPC system)

### Research Computing Resources

- UVA Rivanna High Performance Computing Cluster  
- Globus File Transfer System  
- SLURM Job Scheduler  

## 2. Documentation Map (Project Structure)

### Folder Descriptions

- **data/**: Contains metadata, annotations, and dataset references  
- **output/**: Stores plots, evaluation metrics, and model results  
- **scripts/**: Contains Python scripts for EDA, preprocessing, training, and evaluation  

### Repository File Tree

```text
DS4002-SFW/
│── README.md
│── LICENSE.md
│
├── data/
│   ├── Data Appendix.pdf
│   └── data_collection_process.txt
│
├── output/
│   ├── Animal Distribution.png
│   ├── Animals by Location.png
│   └── Location Distribution.png
│
├── scripts/
│   ├── data_prep.ipynb
│   ├── EDA.ipynb
│   ├── ready_cnn.ipynb
│   ├── large_cnn.ipynb
│   ├── job.sh
│   └── .ipynb_checkpoints/
│       ├── data_prep-checkpoint.ipynb
│       └── EDA-checkpoint.ipynb
3. Dataset Description
Source: North American Camera Trap dataset (LILA BC)
Size: Approximately 3.7 million images
Classes: 28 animal species
Special Feature: Approximately 12% empty images used as controls
Key Variables
id: Unique image identifier
filename: Image file path
location: Camera trap location
category_id: Species label ID
common_name: Animal species name
genus: Genus classification
family: Family classification
order: Order classification
class: Taxonomic class
width: Image width in pixels
height: Image height in pixels
4. Scripts Overview
data_prep.ipynb

Used for preprocessing images and metadata before model training.

Main tasks:

Loads metadata CSV file
Matches image filenames to species labels
Resizes images to 224 x 224
Converts images into arrays
Normalizes pixel values
Prepares training data inputs
EDA.ipynb

Used for exploratory data analysis and visualization generation.

Outputs include:

Animal Distribution.png
Animals by Location.png
Location Distribution.png
ready_cnn.ipynb

Primary model training notebook.

Uses:

PyTorch
Transfer learning with ResNet50
Image augmentation
80/20 train-test split
Accuracy and loss tracking over epochs
large_cnn.ipynb

Expanded CNN version used for larger-scale training experiments.

job.sh

SLURM job script used to run training on UVA HPC.

Resources requested:

1 GPU
8 CPU cores
48 hour runtime
5. Instructions for Reproducing Results
Step 1: Obtain Dataset

The raw image dataset is too large for GitHub. Follow the instructions in:

data/data_collection_process.txt

Download the North American Camera Trap dataset and place files inside the data/ folder.

Expected structure:

data/
│── nacti_metadata.csv
│── part0/
│── part1/
│── part2/
│── part3/
Step 2: Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision pillow tqdm
Step 3: Run Data Preparation

Open and run:

scripts/data_prep.ipynb

This notebook loads metadata, prepares image arrays, and creates labels for model training.

Step 4: Run Exploratory Data Analysis

Open and run:

scripts/EDA.ipynb

This notebook generates charts saved to the output/ folder.

Step 5: Train CNN Locally

Open and run:

scripts/ready_cnn.ipynb

This trains the CNN model and reports accuracy and loss metrics.

Step 6: Train CNN on UVA HPC

Submit the SLURM batch job:

sbatch scripts/job.sh
Step 7: Review Results

Training metrics are written to:

output_metrics.txt

Outputs include:

Training Accuracy
Testing Accuracy
Training Loss
Testing Loss
6. Use of UVA HPC and Globus

Because the dataset contains millions of high-resolution wildlife images and requires GPU acceleration, training was performed using the University of Virginia High Performance Computing Cluster.

Tools used:

Rivanna HPC for GPU model training
SLURM for job scheduling
Globus for secure large-file transfers between systems

This allowed efficient storage, faster training, and handling of terabyte-scale image data.

7. Model Goal

The final objective of this project is to train a convolutional neural network capable of classifying North American animal species from camera trap images with greater than 90% accuracy.

8. Notes on Reproducibility
Random seeds were fixed during train-test split where applicable.
File paths may need to be updated depending on user operating system.
Large image folders must be stored locally or on HPC storage.
Required Python packages must be installed before running notebooks.
9. References

[1] D. Morris, "North American Camera Trap Images," LILA BC. [Online]. Available: https://lila.science/datasets/nacti

[2] PyTorch Documentation. [Online]. Available: https://docs.pytorch.org

[3] University of Virginia, Rivanna High Performance Computing Cluster.