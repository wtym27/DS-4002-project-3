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
- Linux (UVA HPC system)

### Research Computing Resources

- UVA Rivanna High Performance Computing Cluster  
- Globus File Transfer System  
- SLURM Job Scheduler  

## 2. Documentation Map (Project Structure)

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
│   ├── Location Distribution.png
│   └── output_metrics.txt
│
├── scripts/
│   ├── EDA.ipynb
│   ├── ready_cnn.py
│   ├── large_cnn.ipynb
│   ├── job.sh
```
**Folder descriptions:**
- `data/`: Contains data files, including image folders and metadata
- `output/`: Output files (exploratory analysis plots and CNN performance metrics)
- `scripts/`: Contains the Python scripts used to perform EDA, CNN, and slurm job.

## 3. Instructions for Reproducing Results
Here is an outline for how we produced our results: 

### Step 1: Clone the Repository
1. Open a terminal or command prompt in an HPC with AT LEAST 2 terabytes of storage available. 
2. Clone the repository: `git clone https://github.com/wtym27/DS-4002-project-3.git`
3. Navigate into the project directory by doing `cd ds4002-project-3`

### Step 2: Set up virtual environment
1. Ensure that the latest version of Python is installed and create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment:`source venv/bin/activate`
3. Install the required packages as mentioned above using `pip install [PACKAGE NAMES HERE]`

### Step 3: Prepare the dataset
1. Download all the files at https://lila.science/datasets/nacti in the data/ directory using `wget` and `unzip`. 
2. See "data/data_collection_process.md" for specific details on downloading all of the data. 

### Step 4: Run EDA
1. Run the EDA script by pressing the triangle on the top left that appears when you open`src/EDA.ipynb`.
2. This script generates exploratory plots in the output/ folder. 

### Step 5: Run CNN
1. Move ready_cnn.py and job.sh into the DS-4002-project-3/ directory by doing `mv [FILE_NAME_HERE] ../`. This is necessary because for some reason they cannot access the data/ folder unless it is a child folder. 
2. The CNN takes about 20 hours to run, so it is not recommended to run the large_cnn.ipynb file. Instead, schedule a slurm job that uses ready_cnn.py, which uses the same code, by doing `sbatch job.sh`. 

### Step 7: Review Results
1. Inspect the slurm output log or the output_metric.txt document to see results. 
