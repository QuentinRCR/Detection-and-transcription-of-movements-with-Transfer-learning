# Detection and transcription of movements with Transfer learning

## About this repository
This repository hosts the code developed as part of a research project aimed at enhancing the identification of letters written in the air with a new handwriting style. The system utilizes an IMU sensor attached to the fingertip to track the letter movements.

It is composed of 3 main parts:
- **The performance evaluation folder** contains a code to evaluate the RAM usage of the inference of one letter.  It outputs in the terminal a memory report.
- **The preprocessing folder** contains 3 codes that pre-processes the data from different datasets to output in a consistent formatting in `Preprocessed_datasets`.
- **The main.ipynb notebook** contains the core of the code with the models, the training, the evaluation and the exploration of the data.

## How to use it
1. Download the different datasets of interest. The EMNIST dataset is mandatory. The [DigiLeTs dataset](https://github.com/CognitiveModeling/DigiLeTs) can be clone, the BRUSH dataset can be downloaded from the README of [this repository](https://github.com/brownvc/decoupled-style-descriptors) and the EMNIST can be downloaded [from Kaggle](https://www.kaggle.com/datasets/crawford/emnist).  
2. Go in `paths` and specify the different relative paths to the root folder of the different datasets
3. Run the scripts located in the preprocessing folder and that corresponds to the datasets you are interested in. Specify the name of the output folder at the top of the file. The EMNIST dataset doesn't require any preprocessing.
4. Explore the main notebook by specifying the parameters you desires in the top cell

## Content of the main file
The main file was run with 2 different versions of tensorflow depending on the model. FOr the LSTM it uses Tensorflow 2.10.0. For the CNN, it uses tensorflow-directml 1.15.8.

The version of tensorflow has to be specified at the very top of the file, before the imports.

If the appropriate options are defined at the beginning of the file, all cells can be executed sequentially without requiring any alterations.

The main file contains:
- Cells where the different dataset are loaded
- Functions to prepare the datasets
- A function splitting the datasets according to options
- A cell to train the dataset in a traditional manner (without fine tuning)
- A cells to train datasets with the source dataset and save the model and then cells to fine-tune with specific data
- A cell to view the training process of any previously trained model
- A cell giving the mean performance of any try matching a regex expression
- A part to load a previously ran model and explore it results:
    - Show the correctly classified values
    - Show the misclassified values
    - Show the confusion matrix of the model
    - Show the confusion matrix of the sum of all the tries matching the regex expression
- A cell to examine the examples of a specific dataset
- Some experimental work to fuse the results of 2 models