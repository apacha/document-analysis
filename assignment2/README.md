# Optical Character Recognition for Document Analysis

This folder contains the source code for the second exercise of the Document Analysis course at the TU Wien, Summer 2018.

# Running the experiments
Make sure you've got all requirements installed, that are specified in [requirements.txt](requirements.txt).

## Running the full pipeline
Run `python main.py` to perform data-preparation, training and evaluation.

## Predicition
> Notice, that `predict.py` assumes the correct dataset to have been prepared previously. To do so, you can run `main.py` and interrupt the process as soon as the actual training starts and then proceed with the prediction script.

Download the trained model from [here](https://owncloud.tuwien.ac.at/index.php/s/v7S2QddBS6cPamw) and put it into the same folder as `predict.py`, then run `python predict.py`

# Other stuff
In case you want to run the Keras sample for image_ocr, you have to 

- Install cairo, via `conda install cairo`
- Install cairo-bindings via `pip install cairocffi`