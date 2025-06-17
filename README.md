# MNIST Digit Classifier

A robust digit recognition system using a Convolutional Neural Network (CNN) trained on MNIST.

## Features
- CNN model for recognizing handwritten digits
- Data augmentation during training
- Predict digits from any image (not just 28x28)

## Setup

pip install -r requirements.txt


## Training the Model

python main.py


## Predicting Your Own Digit
Place your image (e.g., screenshot of a number) in `test_inputs/` folder:

python utils/predict_digit.py



> Image will be auto-preprocessed and resized to 28x28 before prediction.
