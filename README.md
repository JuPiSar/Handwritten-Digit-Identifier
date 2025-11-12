# MNIST Digit Classification with Keras

This project demonstrates a simple neural network implemented with Keras and TensorFlow to classify handwritten digits from the MNIST dataset.

## Project Structure

This notebook covers the following steps:
1.  **Environment Setup**: Importing necessary libraries.
2.  **Data Loading and Preprocessing**: Loading the MNIST dataset, reshaping, and initial filtering (commented out in the notebook).
3.  **Model Definition**: Creating a sequential Keras model with dense layers.
4.  **Model Training**: Compiling and training the model on a subset of the MNIST training data.
5.  **Prediction and Visualization**: Making predictions on individual samples and visualizing predictions on a batch of random samples.

## Setup and Prerequisites

To run this notebook, you need to have the following libraries installed:
-   TensorFlow
-   Keras (usually installed with TensorFlow)
-   NumPy
-   Matplotlib

YouYou can install them using pip:
```bash
pip install tensorflow numpy matplotlib
```

## Data Loading and Preprocessing

The MNIST dataset is loaded using `mnist.load_data()`. The images are then reshaped from 2D (28x28) to 1D (784) vectors and a subset of 3000 training samples is used for faster experimentation.

## Model Architecture

The model is a simple feed-forward neural network defined using `tf.keras.Sequential` with the following layers:
-   Input Layer: `tf.keras.Input(shape=(784,))`
-   Hidden Layer 1: `Dense(units=100, activation='relu', name="layer1")`
-   Hidden Layer 2: `Dense(units=30, activation='relu', name="layer2")`
-   Output Layer: `Dense(units=10, activation='softmax', name="layer3")` (for 10 digit classes)

## Model Training

The model is compiled with:
-   **Loss Function**: `SparseCategoricalCrossentropy(from_logits=True)`
-   **Optimizer**: `Adam(learning_rate=0.001)`
-   **Metrics**: `accuracy`

The model is trained for 20 epochs with a batch size of 32 and a 20% validation split.

## Prediction and Visualization

After training, the notebook demonstrates how to make predictions on individual samples and visualizes the model's predictions on 8 random images from the training set, displaying the predicted label (`P`) and the true label (`T`).
