# Two-Layer Neural Network from Scratch

This repository implements a two-layer neural network from scratch using NumPy. It includes basic functionalities such as forward propagation, backpropagation, and training on the Fashion MNIST dataset. The model is capable of classifying 10 different classes and uses a single hidden layer with 64 neurons.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project aims to demonstrate how a simple neural network can be built from scratch using only NumPy. The neural network is trained on the Fashion MNIST dataset to classify images of clothing into 10 categories. The project includes both a one-layer and two-layer neural network implementation, with key functions such as Xavier initialization, forward propagation, backpropagation, and model evaluation.

### Features:
- **One-layer Neural Network:** Simple model with a single layer and no hidden units.
- **Two-layer Neural Network:** Model with one hidden layer containing 64 neurons and an output layer with 10 neurons.
- **Xavier Initialization:** Used to initialize weights.
- **Activation Function:** Sigmoid function is used for non-linearity.
- **Loss Function:** Mean Squared Error (MSE).
- **Backpropagation:** For adjusting weights and biases based on gradient descent.

## Dataset
We use the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which contains 70,000 grayscale images of 10 categories of clothing. The dataset is split into 60,000 images for training and 10,000 images for testing.

**Classes:**
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Requirements
The code requires Python 3.x and the following libraries:
```bash
numpy
pandas
matplotlib
tqdm
requests
```

You can install the dependencies by running:
```bash
pip install numpy pandas matplotlib tqdm requests
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/StepanetsAnton/neural_network.git
   cd neural_network
   ```
2. **Download the Fashion MNIST dataset**: The script will automatically download the dataset if it is not available locally.

3. **Run the training script**: You can run the training with the following command:
   ```bash
   python train.py
   ```
4. **Visualize results**: The model logs the loss and accuracy for each epoch and generates a plot of the training process. You can inspect the plot.png file for a graphical representation of the loss and accuracy over 20 epochs.

## Model Architecture
The two-layer neural network is structured as follows:
- **Input Layer**: 784 neurons (28x28 pixel images).
- **Hidden Layer**: 64 neurons, using Xavier initialization and sigmoid activation.
- **Output Layer**: 10 neurons, corresponding to the 10 fashion classes, using sigmoid activation.

## Training
The model is trained for 20 epochs with a batch size of 100 and a learning rate (`alpha`) of 0.5. Loss is calculated using the Mean Squared Error (MSE) and backpropagation is used to update weights and biases.

The key functions in the code are:
- **Forward Propagation**: Implements the forward pass through the network layers.
- **Backpropagation**: Adjusts the weights and biases using gradient descent.
- **Training Loop**: Trains the model over multiple epochs, logging loss and accuracy.

## Results
The model logs the accuracy after each epoch. For example, you can expect output similar to:
```bash
[0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91]
```
You can also visualize the training process with the generated plot.

## Acknowledgments
This project is built as a learning exercise to understand the fundamentals of neural networks. Special thanks to the creators of the Fashion MNIST dataset and NumPy for making this project possible.
