import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def scale(X_train, X_test):
    max_value = np.max(X_train)
    X_train_rescaled = X_train / max_value
    X_test_rescaled = X_test / max_value
    return X_train_rescaled, X_test_rescaled

def xavier(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    return correct_predictions / y_true.shape[0]

class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, X):
        z = np.dot(X, self.weights) + self.biases
        self.output = sigmoid(z)
        return self.output

    def backprop(self, X, y, alpha):
        z = np.dot(X, self.weights) + self.biases
        a = sigmoid(z)

        dz = mse_derivative(y, a) * sigmoid_derivative(z)
        dw = np.dot(X.T, dz)/X.shape[0]
        db = np.sum(dz, axis=0, keepdims=True)/X.shape[0]

        self.weights -= alpha * dw
        self.biases -= alpha * db

class TwoLayerNeural:
    def __init__(self, n_features, n_classes, n_hidden=64):
        self.W1 = xavier(n_features, n_hidden)
        self.b1 = xavier(1, n_hidden)

        self.W2 = xavier(n_hidden, n_classes)
        self.b2 = xavier(1, n_classes)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)

        return self.output

    def backprop(self, X, y, alpha):
        # Step 1: Calculate the error at the output layer
        dz2 = mse_derivative(y, self.output) * sigmoid_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        # Step 2: Propagate the error to the hidden layer
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        # Step 3: Update the weights and biases
        self.W2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dw1
        self.b1 -= alpha * db1


def train_one_epoch(model, X_train, y_train, alpha):
    output = model.forward(X_train)
    model.backprop(X_train, y_train, alpha)
    loss = mse(y_train, output)
    return loss


def full_training(model, X_train, y_train, X_test, y_test, n_epochs=20, batch_size=100, alpha=0.5):
    loss_history = []
    accuracy_history = []

    initial_accuracy = accuracy(y_test, model.forward(X_test))
    accuracy_history.append(initial_accuracy)
    print(f"[{initial_accuracy:.4f}]", end=" ")


    for epoch in tqdm(range(n_epochs), desc="Training Progress"):

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss = train_one_epoch(model, X_batch, y_batch, alpha)

        loss_history.append(loss)
        acc = accuracy(y_test, model.forward(X_test))
        accuracy_history.append(acc)


    print(f"[{', '.join([f'{acc:.4f}' for acc in accuracy_history[1:]])}]")

    plot(loss_history, accuracy_history)

    return loss_history, accuracy_history

if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train_scaled, X_test_scaled = scale(X_train, X_test)

    value1 = float(round(X_train_scaled[2, 778], 2))
    value2 = float(round(X_test_scaled[0, 774], 2))

    xavier_output = xavier(2, 3).flatten().tolist()

    sigmoid_input = np.array([-1, 0, 1, 2])
    sigmoid_output = sigmoid(sigmoid_input).flatten().tolist()

    n_features = 784
    n_classes = 10

    model = TwoLayerNeural(n_features=n_features, n_classes=n_classes)

    output_before = model.forward(X_train_scaled[:2])

    model.backprop(X_train_scaled[:2], y_train[:2], alpha=0.1)

    output_after = model.forward(X_train_scaled[:2])

    mse_value = mse(y_train[:2], output_after)

    print(mse_value.flatten().tolist())
