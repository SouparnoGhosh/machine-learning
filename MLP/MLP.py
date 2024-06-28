import numpy as np
from typing import List
from tqdm import trange


LEARNING_RATE = 0.1


# * General MLP with n hidden layers
class MLP:

    def __init__(self, input_m: int, hidden_k: List[int], output_n: int):
        # m - no of neurons in input layer
        self.m = input_m
        # k - list of no of neurons in hidden layers
        self.k = hidden_k
        # n - no of neurons in output layer
        self.n = output_n

        # layers stores the no of neurons in each layer
        self.layers = [input_m] + hidden_k + [output_n]
        # big_print("Layers", self.layers)

        # Initialize the weights and biases
        self.wts: List[np.ndarray] = []
        self.bias: List[np.ndarray] = []

        for i in range(1, len(self.layers)):
            # Fill them with random values
            #! How is randomness being reproduced?
            wt = np.random.random((self.layers[i - 1], self.layers[i]))
            bias = np.random.random((1, self.layers[i]))

            self.wts.append(wt)
            self.bias.append(bias)

    def train(self, X: np.ndarray, D: np.ndarray, val_X: np.ndarray, val_D: np.ndarray, epochs: int):
        # * Training the dataset
        # Storing the accuracy of each epoch
        accuracy_list = []
        error_list = []

        # Creates a range of values from o to epochs-1
        epoch_range = trange(epochs)
        # No of data samples processed in each training step. This is called online learning.
        batch_size = 1

        # We train epoch number of times
        for _ in epoch_range:
            # We train the model (0 to no of rows-1) in batches of batch_size
            for i in range(0, X.shape[0], batch_size):

                # Get the batch of features and targets
                x = X[i:i+batch_size]
                d = D[i:i+batch_size]

                # * Feed forward
                # We get the input layer and the output at each layer
                output = self.predict(x)

                # * Back Propogation
                # y = output[-1]
                output_error = self.gradient_loss_fn(d, output[-1])
                output_delta = output_error * self.sigmoid_deriv_fn(output[-1])

                deltas = [output_delta]

                # We iterate from the last hidden layer to the first hidden layer
                # We are propagating the error backwards at each layer.
                # This error is used to update the weights at each layer.
                for j in range(len(self.layers) - 2, 0, -1):

                    hidden_error = deltas[-1].dot(self.wts[j].T)
                    hidden_delta = hidden_error * \
                        self.sigmoid_deriv_fn(output[j])

                    deltas.append(hidden_delta)

                # Reverse the deltas list to match the order of layers
                deltas.reverse()

                for i in range(len(self.wts)):
                    # Calculating the updates
                    del_w = LEARNING_RATE * np.dot(output[i].T, deltas[i])
                    # np.sum(del_w, axis=0) is the sum of all the rows of del_w
                    del_b = np.sum(del_w, axis=0)

                    # Updating the weights and bias
                    self.wts[i] = self.wts[i] + del_w
                    self.bias[i] = self.bias[i] + del_b

            # * Testing the dataset
            # Taking the last row of the outputs - final output y (predicted)
            *_, y_pred = self.predict(val_X)
            accuracy_list.append(self.accuracy(val_D, y_pred))
            error_list.append(self.mean_sq_err_fn(val_D, y_pred))

        return accuracy_list, error_list

    def accuracy(self, D: np.ndarray, Y: np.ndarray):
        Y = (Y >= 0.5).astype(int).flatten()
        return np.sum(D == Y) / D.shape[0]

    def predict(self, X: np.ndarray):
        # We are making a copy of input layer because we don't want to change the original input layer
        x_temp = X
        # Stores the output of each layer. We add the input layer to it.
        outputs = [x_temp]

        # Iterate through each layer
        for i in range(len(self.wts)):
            # V = W^T.X + B
            # In this case x_temp is X^T.
            # V will be in transpose form. All calculations are done in transpose form.
            # It is essentially V^T = X^T.W + B^T
            v = x_temp.dot(self.wts[i]) + self.bias[i]
            # big_print(f"{i} - V", v)
            # Y = f(V)
            y = self.sigmoid_fn(v)

            # The output of this layer becomes the input of the next layer
            x_temp = y

            # Add the output of each layer to the outputs list
            outputs.append(y)

        # Return the output of each layer
        return outputs

    # * activation functions and their derivatives
    # Sigmoid function

    def sigmoid_fn(self, v: np.ndarray) -> np.ndarray: return 1 / \
        (1 + np.exp(-v))

    def sigmoid_deriv_fn(self, y: np.ndarray) -> np.ndarray: return y * (1 - y)

    # ReLU function (Rectified Linear Unit) (Gives worse results)
    def relu_fn(self, v: np.ndarray) -> np.ndarray: return np.maximum(0, v)

    def relu_deriv_fn(
        self, y: np.ndarray) -> np.ndarray: return np.where(y <= 0, 0, 1)

    # Tanh function (Gives worse results)
    def tanh_fn(self, v: np.ndarray) -> np.ndarray: return np.tanh(v)

    def tanh_deriv_fn(
        self, y: np.ndarray) -> np.ndarray: return 1 - np.power(y, 2)

    # * Mean square error and its derivative
    def mean_sq_err_fn(self, D: np.ndarray, Y: np.ndarray): return (
        0.5 * np.sum(np.power(D - Y, 2))) ** 0.5

    # Gradient of loss in Y w.r.t. D
    # Returns the sum of differences for each column
    def gradient_loss_fn(
        self, D: np.ndarray, Y: np.ndarray) -> np.ndarray: return np.sum(D - Y, axis=0)
