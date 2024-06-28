# #### Libraries
import numpy  # for numerical operations


# #### Class of SLP. 2 classes. So 1 perceptron is enough.
class SinglePerceptron:
    def __init__(self, m: int, eta: int):
        # Initializing Hyperparameters
        self.m = m  # No. of input neurons
        self.eta = eta  # Learning Rate
        # Initializing weights. 1 bias + Weight array for each input neuron
        self.wts = numpy.zeros(1 + self.m)
        self.accuracy = [] # List to store accuracy after each epoch
        self.error = [] # List to store error after each epoch

    # #### Activation Functions
    def sigmoid_activation(self, v: numpy.number) -> numpy.number:
        return 1 / (1 + numpy.exp(-v))

    def unipolar_activation(self, v: numpy.number) -> numpy.number:
        return 1 if v >= 0 else 0

    def bipolar_activation(self, v: numpy.number) -> numpy.number:
        return 1 if v >= 0 else -1

    def tanh_activation(self, v: numpy.number) -> numpy.number:
        return numpy.tanh(v)
    # #### End of Activation Functions

    def predict(self, X: numpy.array) -> numpy.array:
        X_bias = numpy.insert(X, 0, 1)  # Prepending bias
        v = self.wts.T.dot(X_bias)  # W^T * X. Linear Combiner

        y = self.unipolar_activation(v)  # Activation Function

        return y  # array of single value

    # X is input and D is desired output
    # ndarray is n-dimensional array
    def train(self, X: numpy.ndarray, D: numpy.ndarray, EPOCHS: int):
        for _ in range(EPOCHS):
            correct = 0;

            for x, d in zip(X, D):
                x_bias = numpy.insert(x, 0, 1)  # Prepend bias
                y = self.predict(x)  # Predict with existing weights (bias is being added in predict function, so no need to add bias to x here)
                error = d - y  # Error = desired - actual
                correct += 1 if d == y else 0
                delta_w = self.eta * error * x_bias  # Calculating change in weights
                self.wts = self.wts + delta_w  # Updating wts

            self.accuracy.append(correct * 100 / X.shape[0])
            self.error.append( 100 - (correct * 100 / X.shape[0]) )