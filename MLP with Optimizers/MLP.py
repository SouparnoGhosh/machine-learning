import numpy
from typing import List
from tqdm import trange

from optimizer import Optimizer
from activation import Activation
from loss import Loss

LEARNING_RATE = 0.1


class MLP:
    #* MLP with multiple hidden layers
    
    def __init__(self, 
                 input_m_neurons: int, 
                 hidden_k_neurons: List[int], 
                 output_n_neurons: int, 
                 loss_fn: Loss,
                 optimizer_fn: Optimizer, 
                 activation_fn: Activation,
                ):
        
        self.m = input_m_neurons
        self.k = hidden_k_neurons
        self.n = output_n_neurons
        
        self.layers = [input_m_neurons] + hidden_k_neurons + [output_n_neurons]
        
        self.wts: List[numpy.ndarray] = []
        self.bias: List[numpy.ndarray] = []
        
        self.loss_fn: Loss = loss_fn
        self.optimizer_fn: Optimizer = optimizer_fn
        self.activation_fn: Activation = activation_fn
        
        # For the first hidden layer to the output layer, initialize weights and bias
        for i in range(1, len(self.layers)):
            
            wt = numpy.random.random((self.layers[i - 1], self.layers[i]))
            bias = numpy.random.random((1, self.layers[i]))
            
            self.wts.append(wt)
            self.bias.append(bias)

        # Passes the initial weights and bias to the optimizer function 
        self.optimizer_fn.initialize_params({'wts': self.wts, 'bias': self.bias})
            
    def train(self, X: numpy.ndarray, D: numpy.ndarray, test_X: numpy.ndarray = None, test_D: numpy.ndarray = None, epochs: int = 5, batch_size: int = 8):
        #* Training the dataset
        
        metrics = {
            "loss": [], 
            "accuracy": [],
        }
        
        
        print(f"Using {self.optimizer_fn.name} optimizer for {epochs} epochs:")
        epoch_range = trange(epochs)
        for _ in epoch_range:
            for i in range(0, X.shape[0], batch_size):
                
                x = X[i:i+batch_size]
                d = D[i:i+batch_size]
                
                # Feed forward
                output = self.predict(x)
                deltas = []
                
                # Backpropogation
                
                output_error = self.loss_fn.deriv_fn(d, output[-1])
                output_delta = output_error * self.activation_fn.deriv_fn(output[-1])
                
                # Stor the output layer delta
                deltas = [output_delta]

                # Store the hidden layer deltas
                for j in range(len(self.layers) - 2, 0, -1):
                                       
                    hidden_error = deltas[-1].dot(self.wts[j].T)
                    hidden_delta = hidden_error * self.activation_fn.deriv_fn(output[j])
                                            
                    deltas.append(hidden_delta)
                    
                deltas.reverse()
                                        
                # Update weights and bias using the optimizer function  
                for k in range(len(self.wts)):

                    delta_w = numpy.dot(output[k].T, deltas[k])
                    delta_b = numpy.sum(delta_w, axis = 0)
                    
                    parameters = {
                        'old_wts': self.wts[k],
                        'del_wts': delta_w,
                        'old_bias': self.bias[k],
                        'del_bias' : delta_b,
                        'i': i, # Which row of input data
                        'k': k  # Which layer
                    }
                    
                    self.wts[k], self.bias[k] = self.optimizer_fn.update(parameters)
            
            # Testing the model       
            *_, y_pred = self.predict(test_X)
            metrics['loss'].append((1 / test_X.shape[0]) * numpy.sum(self.loss_fn.fn(test_D, y_pred)))
            metrics['accuracy'].append(MLP.accuracy(test_D, y_pred))            
       
        return metrics
        
    def accuracy(D: numpy.ndarray, Y: numpy.ndarray):
                
        D_encoded = (D > 0.5).astype(int)
        Y_encoded = (Y > 0.5).astype(int)
        
        return numpy.sum(D_encoded == Y_encoded) / D.shape[0]        
            
    def predict(self, X: numpy.ndarray):
        #* Weight and Bias Prediction

        x_temp = X
        # Will store the outputs of each layer
        outputs = [x_temp]
        
        for i in range(len(self.wts)):
            v = x_temp.dot(self.wts[i]) + self.bias[i]
            y = self.activation_fn.fn(v)
            
            x_temp = y
            
            outputs.append(y)
        
        return outputs