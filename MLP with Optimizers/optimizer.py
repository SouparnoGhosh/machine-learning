from abc import ABC, abstractmethod
import numpy


class Optimizer(ABC):      
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod 
    def initialize_params(self, params):
        pass
    
    @abstractmethod
    def update(self, parameters):
        pass

#* Normal Gradient Descent
class SGD(Optimizer):
    def __init__(self, lr=0.5):
        super().__init__("Normal Gradient Descent")
        self.lr = lr
        
    def initialize_params(self, params):
        pass

    def update(self, parameters):
        old_wts = parameters["old_wts"]
        del_wts = parameters["del_wts"]
        old_bias = parameters["old_bias"]
        del_bias = parameters["del_bias"]

        return (old_wts + self.lr * del_wts, old_bias + self.lr * del_bias)

#* Momentum Based Gradient Descent
class Momentum(Optimizer):
    def __init__(self, lr=0.5, alpha=0.5):
        super().__init__("Momentum Gradient Descent")
        self.lr = lr
        self.alpha = alpha
        
    def initialize_params(self, params):
        self.prev_delta_w = [numpy.zeros_like(i) for i in params['wts']]
        self.prev_delta_b = [numpy.zeros_like(i) for i in params['bias']]

    def update(self, parameters):
        old_wts = parameters["old_wts"]
        del_wts = parameters["del_wts"]
        old_bias = parameters["old_bias"]
        del_bias = parameters["del_bias"]
        k = parameters["k"] # Which layer

        new_delta_wts = self.alpha * self.prev_delta_w[k] + self.lr * del_wts
        new_delta_bias = self.alpha * self.prev_delta_b[k] + self.lr * del_bias

        new_wts = old_wts + new_delta_wts
        new_bias = old_bias + new_delta_bias 

        self.prev_delta_w[k] = new_delta_wts
        self.prev_delta_b[k] = new_delta_bias

        return new_wts, new_bias

#* Root Mean Square Propagation
class RMSProp(Optimizer):

    def __init__(self, lr=0.5, alpha=0.99, epsilon=1):
        super().__init__("RMSProp")

        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.wts_a = None
        self.bias_a = None
        
    def initialize_params(self, params):
        
        self.wts_a = [numpy.zeros_like(i) for i in params['wts']]
        self.bias_a = [numpy.zeros_like(i) for i in params['bias']]

    def update(self, parameters):
        old_wts = parameters["old_wts"]
        old_bias = parameters["old_bias"]
        del_wts = parameters["del_wts"]
        del_bias = parameters["del_bias"]
        k = parameters['k']

        self.wts_a[k] = self.alpha * self.wts_a[k] + (1 - self.alpha) * (del_wts**2)
        self.bias_a[k] = self.alpha * self.bias_a[k] + (1 - self.alpha) * (del_bias**2)

        new_wts = old_wts + self.lr * (del_wts / numpy.sqrt(self.wts_a[k] + self.epsilon))
        new_bias = old_bias + self.lr * (del_bias / numpy.sqrt(self.bias_a[k] + self.epsilon))

        return (new_wts, new_bias)

#* Adaptive Gradient Descent
class AdaGrad(Optimizer):
    
    def __init__(self, lr = 0.5, epsilon = 0.001) -> None:
        super().__init__("AdaGrad")
        
        self.lr = lr
        self.epsilon = epsilon
        
        self.wts_a = None
        self.bias_a = None
    
    def initialize_params(self, params):
        self.wts_a = [numpy.zeros_like(i) for i in params['wts']]
        self.bias_a = [numpy.zeros_like(i) for i in params['bias']]

    
    def update(self, parameters):
        old_wts = parameters["old_wts"]
        old_bias = parameters["old_bias"]
        del_wts = parameters["del_wts"]
        del_bias = parameters["del_bias"]
        k = parameters['k']

        self.wts_a[k] += (del_wts**2)
        self.bias_a[k] += (del_bias**2)

        new_wts = old_wts + self.lr * (del_wts / numpy.sqrt(self.wts_a[k] + self.epsilon))
        new_bias = old_bias + self.lr * (del_bias / numpy.sqrt(self.bias_a[k] + self.epsilon))

        return (new_wts, new_bias)

#! Do from here
#* Adaptive Momentum Estimation
class Adam(Optimizer):

    def __init__(self, lr=0.5, alpha=0.99, beta=0.5, epsilon=1) -> None:
        super().__init__("Adam")
        
        self.lr = lr
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon

        self.wts_a = None
        self.bias_a = None
        
        self.wts_m = None
        self.bias_m = None

    def initialize_params(self, params):
        
        self.wts_a = [numpy.zeros_like(i) for i in params['wts']]
        self.bias_a = [numpy.zeros_like(i) for i in params['bias']]
        
        self.wts_m = [numpy.zeros_like(i) for i in params['wts']]
        self.bias_m = [numpy.zeros_like(i) for i in params['bias']]
        
        self.prev_delta_w = [numpy.zeros_like(i) for i in params['wts']]
        self.prev_delta_b = [numpy.zeros_like(i) for i in params['bias']]
        
    def update(self, parameters):
        old_wts = parameters["old_wts"]
        old_bias = parameters["old_bias"]
        del_wts = parameters["del_wts"]
        del_bias = parameters["del_bias"]
        i = parameters["i"]
        k = parameters["k"]
    
        self.wts_m[k] = self.alpha * self.prev_delta_w[k] + (1 -  self.alpha) * del_wts
        self.bias_m[k] = self.alpha * self.prev_delta_b[k] + (1 -  self.alpha) * del_bias

        self.wts_a[k] = self.beta * self.wts_a[k] + (1 - self.beta) * (del_wts**2)
        self.bias_a[k] = self.beta * self.bias_a[k] + (1 - self.beta) * (del_bias**2)

        wts_hat_a = self.wts_a[k] / (1 - numpy.power(self.beta, i) + self.epsilon)
        bias_hat_a = self.bias_a[k] / (1 - numpy.power(self.beta, i) + self.epsilon)

        wts_hat_m = self.wts_m[k] / (1 - numpy.power(self.alpha, i) + self.epsilon)
        bias_hat_m = self.bias_m[k] / (1 - numpy.power(self.alpha, i) + self.epsilon)        

        delta_w = self.lr * wts_hat_m / numpy.sqrt(wts_hat_a + self.epsilon)
        delta_b = self.lr * bias_hat_m / numpy.sqrt(bias_hat_a + self.epsilon)
        
        self.prev_delta_w[k] = delta_w
        self.prev_delta_b[k] = delta_b
        
        new_wts = old_wts + delta_w
        new_bias = old_bias + delta_b

        return (new_wts, new_bias)
