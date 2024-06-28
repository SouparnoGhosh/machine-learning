import numpy
from abc import ABC, abstractmethod

class Activation(ABC):

    @abstractmethod    
    def fn(v: numpy.ndarray) -> numpy.array:
        pass
    
    @abstractmethod
    def deriv_fn(y: numpy.ndarray) -> numpy.ndarray:
        pass

class Sigmoid(Activation):
    #* Sigmoid Activation Function
    
    def fn(v: numpy.ndarray) -> numpy.ndarray:
        #* Sigmoid function
        return 1. / ( 1. + numpy.exp(-v) )
    
    def deriv_fn(y: numpy.ndarray) -> numpy.ndarray:
        #* Derivative of Sigmoid Function
        y_temp = y.astype(float)
        return y_temp * (1 - y_temp)