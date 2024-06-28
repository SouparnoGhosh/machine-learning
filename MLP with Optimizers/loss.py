from abc import ABC, abstractmethod
import numpy


class Loss(ABC):
    @abstractmethod
    def fn():
        pass
    
    @abstractmethod
    def deriv_fn():
        pass


class MSE_Loss(Loss):
    #* Mean Squared Error Loss Function
    
    def fn(D: numpy.ndarray, Y: numpy.ndarray):
        #* Mean Squared Error
        return .5 * numpy.sum(numpy.power(D - Y, 2), axis = 0)
        # return (1 / D.shape[0]) * numpy.sum(0.5 * numpy.sum(numpy.power(D - Y, 2), axis = 0))
    
    def deriv_fn(D: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
        #* Derivative of Mean Squared Error
        return numpy.sum(D - Y, axis = 0)

