__author__ = 'harri'
__project__ = 'dds'

import theano
import theano.tensor as T
import numpy as np


class Layer(object):

    def __init__(self, name="layer"):
        self.name = name

    def get_output(self, input_var):
        #Given a symbolic input variable, returns a symbolic output variable.
        pass

    def get_params(self):
        #Returns a list of parameters which are theano shared variables.
        pass


class AffineLayer(Layer):

    def __init__(self, input_dim, output_dim, init_W=None, init_b = None, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_W = init_W
        self.init_b = init_b
        super(AffineLayer, self).__init__(**kwargs)
        self.initialise_params()

    def initialise_params(self):
        if self.init_W is not None:
            W = self.init_W(self.input_dim, self.output_dim)

        else:
            W = np.random.uniform(0,0.01, (self.output_dim,self.input_dim))

        self.W = theano.shared(np.asarray(W,dtype = theano.config.floatX), self.name+"_W")
        if self.init_b is not None:
            b = self.init_b(self.output_dim)
        else:
            b = np.zeros((self.output_dim,))
        self.b = theano.shared(np.asarray(b,dtype = theano.config.floatX), self.name+"_b")

    def get_output(self, input_var):
        return T.dot(input_var, self.W) + self.b

    def get_params(self):
        return [self.W, self.b]



class DenseLayer(AffineLayer):

    def __init__(self,nonlinearity, *args, **kwargs):
        self.nonlinearity = nonlinearity
        super(DenseLayer, self).__init__(*args, **kwargs)

    def get_output(self, input_var):
        pre_activation = super(DenseLayer, input_var).get_output(input_var)
        if self.nonlinearity is None:
            return pre_activation
        else:
            return self.nonlinearity(pre_activation)


class NeuralNetwork(Layer):

    def __init__(self,layers, **kwargs):
        self.layers = layers
        super(NeuralNetwork, self).__init__(**kwargs)

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return list(set(params))

    def get_output(self, input_var):
        #Chains outputs from layer to layer.
        output_var = input_var
        for layer in self.layers:
            output_var = layer.get_output(output_var)
        return output_var








def test():
    L = Layer(name="test_layer")
    L = AffineLayer(10,10, name="test_layer")
    L.W.set_value(np.eye(10, dtype=theano.config.floatX))
    x = theano.tensor.matrix("x")
    output = L.get_output(x)
    print output.eval({x:np.ones((2,10), dtype=theano.config.floatX)})
    N = NeuralNetwork([L]*10)
    print N.get_params()
    output = N.get_output(x)
    print output.eval({x:np.ones((2,10), dtype=theano.config.floatX)})

if __name__ == "__main__":
    test()
