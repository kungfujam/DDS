__author__ = 'harri'
__project__ = 'dds'

import theano
import lasagne
import helper
import numpy as np
import cPickle as cp
import matplotlib.pyplot as plt

#Load/prepare the data
data = helper.load_data()
train_X, train_y = data["train"]
train_y = np.reshape(train_y, (-1,1))
N,d = train_X.shape
train_X = theano.shared(lasagne.utils.floatX(train_X), "train_X")
train_y = theano.shared(lasagne.utils.floatX(train_y), "train_y")
val_X, val_y = data["validation"]
val_y = np.reshape(val_y, (-1,1))
val_X = theano.shared(lasagne.utils.floatX(val_X), "val_X")
val_y = theano.shared(lasagne.utils.floatX(val_y), "val_y")


def get_errors(penalty=0):


    #Build network.
    input_layer = lasagne.layers.InputLayer((None,d), name="input_layer")
    output_layer = lasagne.layers.DenseLayer(incoming=input_layer,num_units=1, name="output_layer", nonlinearity=None)

    #Build cost and symbolic variables.
    X = input_layer.input_var
    y = theano.tensor.matrix("y")
    prediction = lasagne.layers.get_output(output_layer)
    mse = theano.tensor.mean((y-prediction)**2)

    L1_penalty = penalty*lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l1)
    cost = mse+L1_penalty
    lower_index, upper_index = theano.tensor.lscalar("lower_index"), theano.tensor.lscalar("upper_index")

    #Get updates
    learning_rate = 0.01
    updates = lasagne.updates.sgd(loss_or_grads=cost, params = lasagne.layers.get_all_params(output_layer),
                                  learning_rate=learning_rate)

    #Compile functions.
    train_foo = theano.function(inputs=[lower_index,upper_index],updates=updates,
                                givens={X:train_X[lower_index:upper_index,:],
                                        y:train_y[lower_index:upper_index,:]})

    get_cost = theano.function(inputs=[X,y], outputs=mse)

    #Training time.

    n_epochs = 1000
    mini_batch_size = 1000
    costs = []
    for epoch in range(n_epochs):
        current_cost = get_cost(val_X.get_value(borrow=True), val_y.get_value(borrow=True))
        costs.append(current_cost)
        print "Epoch: %s, Validation MSE: %s" % (epoch, current_cost)
        for index in range(0,N,mini_batch_size ):
            train_foo(index,index+mini_batch_size)

    return costs
    #Save results.
    save = False
    if save:
        W = output_layer.W.get_value()
        plot_path  = "/home/harri/Dropbox/Work/CDT/DDS/weights.png"
        helper.plot_image(W, save_path=plot_path)
        params = {"W":W, "b":output_layer.b.get_value()}
        pickle_path = "/home/harri/Dropbox/Work/CDT/DDS/params.pkl"
        with open(pickle_path, "wb") as param_file:
            cp.dump(params, param_file)
    else:
        W = output_layer.W.get_value()
        helper.plot_image(W)


penalties = [0.1,0.01,0.001, 0.0001,0]

costs = [get_errors(penalty) for penalty in penalties]
fig, ax = plt.subplots(1,1)
for penalty,cost in zip(penalties,costs):
    ax.plot(cost[5:], label="L1 penalty: %s" % penalty)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
plt.legend()
plt.show()



