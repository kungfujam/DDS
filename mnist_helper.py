import cPickle
import gzip
import os
import matplotlib.pyplot as plt

def load_data(path=None):
    ''' Loads the dataset
    '''
    folder="MNIST_data"

    #############
    # LOAD DATA #
    #############
    if path is None:


        # Download the MNIST dataset if it is not present
        dataset=os.path.join(folder, "mnist.pkl.gz")

        if os.path.isfile(dataset):
            pass

        else:
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)
    else:
        dataset=path


    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    return {"train":train_set, "validation":valid_set, "test":test_set}

def plot_image(x, save_path = None, width=28, height=28):
    #Plots a single greyscale image vector.
    plt.imshow(x.reshape(width, height), cmap = plt.cm.Greys_r)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def test():
    #This is a test, and this comment is also a test of the slack integration.
    data = load_data()
    train = data["train"]
    train_X = train[0]
    plot_image(train_X[0,:])
    


