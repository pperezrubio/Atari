import time

import numpy as np

import theano as thn
import theano.tensor as tn
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from sandbox import PerceptronLayer


class ConvLayer():

    def __init__(self, data, filter_shape, poolsize=(2, 2), w=None, b=None):

        self.input = data
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))

        if w is None:
            w_values = np.sqrt(6/(fan_in + fan_out)) * np.random.randn(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3])
            w = thn.shared(value=w_values, name='w', borrow=True)

        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=thn.config.floatX)
            b = thn.shared(value=b_values, borrow=True)

        self.w, self.b = w, b

        conv_out = conv.conv2d(input=data, filters=self.w, filter_shape=filter_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        self.output = tn.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.w, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, filename='data/cnnMnist2.npz', nkerns=[20, 50]):

    dataset = np.load(filename)

    train_data  = tn.cast(thn.shared(dataset['train_data'][0:5000].reshape(5000, 1, 28, 28), borrow=True), 'float32')
    train_label = tn.cast(thn.shared(dataset['train_label'][0:5000], borrow=True), 'int32')
    valid_data  = tn.cast(thn.shared(dataset['valid_data'][0:200].reshape(200, 1, 28, 28), borrow=True), 'float32')
    valid_label = tn.cast(thn.shared(dataset['valid_label'][0:200], borrow=True), 'int32')
    test_data   = tn.cast(thn.shared(dataset['test_data'][0:100].reshape(100, 1, 28, 28), borrow=True), 'float32')
    test_label  = tn.cast(thn.shared(dataset['test_label'][0:100], borrow=True), 'int32')

    x = tn.ftensor4'x')   # the data is presented as rasterized images
    y = tn.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x
    #Note: image_shape[1] == filter_shape[1]
    layer0 = ConvLayer(data=layer0_input, filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
    layer1 = ConvLayer(data=layer0.output, filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))


    layer2_input = layer1.output.flatten(2)
    layer2 = PerceptronLayer(data=layer2_input, no_inputs=nkerns[1] * 4 * 4, no_outputs=500, activation=tn.tanh)
    layer3 = PerceptronLayer(data=layer2.output, no_inputs=500, no_outputs=10, activation=tn.nnet.softmax)

    cost = -tn.mean(tn.log(layer3.output)[tn.arange(y.shape[0]), y])
    errors = tn.mean(tn.neq(layer3.pred, y))

    # create a function to compute the mistakes that are made by the model
    test_error = thn.function([], errors, givens={x: test_data, y: test_label})
    valid_error = thn.function([], errors, givens={x: valid_data, y: valid_label})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = tn.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = thn.function([], cost, updates=updates, givens={x: train_data, y: train_label})

    print 'Training the model...'

    start_time = time.clock()

    for epoch in xrange(n_epochs):

        train_model()
        print 'Epoch %i, Validation error: %f %%' % (epoch, valid_error())

    print 'Test error: %f' % (test_error())

    end_time = time.clock()
    print 'The code ran for %.1fs' % (end_time - start_time)


if __name__ == '__main__':

    evaluate_lenet5()
