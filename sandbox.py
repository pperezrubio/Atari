import os
import sys
import time

import numpy as np

import theano as thn
import theano.tensor as tn


class PerceptronLayer(object):

    def __init__(self, data, no_inputs, no_outputs, activation=None, w=None, b=None):

        if w is None and activation is tn.tanh or activation is tn.nnet.sigmoid: 
            w_values = np.sqrt(6/(no_inputs + no_outputs)) * np.random.randn(no_inputs, no_outputs)
            w = thn.shared(value=w_values, name='w', borrow=True)
        elif w is None and activation is not tn.tanh or activation is not tn.nnet.sigmoid:
            w_values = 0.01 * np.random.randn(no_inputs, no_outputs)
            w = thn.shared(value=w_values, name='w', borrow=True)

        if b is None:
            b_values = np.zeros((no_outputs,))
            b = thn.shared(value=b_values, name='b', borrow=True)

        self.w, self.b = w, b

        self.output = activation(tn.dot(data, self.w) + self.b)
        self.pred = tn.argmax(self.output, axis=1)
        self.params = [self.w, self.b]


class Mlp(object):

    def __init__(self, layers):

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.layers = layers

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        for layer in self.layers:
            self.params = self.params + layer.params

        self.output = self.layers[0].output
        self.pred = self.layers[0].pred


    def train(self, train_data, valid_data, train_label, valid_label, params):

        cost = -tn.mean(tn.log(self.output)[tn.arange(y.shape[0]), y])
        errors = tn.mean(tn.neq(self.pred, y))

        gparams = []
        for param in self.params:
            gparams.append(tn.grad(cost, param))

        updates = [
            (param, param - params['learn_rate'] * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        train_model = thn.function([], outputs=cost, updates=updates, givens={x: train_data, y: train_label})
        valid_error = thn.function([], outputs=errors, givens={x: valid_data, y: valid_label})

        print 'Training the model...'

        start_time = time.clock()
        no_epochs = params['epochs']

        for epoch in xrange(no_epochs):

            train_model()
            print 'Epoch %i, Validation error: %f %%' % (epoch, valid_error())

        end_time = time.clock()
        print 'The code ran for %.1fs' % (end_time - start_time)



def testmnist(filename='data/mlpMnist2.npz', params={'epochs':114, 'learn_rate':0.05, 'no_hidden':600}):

    dataset = np.load(filename)

    train_data  = tn.cast(thn.shared(dataset['train_data'], borrow=True), 'float64')
    train_label = tn.cast(thn.shared(dataset['train_label'], borrow=True), 'int32')
    valid_data  = tn.cast(thn.shared(dataset['valid_data'], borrow=True), 'float64')
    valid_label = tn.cast(thn.shared(dataset['valid_label'], borrow=True), 'int32')
    test_data   = tn.cast(thn.shared(dataset['test_data'], borrow=True), 'float64')
    test_label  = tn.cast(thn.shared(dataset['test_label'], borrow=True), 'int32')

    global x, y

    x = tn.matrix('x')
    y = tn.ivector('y')

    print '... building the model'

    layer1 = PerceptronLayer(data=x, no_inputs=28*28, no_outputs=600, activation=tn.tanh)
    layer2 = PerceptronLayer(data=layer1.output, no_inputs=600, no_outputs=10, activation=tn.nnet.softmax)

    mlp = Mlp([layer2, layer1])

    mlp.train(train_data, valid_data, train_label, valid_label, params)

    test_error = thn.function([], outputs=tn.mean(tn.neq(mlp.pred, y)), givens={x: test_data, y: test_label})
    print 'Test error: %f' % (test_error())


if __name__ == '__main__':

    testmnist()
