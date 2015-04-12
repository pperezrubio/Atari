
import time
import numpy as np
import theano as thn
import theano.tensor as tn


class PerceptronLayer(object):

    def __init__(self, data, no_inputs, no_outputs, activation=None, w=None, b=None):

        self.input = data

        if w is None and activation is tn.tanh or activation is tn.nnet.sigmoid: 
            w_values = np.sqrt(6.0/(no_inputs + no_outputs)) * np.random.randn(no_inputs, no_outputs)
            w = thn.shared(value=w_values, name='w', borrow=True)
        elif w is None and activation is not tn.tanh or activation is not tn.nnet.sigmoid:
            w_values = 0.01 * np.random.randn(no_inputs, no_outputs)
            w = thn.shared(value=w_values, name='w', borrow=True)

        if b is None:
            b_values = np.zeros((no_outputs,))
            b = thn.shared(value=b_values, name='b', borrow=True)

        self.w, self.b = w, b

        if activation is None:
            self.output = tn.dot(self.input, self.w) + self.b
        else:
            self.output = activation(tn.dot(self.input, self.w) + self.b)

        self.pred = tn.argmax(self.output, axis=1) #Change: Take the 
        self.params = [self.w, self.b]
        self.activation_type = activation
        self.no_outputs = no_outputs
        self.no_inputs = no_inputs


class Mlp():

    def __init__(self, layers):

        self.layers = layers
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


class Qnn():

    def __init__(self, layers):

        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params = self.params + layer.params

        self.old_layers = []
        for i in xrange(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                layer = self.layers[i]
                self.old_layers.insert(0, PerceptronLayer(data=layer.input, no_inputs=layer.no_inputs, no_outputs=layer.no_outputs, activation=layer.activation_type))
            else:
                layer = self.layers[i]
                self.old_layers.insert(0, PerceptronLayer(data=self.old_layers[0].output, no_inputs=layer.no_inputs, no_outputs=layer.no_outputs, activation=layer.activation_type))

        self.output = self.layers[0].output
        self.pred = self.layers[0].pred


    def train(self, s, s_prime, action, reward, gamma, term, hyperparams):

        #Define symbolic variables
        y = tn.matrix('y')
        a, r, t, g = tn.bscalar('a'), tn.bscalar('r'), tn.bscalar('t'), tn.bscalar('g') 

        print 'Training the model...'

        #gparams = []
        #for param in self.params:
        #    gparams.append(tn.grad(cost, param))

        #updates = [
        #    (param, param - hyperparams['learn_rate'] * gparam)
        #    for param, gparam in zip(self.params, gparams)
        #]

        #Predict qs_prime with prev weights.
        #theta = cpy(self.layers)
        #self.layers = self.layers_old
        f_prime = thn.function([], outputs=self.pred, givens={x: state_prime})
        qs_prime = f_prime()
        print ps_prime
        #self.layers = theta

        #Set prev weights to the current ones.
        #self.layers_old = cpy(self.layers)

        #Change current weights according to update equation
        #dE = np.zeros(qs.shape)
        #cost = self.output[0, a] - r - t * g * tn.max(y)
        #train_model = thn.function([], outputs=cost, updates=updates, givens={x: state, y: qs_prime, r: reward, t: term, g: gamma})


    def predict(self, s):
        return thn.function([], outputs=self.pred, givens={x: s})



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
