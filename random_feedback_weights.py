from sklearn.datasets import fetch_mldata
import torch
import numpy as np
import pickle
import os
import copy
from matplotlib import pyplot as plt


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
training_set_size, D_in, H, D_out = 40000, 28 * 28, 20, 10
batch_size = 200
testset_size = 2000
randomWeights_path = "randomWeights"

# preparing data
mnist = fetch_mldata('MNIST original', data_home='./')

learning_rate = 0.07
epochs = 50

# create a list of random indices for training set
train_idx = np.random.choice(len(mnist.data), training_set_size, replace=False)
# create x and y by picking samples from the random indices
mnist_x = np.array([mnist.data[i] for i in train_idx])
mnist_x = torch.ByteTensor(mnist_x).type(dtype)
mnist_y = np.array([[mnist.target[i] for i in train_idx]]).transpose()
mnist_y = torch.DoubleTensor(mnist_y).type(torch.LongTensor)

# One hot encoding
y_onehot = torch.zeros([training_set_size, D_out]).type(dtype)
y_onehot.scatter_(1, mnist_y, 1.0)
mnist_x /= 255  # scaling down x to fall between 0 and 1
x = torch.cat((mnist_x, torch.ones([training_set_size, 1])), 1)  # adding biases

x_batches = torch.split(x, batch_size)
y_batches = torch.split(mnist_y, batch_size)
y_onehot_batches = torch.split(y_onehot, batch_size)
losses = [[], []]

# Setup plotting
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.set_facecolor('black')
f.set_dpi(200)



def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def dSigmoid(h):
    return np.multiply(h, 1 - h)


def initialize_weights(fromFile=False):
    """
    This method generates randomized weights, and save them so they can be used later.
    Save is used when we want to use the same random initializations weights between experiments.
    :param fromFile: If True, weights are initialized from file.
    :return: weight tensors
    """
    if fromFile == True and os.path.isfile(randomWeights_path):
        w1, w2, w2_feedback = pickle.load(open(randomWeights_path, "rb"))

    else:
        # "+ 1"s are the bias terms
        w1 = torch.randn(D_in + 1, H).type(dtype) / np.sqrt(D_in + 1)
        w2 = torch.randn(H + 1, D_out).type(dtype) / np.sqrt(H + 1)
        w2_feedback = torch.randn(H + 1, D_out).type(dtype) / np.sqrt(H + 1)
        pickle.dump((w1, w2, w2_feedback), open(randomWeights_path, "wb"))
    return w1, w2, w2_feedback


def calc_forward(input, w1, w2):
    """This function performs forward pass given `input`, `w1`, `w2`.
    It return class predictions and hidden layer values needed for back prop."""
    hidden = sigmoid(input.mm(w1))
    hidden_biased = torch.cat((hidden, torch.ones([batch_size, 1])), 1)  # adding biases
    predictions = sigmoid(hidden_biased.mm(w2))
    return predictions, hidden_biased


def calc_backward(w2, hidden, prediction, target):
    """This function takes in w2, layer activations, and target.
     It returns gradients for weights"""
    delta_prediction = torch.mul((prediction - target), dSigmoid(prediction))
    gradient_w2 = hidden.t().mm(delta_prediction)
    dError_dHidden = delta_prediction.mm(w2[:-1, :].t())
    delta_hidden = torch.mul(dError_dHidden, dSigmoid(hidden[:, :-1]))
    gradient_w1 = x_batches[b].t().mm(delta_hidden)
    # Update weights using gradient descent
    return gradient_w1, gradient_w2

# Randomly initialize weights
w1, w2, w2_random = initialize_weights(fromFile=True)
w1_experiment = w1.clone()
w2_experiment = w2.clone()

for t in range(epochs):
    num_of_batches = int(training_set_size / batch_size)
    for b in range(num_of_batches):
        # Forward pass: compute predicted y
        y_pred, h_biased = calc_forward(x_batches[b], w1, w2)
        if b == 0:
            # Compute and print loss
            loss = (y_pred - y_onehot_batches[b]).pow(2).sum()
            losses[0].append(loss)
            print(t, loss)
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_w1, grad_w2 = calc_backward(w2, h_biased, y_pred, y_onehot_batches[b])
        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        #doing forward and backward pass for the second network
        # Forward pass: compute predicted y
        y_pred, h_biased = calc_forward(x_batches[b], w1_experiment, w2_experiment)
        if b == 0:
            # Compute and print loss
            loss = (y_pred - y_onehot_batches[b]).pow(2).sum()
            losses[1].append(loss)
            print(t, loss)
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_w1, grad_w2 = calc_backward(w2_random, h_biased, y_pred, y_onehot_batches[b])
        # Update weights using gradient descent
        w1_experiment -= learning_rate * grad_w1
        w2_experiment -= learning_rate * grad_w2

normal_loss, = ax1.plot(losses[0], color='green', label='normal')
random_loss, = ax1.plot(losses[1], color='orange', label='random w2')
ax1.legend(handles=[normal_loss, random_loss])
plt.show()

# Starting test
# removing samples that are present in training set
data_training_removed = np.delete(mnist.data, train_idx, 0)
target_training_removed = np.delete(mnist.target, train_idx, 0)
# picking test data
test_idx = np.random.choice(len(data_training_removed), testset_size)
test_x = np.array([data_training_removed[i] for i in test_idx])
test_x = torch.ByteTensor(test_x).type(dtype)
test_y = np.array([[target_training_removed[i] for i in test_idx]]).transpose()
test_y = torch.DoubleTensor(test_y).type(torch.LongTensor)
test_x = torch.cat((test_x, torch.ones([testset_size, 1])), 1)  # adding biases
# One hot encoding
test_y_onehot = torch.zeros([testset_size, D_out]).type(dtype)
test_y_onehot.scatter_(1, test_y, 1.0)

# Forward pass: compute predicted y
h_logits = test_x.mm(w1)
h = sigmoid(h_logits)
h_biased = torch.cat((h, torch.ones([testset_size, 1])), 1)  # adding biases
y_logits = h_biased.mm(w2)
y_pred = sigmoid(y_logits)

# Compute and print loss for testset
_, predicted_classes = torch.max(y_pred, dim=1)
accuracy = torch.sum(torch.eq(predicted_classes, test_y[:, 0])) / testset_size
print('test accuracy: ', accuracy)


