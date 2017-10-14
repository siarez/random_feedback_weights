from sklearn.datasets import fetch_mldata
import torch
import numpy as np
from scipy.stats import ttest_ind
import pickle
import os
import math
from matplotlib import pyplot as plt
import csv
import time

dtype = torch.FloatTensor
# loads data or downloads if necessary
mnist = fetch_mldata('MNIST original', data_home='./')


class Configs:
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    training_set_size, D_in, H, D_out = 50000, 28 * 28, 20, 10
    batch_size = 200
    testset_size = 2000
    learning_rate = 0.07
    epochs = 200
    randomWeights_path = "randomWeights"

    @classmethod
    def to_string(cls):
        keypair = [(a,  cls.__dict__[a]) for a in dir(cls) if not a.startswith('__') and not callable(getattr(cls, a))]
        return str(keypair)

    def __str__(self):
        return self.to_string()


def create_dataset(dataset, set_size, one_hot_size, indices_to_exclude=None):
    """
    Takes in the MNIST dataset and picks a subset.
    x is normalized to be in range 0 and 1
    y (target) is one-hot encoded
    :param dataset: MNIST dataset
    :param set_size: number of samples in the dataset that is returned
    :param one_hot_size: size of the one-hot target vectors
    :param indices_to_exclude: When sampling from MNIST this indices are excluded
    :return:
    """
    if (indices_to_exclude is not None):
        _data = np.delete(dataset.data, indices_to_exclude, 0)
        _target = np.delete(dataset.target, indices_to_exclude, 0)
    else:
        _data = dataset.data
        _target = dataset.target
    # create a list of random indices to pick the training set from
    indices = np.random.choice(len(_data), set_size, replace=False)
    # create x and y by picking samples from the random indices
    mnist_x = np.array([_data[i] for i in indices])
    mnist_x = torch.ByteTensor(mnist_x).type(dtype)
    mnist_y = np.array([[_target[i] for i in indices]]).transpose()
    mnist_y = torch.DoubleTensor(mnist_y).type(torch.LongTensor)

    # One hot encoding
    _y_onehot = torch.zeros([set_size, one_hot_size]).type(dtype)
    _y_onehot.scatter_(1, mnist_y, 1.0)
    mnist_x /= 255  # scaling down x to fall between 0 and 1
    _x = torch.cat((mnist_x, torch.ones([set_size, 1])), 1)  # adding biases
    return _x, _y_onehot, indices


def create_batches(x, y, _batch_size):
    _x_batches = torch.split(x, _batch_size)
    _y_batches = torch.split(y, _batch_size)
    return _x_batches, _y_batches


def sigmoid(_logits):
    return 1.0 / (1.0 + np.exp(-_logits))


def dSigmoid(_h):
    return np.multiply(_h, 1 - _h)


def initialize_weights(input_size, hidden_size, output_size, from_file=False):
    """
    This method generates randomized weights, and save them so they can be used later.
    Save is used when we want to use the same random initializations weights between experiments.
    :param input_size: number of input pixels.
    :param hidden_size: number of hidden units.
    :param output_size: number of output classes.
    :param from_file: If True, weights are initialized from file.
    :return: weight tensors
    """
    if from_file is True and os.path.isfile(Configs.randomWeights_path):
        _w1, _w2, _w2_feedback = pickle.load(open(Configs.randomWeights_path, "rb"))
    else:
        # "+ 1"s are the bias terms
        _w1 = torch.randn(input_size + 1, hidden_size).type(dtype) / np.sqrt(input_size + 1)
        _w2 = torch.randn(hidden_size + 1, output_size).type(dtype) / np.sqrt(hidden_size + 1)
        _w2_feedback = torch.randn(hidden_size + 1, output_size).type(dtype) / np.sqrt(hidden_size + 1)
        pickle.dump((_w1, _w2, _w2_feedback), open(Configs.randomWeights_path, "wb"))
    return _w1, _w2, _w2_feedback


def calc_forward(_input, _w1, _w2):
    """This function performs forward pass given `input`, `w1`, `w2`.
    It return class predictions and hidden layer values needed for back prop."""
    hidden = sigmoid(_input.mm(_w1))
    hidden_biased = torch.cat((hidden, torch.ones([_input.shape[0], 1])), 1)  # adding biases
    predictions = sigmoid(hidden_biased.mm(_w2))
    return predictions, hidden_biased


def calc_backward(_w2, _input, hidden, prediction, target):
    """This function takes in w2, layer activations, and target.
     It returns gradients for weights"""
    delta_prediction = torch.mul((prediction - target), dSigmoid(prediction))
    gradient_w2 = hidden.t().mm(delta_prediction)
    dError_dHidden = delta_prediction.mm(_w2[:-1, :].t())
    delta_hidden = torch.mul(dError_dHidden, dSigmoid(hidden[:, :-1]))
    gradient_w1 = _input.t().mm(delta_hidden)
    # Update weights using gradient descent
    return gradient_w1, gradient_w2


accuracies = []  # holds accuracies for the two network for the test dataset for each experiment run
result_folder_path = os.path.join("./", time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)
with open(os.path.join(result_folder_path, "config.txt"), 'w') as file:
    file.write(str(Configs()))

for expr in range(30):
    # Setup plotting
    f, (ax1) = plt.subplots(1, 1, sharey=True)
    ax1.set_facecolor('black')
    f.set_dpi(200)
    losses = [[], []]  # holds the losses for the two networks for each epoch in the experiment
    x_training, y_training, indices_training = create_dataset(mnist, Configs.training_set_size, Configs.D_out)
    x_training_batches, y_training_batches = create_batches(x_training, y_training, Configs.batch_size)
    x_test, y_test, _ = create_dataset(mnist, Configs.testset_size, Configs.D_out, indices_training)
    # Randomly initialize weights
    w1, w2, w2_random = initialize_weights(Configs.D_in, Configs.H, Configs.D_out, from_file=False)
    # Experiment network (the one with random backprop w2) starts off with the same weights as the normal network
    w1_experiment = w1.clone()
    w2_experiment = w2.clone()

    for t in range(Configs.epochs):
        num_of_batches = int(Configs.training_set_size / Configs.batch_size)
        for b in range(num_of_batches):

            # doing forward and backward pass for the second network
            # Forward pass: compute predicted y
            y_pred_expr, h_biased_expr = calc_forward(x_training_batches[b], w1_experiment, w2_experiment)
            if b == 0:
                # Compute and print loss
                loss = (y_pred_expr - y_training_batches[b]).pow(2).sum()
                losses[1].append(loss)
                print(t, loss)
            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_w1_expr, grad_w2_expr = calc_backward(w2_random * math.log(t+1), x_training_batches[b], h_biased_expr, y_pred_expr, y_training_batches[b])
            # Update weights using gradient descent
            w1_experiment -= Configs.learning_rate * grad_w1_expr
            w2_experiment -= Configs.learning_rate * grad_w2_expr

            # Forward pass: compute predicted y
            y_pred, h_biased = calc_forward(x_training_batches[b], w1, w2)
            if b == 0:
                # Compute and print loss
                loss = (y_pred - y_training_batches[b]).pow(2).sum()
                losses[0].append(loss)
                print(t, loss)
            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_w1, grad_w2 = calc_backward(w2, x_training_batches[b], h_biased, y_pred, y_training_batches[b])
            # Update weights using gradient descent
            w1 -= Configs.learning_rate * grad_w1
            w2 -= Configs.learning_rate * grad_w2

        # w2_random *= 1.02


    normal_loss, = ax1.plot(losses[0], color='green', label='normal',  linewidth=1)
    random_loss, = ax1.plot(losses[1], color='orange', label='random w2',  linewidth=1)
    ax1.legend(handles=[normal_loss, random_loss])
    ax1.set_ylim([0, 250])
    plt.savefig(os.path.join(result_folder_path,  "expr-" + str(expr+1) + "-loss.png"))

    # Starting test
    # Forward pass: compute predicted y
    y_pred, _ = calc_forward(x_test, w1, w2)
    y_pred_experiment, _ = calc_forward(x_test, w1_experiment, w2_experiment)


    # Compute and print loss for testset
    _, predicted_classes = torch.max(y_pred, dim=1)
    _, predicted_classes_experiment = torch.max(y_pred_experiment, dim=1)
    _, target_class = torch.max(y_test, dim=1)
    accuracy = torch.sum(torch.eq(predicted_classes, target_class)) / Configs.testset_size
    accuracy_experiment = torch.sum(torch.eq(predicted_classes_experiment, target_class)) / Configs.testset_size
    print('test accuracies: ', accuracy, accuracy_experiment)
    accuracies.append((accuracy, accuracy_experiment))

# calculating t-statistics to judge if the difference is significant
accuracies_np = np.array(accuracies)
t_statistic, p_val = ttest_ind(accuracies_np[:, 0], accuracies_np[:, 1], equal_var=False)
print("t stat and p val", t_statistic, p_val)

with open(os.path.join(result_folder_path, "accuracies.csv"), 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["normal", "random w2"])
    for row in accuracies:
        csvwriter.writerow([row[0], row[1]])
    csvwriter.writerow(["averages"])
    csvwriter.writerow([accuracies_np[:, 0].mean(), accuracies_np[:, 1].mean()])
    csvwriter.writerow(["t-statistic", "p value"])
    csvwriter.writerow([t_statistic, p_val])

