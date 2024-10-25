"""
Gloria:
    - Base answer for problems 9-12
    - Helped double check all other problems 

Ashley: 
    -

Luke:
    - Led the initial implementation of problems 1-4
    - Wrote graphical tests to validate 6 - 12 
    - Modified free response (10 - 12) for better clarity
"""


import numpy as np
import torch
from torch import nn
import torch.optim as optim


################################ BEGIN NUMPY STARTER CODE #################################################
def sigmoid(x):
    #Numerically stable sigmoid function.
    #Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


def sample_logistic_distribution(x,a):
    #np.random.seed(1)
    num_samples = len(x)
    y = np.empty(num_samples)
    for i in range(num_samples):
        y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
    return y

def create_input_values(dim,num_samples):
    #np.random.seed(100)
    x_inputs = []
    for i in range(num_samples):
        x = 10*np.random.rand(dim)-5
        x_inputs.append(x)
    return x_inputs


def create_dataset():
    x= create_input_values(2,100)
    a=np.array([12,12])
    y=sample_logistic_distribution(x,a)

    return x,y

################################ END NUMPY STARTER CODE ####################################################



################################ BEGIN PYTORCH STARTER CODE ################################################

class TorchLogisticClassifier(nn.Module):

  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(num_features))

  def forward(self, x_vector):
    logit = torch.dot(self.weights, x_vector)
    prob = torch.sigmoid(logit)
    return prob


def loss_fn(y_predicted, y_observed):
    return -1 * (y_observed * torch.log(y_predicted)
                 + (1 - y_observed) * torch.log(1 - y_predicted))

def extract_num_features(dataset):
    first_example = dataset[0]
    # first_example is a pair (x,y), where x is a vector of features and y is 0 or 1
    # note that both x and y are torch tensors
    first_example_x = first_example[0]
    first_example_y = first_example[1]
    num_features = first_example_x.size(0)
    return num_features

def nonbatched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(num_epochs):
        for d_x, d_y in dataset:
            optimizer.zero_grad()
            prediction = model(d_x)
            loss = loss_fn(prediction, d_y)
            loss.backward()
            optimizer.step()
    return model

def generate_nonbatched_data(num_features=3, num_examples=100):
    x_vectors = [torch.randn(num_features) for _ in range(num_examples)]
    prob_val = 0.5 * torch.ones(1)
    y_vectors = [torch.bernoulli(prob_val) for _ in range(num_examples)]

    dataset = list(zip(x_vectors, y_vectors))

    return dataset

def main():
    nonbatched_dataset = generate_nonbatched_data()
    nonbatched_gradient_descent(nonbatched_dataset)

################################ END PYTORCH STARTER CODE ###################################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


# PROBLEM 1
def logistic_positive_prob(x,a):
    z = np.dot(a, x)
    return sigmoid(z)

# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
    prob = logistic_positive_prob(x_i, a)
    return -(y_i - prob) * x_i[j]

# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    n = len(y)
    partial_derivative = 0
    for i in range(n):
        partial_derivative += logistic_derivative_per_datapoint(y[i], x[i], a, j)
    return partial_derivative / n

# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    k = len(a)
    gradient = np.zeros(k)
    for j in range(k):
        gradient[j] = logistic_partial_derivative(y, x, a, j)
    return gradient

# PROBLEM 5
def gradient_update(a,lr,gradient):
    return a - lr * gradient

# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    a = initial_a
    for _ in range(num_iterations):
        gradient = compute_logistic_gradient(a, y, x)
        a = gradient_update(a, lr, gradient)
    return a

# PROBLEM 7
# Free Response Answer Here:
'''
The function __init__ is a constructor that calls its parent class (nn.Module) 
using super() and initializes a vector of zeroes called weights with the length
based on the number of features given by the parameter (num_feature).
The function is called at line 78: model = TorchLogisticClassifier(num_features)
'''

# PROBLEM 8
# Free Response Answer Here:
'''
The forward function is a transformer that calculates the sigmoid of the dot 
product of the weights parameter and x vector input, or 
(latex) $\sigma(\vec{w} \cdot \vec{x})$.
The function is called at line 83: prediction = model(d_x)
'''

# PROBLEM 9
def batched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    batches = split_into_batches(dataset, batch_size)
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(num_epochs):         # epoch loop
        for batch in batches:           # batch loop
            # only call backward once per batch per epoch
            optimizer.zero_grad()
            for d_x, d_y in batch:      # loop inside batch
                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
                loss.backward()
            optimizer.step()
    return model

# PROBLEMS 10-12
def split_into_batches(dataset, batch_size):
    k_batches = len(dataset) / batch_size
    batches = torch.zeros_like(k_batches)
    i = 0
    for k in k_batches:
        batches[k] = dataset[i:1+batch_size]
        i += batch_size
    return batches

def alt_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    batches = split_into_batches(dataset, batch_size)
    for i in range(num_epochs):
        # optimizer.zero_grad() # 1
        for batch in batches:
            # optimizer.zero_grad() # 2
            for d_x, d_y in batch:
                # optimizer.zero_grad() # 3
                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
                loss.backward()
                # optimizer.step() # C
            # optimizer.step() # B
        # optimizer.step() # A
    return model

# PROBLEM 10
# Free Response Answer Here: 
"""
In this gradient descent structure, .backward() calculates the gradient, optimizer.step() 
updates the model parameters based on the current gradient, and optimizer.zero_grad() 
resets the gradient to prevent it from accumulating across iterations.

Uncommenting line A for optimizer.step() means parameters are updated only once per epoch, 
so they would be adjusted a total of num_epochs times. Similarly, uncommenting line 1 for 
optimizer.zero_grad() resets the gradient only once per epoch, allowing gradients to 
accumulate over all data points in the dataset before each reset.

Thus, by the end of each epoch, the total gradient used in the update is effectively a 
summation of gradients over the entire dataset D, averaged across all examples. Letting 
j denote the total number of epoch loops, the effective gradient applied to update 
parameters after each epoch can be expressed as:

Latex equation: \nabla_{\vec{w}}L(\vec{w}|B) := \frac{1}{j}\sum_{d\in D} \nabla_{\vec{w}}L(\vec{w}|d)

Equation (13) provides the batch-wise gradient update rule, but the above equation represents the full 
dataset's gradient after each epoch. Since zero_grad() is called once per epoch, the accumulated gradient 
is summed over all data points before a reset, with updates occurring only after each epoch loop.
"""

# PROBLEM 11
# Free Response Answer Here: 
"""
In this setup, optimizer.zero_grad() is never called, and optimizer.step() is called once per epoch. 
As a result, gradients are never cleared, and instead, they continue accumulating over all epochs. 
This means that the parameters are updated based on the summation of all gradients across epochs, 
without ever resetting in between.

The effective gradient applied to the parameter update after each epoch t can be expressed as:

Latex equation: \nabla_{\vec{w}}L(\vec{w}|B) := \sum_{d\in D} \nabla_{\vec{w}}L(\vec{w}|d)

In this configuration, the gradient calculation does not average over the number of epochs, 
since it continually accumulates without resetting. Thus, the gradient applied during each 
parameter update is the cumulative sum of gradients over all data points across all epochs.
This setup would likely prevent the model from converging properly and would risk diverging 
or fitting poorly to the data.
"""

# PROBLEM 12
# Free Response Answer Here: 
"""
Uncommenting lines 2B will make alt_gradient_descent equivalent to batched_gradient_descent.

In this configuration:

- optimizer.zero_grad() is called once per batch within each epoch, ensuring gradients are 
    reset between batches, as required in Problem 9.
- optimizer.step() is also called at line B within the batch loop, applying the accumulated 
    gradient for each batch update.

This configuration implements batched gradient descent by ensuring that the gradient is 
computed, applied, and then cleared for each batch within every epoch.
"""