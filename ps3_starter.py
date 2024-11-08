import numpy as np
import torch
from torch import nn

"""
Contributions

Ash:
    - Problems 1-4
    - Helped revise 5-8
    
Gloria:
    - Problems 5-8

Luke:
    - Problems 9-12
    - Test code

"""

######################################## BEGIN STARTER CODE ########################################

def relu(x):
    if x<0:
        return 0
    else:
        return x

def loss(y_predicted, y_observed):
    return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):

    r0_0 = x*W0[0]
    r0_1 = x*W0[1]
    r0_2 = x*W0[2]
    r0_3 = x*W0[3]
    r0 = np.array([r0_0,r0_1,r0_2,r0_3])

    h0_0 = relu(r0_0)
    h0_1 = relu(r0_1)
    h0_2 = relu(r0_2)
    h0_3 = relu(r0_3)
    h0 = np.array([h0_0,h0_1,h0_2,h0_3])

    r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2] + h0_3*W1[0,3]
    r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2] + h0_3*W1[1,3]
    r1 = np.array([r1_0,r1_1])

    h1_0 = relu(r1_0)
    h1_1 = relu(r1_1)
    h1 = np.array([h1_0,h1_1])

    y_predicted = h1_0*W2[0] + h1_1*W2[1]

    variable_dict = {}
    variable_dict['x'] = x
    variable_dict['r0'] = r0
    variable_dict['h0'] = h0
    variable_dict['r1'] = r1
    variable_dict['h1'] = h1
    variable_dict['y_predicted'] = y_predicted

    return variable_dict

### Example arguments ###
# x = 10
# W0 = np.array([1,2,3,4])
# W1 = np.array([[3,4,5,6],[-5,4,3,-2]])
# W2 = np.array([1,-3])

######################################## END STARTER CODE ########################################

# NOTICE: DO NOT EDIT FUNCTION SIGNATURES
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES

#Problem 1
def d_loss_d_ypredicted(variable_dict,y_observed):
    # Calculates partial derivative of loss with respect to y_predicted
    y_predicted = variable_dict['y_predicted']
    return 2 * (y_predicted - y_observed)

#Problem 2
def d_loss_d_W2(variable_dict,y_observed):
    dL = d_loss_d_ypredicted(variable_dict, y_observed)

    h1 = variable_dict['h1']

    # Calculate partial derivatives for each weight in W2
    d_W2 = np.array([dL * h1[0], dL * h1[1]])
    return d_W2

#Problem 3
def d_loss_d_h1(variable_dict,W2,y_observed):
    dL = d_loss_d_ypredicted(variable_dict, y_observed)
    
    d_h1 = np.array([dL * W2[0], dL * W2[1]])
    return d_h1


#Problem 4
def relu_derivative(x):
    if x > 0:
        return 1 # Derivative of x with respect to x
    else:
        return 0 # Derivative of 0

#Problem 5
def d_loss_d_r1(variable_dict,W2,y_observed):
    dL = d_loss_d_h1(variable_dict, W2, y_observed)
    r1 = variable_dict['r1']
    d_r1 = np.array([dL[0]*relu_derivative(r1[0]), dL[1]*relu_derivative(r1[1])])
    return d_r1

#Problem 6
def d_loss_d_W1(variable_dict,W2,y_observed):
    dL = d_loss_d_r1(variable_dict,W2,y_observed)
    h0 = variable_dict['h0']

    d_W1 = np.outer(dL, h0)
    return d_W1

#Problem 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
    """
    >>> variable_dict = {'y_predicted': 2.0, 'r1': np.array([1.0, -1.0])}
    >>> W1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    >>> W2 = np.array([1.0, 2.0])
    >>> y_observed = 1.0
    >>> d_loss_d_h0(variable_dict, W1, W2, y_observed)
    array([2., 4., 6., 8.])
    """
    dL = d_loss_d_r1(variable_dict, W2, y_observed)  #1*2
    # W1 = np.array([[3,4,5,6],[-5,4,3,-2]]) -> 2*4
    d_h0 = np.matmul(dL, W1) # we want 1*4
    return d_h0

#Problem 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
    dL = d_loss_d_h0(variable_dict, W1, W2, y_observed) #1*4
    r0 = variable_dict['r0'] #1*4
    return np.array([dL[0]*relu_derivative(r0[0]), 
                     dL[1]*relu_derivative(r0[1]), 
                     dL[2]*relu_derivative(r0[2]), 
                     dL[3]*relu_derivative(r0[3])])

#Problem 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
    d_loss_d_r0_val = d_loss_d_r0(variable_dict, W1, W2, y_observed)
    x = variable_dict['x']
    return d_loss_d_r0_val * x

#PROBLEM 10
class TorchMLP(nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        self.W0 = nn.Linear(1, 4, bias=False)
        self.W1 = nn.Linear(4, 2, bias=False)
        self.W2 = nn.Linear(2, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PROBLEM 11
def torch_loss(y_predicted,y_observed):
    loss = (y_predicted - y_observed) ** 2
    return loss

# PROBLEM 12
def torch_compute_gradient(x,y_observed,model):
    y_pred = model(x)
    loss = torch_loss(y_pred, y_observed)
    loss.backward()
    return model

