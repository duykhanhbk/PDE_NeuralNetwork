import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import random
import numpy as np
import math
import itertools
from PDENet import PDENet

# Program to solve boundary initial parabolic equation in 2D space and 1D time, unit square region


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(3, 20)
        self.hidden1 = nn.Linear(20, 200)
        self.hidden2 = nn.Linear(200, 200)
        self.hidden3 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 1)
        torch.nn.init.xavier_uniform(self.input.weight)
        torch.nn.init.xavier_uniform(self.hidden1.weight)
        torch.nn.init.xavier_uniform(self.hidden2.weight)
        torch.nn.init.xavier_uniform(self.output.weight)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


def ut(f, input_vector):
    ut = grad(f, input_vector, create_graph=True)[0].take(torch.tensor([2]))
    return ut


def laplacian(f, input_vector):
    gradient = grad(f, input_vector, create_graph=True)[0]
    ux = gradient.take(torch.tensor([0]))
    uxx = grad(ux, input_vector, create_graph=True)[0].take(torch.tensor([0]))
    uy = gradient.take(torch.tensor([1]))
    uyy = grad(uy, input_vector, create_graph=True)[0].take(torch.tensor([1]))
    return uxx + uyy


def initial_condition(Omegainput):
    return math.sin(math.pi*Omegainput.take(torch.tensor([0])))*math.sin(math.pi*Omegainput.take(torch.tensor([1])))


def boundary_condition(Sinput):
    return 0


# f: ut - laplace(u) = f
def right_hand_side(Qinput):
    return (1 + 2*math.pi*math.pi)*math.exp(Qinput.take(torch.tensor([2])))*math.sin(math.pi*Qinput.take(torch.tensor([0])))*math.sin(math.pi*Qinput.take(torch.tensor([1])))
    # return 0

# spatial_time_point = (x, y, t)
def exact_solution(spatial_time_point):
    # return math.exp(-2*math.pi*math.pi*spatial_time_point.take(torch.tensor([2])))*math.sin(math.pi*spatial_time_point.take(torch.tensor([0])))*math.sin(math.pi*spatial_time_point.take(torch.tensor([1])))
    return math.exp(spatial_time_point.take(torch.tensor([2]))) * math.sin(math.pi * spatial_time_point.take(torch.tensor([0]))) * math.sin(math.pi * spatial_time_point.take(torch.tensor([1])))


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


# Q =  Ω × [0, T], S = ∂Ω × [0, T]
def random_data_points(batch_size):
    Qpoints = []
    Spoints = []
    Omegapoints = []
    for i in range (batch_size):
        Qpoint = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        Qpoints.append(Qpoint)
    for i in range (batch_size):
        edge1 = random.randint(0, 1)
        edge2 = random.randint(0, 1)  # when choosing a random point on the boundary, one index must be either 0 or 1, these variable determine which
        if edge1 == 0 and edge2 == 0:
            Spoint = [0, random.uniform(0, 1), random.uniform(0, 1)]
        elif edge1 == 0 and edge2 == 1:
            Spoint = [1, random.uniform(0, 1), random.uniform(0, 1)]
        elif edge1 == 1 and edge2 == 0:
            Spoint = [random.uniform(0, 1), 0, random.uniform(0, 1)]
        elif edge1 == 1 and edge2 == 1:
            Spoint = [random.uniform(0, 1), 1, random.uniform(0, 1)]
        Spoints.append(Spoint)

    for i in range (batch_size):
        Omegapoint = [random.uniform(0, 1), random.uniform(0, 1), 0]
        Omegapoints.append(Omegapoint)
    return Qpoints, Spoints, Omegapoints


def batch_loss(net, datapoints):
    Qpoints, Spoints, Omegapoints = datapoints
    G1 = G2 = G3 = 0
    for Qpoint in Qpoints:
        Qpoint_input = Variable(torch.Tensor(Qpoint).resize_(3, 1), requires_grad=True)
        Qpoint_output = net(Qpoint_input)
        G1 += (ut(Qpoint_output, Qpoint_input) - laplacian(Qpoint_output, Qpoint_input) - right_hand_side(Qpoint_input)) ** 2

    for Spoint in Spoints:
        Spoint_input = Variable(torch.Tensor(Spoint).resize_(3, 1), requires_grad=True)
        Spoint_output = net(Spoint_input)
        G2 += (Spoint_output - boundary_condition(Spoint_input))**2

    for Omegapoint in Omegapoints:
        Omegapoint_input = Variable(torch.Tensor(Omegapoint).resize_(3, 1), requires_grad=True)
        Omegapoint_output = net(Omegapoint_input)
        G3 += (Omegapoint_output - initial_condition(Omegapoint_input))**2
    G1 = G1 / len(Qpoints)
    G2 = G2 / len(Spoints)
    G3 = G3 / len(Omegapoints)
    return G1 + G2 + G3


def unit_square_time_grid(step):
    # X = np.linspace(0, 1, land_mark)
    X = [x * step for x in range(0, 1 + math.floor(1/step))]
    Y = [x * step for x in range(0, 1 + math.floor(1/step))]
    T = [x * step for x in range(0, 1 + math.floor(1/step))]
    return list(itertools.product(X, Y, T))

# init training
net = PDENet(3, 100)
net.init_weights()

# resume training
# net = torch.load('./model2')

# test resume
# sample = random_data_points(1000)
# square_error = batch_loss(net, sample)
# print(square_error)

# count_parameters(net)
learning_rate = 0.001
lr_scale = 0.9
iterations_count = 0
# training
for i in range(10000):
    print('Iteration number: ' + str(i + 1))
    sample = random_data_points(1000)  # sample space time point

    for j in range(4):
        print('Batch iteration number: ' + str(j + 1))
        net.zero_grad()
        square_error = batch_loss(net, sample)  # calculate square error loss
        square_error.backward()  # calculate gradient of square loss w.r.t the parameters
        print('Batch loss: ' + str(square_error))
        for param in net.parameters():
            param.data -= learning_rate*param.grad.data

    L2erroronQ = 0
    for spatial_time_point in unit_square_time_grid(0.1):
        spatial_time_point = torch.Tensor(spatial_time_point).resize_(3,1)
        L2erroronQ += (net(spatial_time_point) - exact_solution(spatial_time_point))**2
    L2erroronQ = L2erroronQ/1331
    print('L2error = ' + str(L2erroronQ))
    # print('Loss function = ' + str(square_error))
    if L2erroronQ < 0.000001:
        break
    if (i + 1) % 50 == 0:
        print('Finished iteration number ' + str(i + 1) + ', saving model')
        # net.save('./model2')


