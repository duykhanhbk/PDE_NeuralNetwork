import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import random
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PDENet import PDENet
from PDENetLight import PDENetLight
import time

def laplacian(f, input_vector):
    gradient = grad(f, input_vector, create_graph=True)[0]
    ux = gradient.take(torch.tensor([0]))
    uxx = grad(ux, input_vector, create_graph=True)[0].take(torch.tensor([0]))
    uy = gradient.take(torch.tensor([1]))
    uyy = grad(uy, input_vector, create_graph=True)[0].take(torch.tensor([1]))
    return uxx + uyy


def boundary_condition(input):
    return input.take(torch.tensor([1])) + input.take(torch.tensor([1]))**2


# -laplace(u) = f
def right_hand_side(input):
    return 0


def exact_solution(spatial_time_point):
    return 3 + spatial_time_point.take(torch.tensor([1])) +0.5*(spatial_time_point.take(torch.tensor([1]))**2 - spatial_time_point.take(torch.tensor([0]))**2)


def exact_solution_scalar_value(x, y):
    return 3 + y + 0.5*(y**2 - x**2)


def random_data_points(batch_size):
    Omegapoints = []
    boundary_points = []

    radius = math.sqrt(6)
    for i in range(batch_size):
        phi = 2 * math.pi * random.random()
        r = radius * random.random()
        Omegapoints.append([r*math.cos(phi), r*math.sin(phi)])

    for i in range(batch_size):
        phi = 2 * math.pi * random.random()
        boundary_points.append([radius * math.cos(phi), radius * math.sin(phi)])
    return Omegapoints, boundary_points


def batch_loss(net, datapoints):
    G1 = G2 = 0
    Omegapoints, boundary_points = datapoints
    for Omegapoint in Omegapoints:
        Omegapoint_input = Variable(torch.Tensor(Omegapoint).resize_(2, 1), requires_grad=True)
        Omegapoint_output = net(Omegapoint_input)
        G1 += (- laplacian(Omegapoint_output, Omegapoint_input) - right_hand_side(Omegapoint_input)) ** 2

    for boundary_point in boundary_points:
        boundary_point_input = Variable(torch.Tensor(boundary_point).resize_(2, 1), requires_grad=True)
        boundary_point_output = net(boundary_point_input)
        G2 += (boundary_point_output - boundary_condition(boundary_point_input))**2

    G1 = G1 / len(Omegapoints)
    G2 = G2 / len(boundary_points)
    return G1 + G2

def unit_square_grid(step):
    X = [x * step for x in range(0, 1 + math.floor(1 / step))]
    return list(itertools.product(X, repeat=d))


def random_points_in_circle():
    points = []
    radius = math.sqrt(6)
    for i in range(2000):
        phi = 2 * math.pi * random.random()
        r = radius * random.random()
        points.append([r*math.cos(phi), r*math.sin(phi)])
    return points


def train(no_hidden_neuron, learning_rate, num_gd):
    net = PDENetLight(2, no_hidden_neuron)
    net.init_weights()
    l2_errors = []
    losses = []
    avg_eoc = 0

    # net = torch.load('./model3')
    torch.set_printoptions(precision=10)
    learning_rate = learning_rate
    iterations_count = 0
    start_training = time.time()
    for i in range(20):
        print('Iteration number: ' + str(i + 1))
        sample = random_data_points(1000)  # sample space time point
        start_ite = time.time()
        for j in range(num_gd):
            # print('Batch iteration number: ' + str(j + 1))
            net.zero_grad()
            square_error = batch_loss(net, sample)  # calculate square error loss
            square_error.backward()  # calculate gradient of square loss w.r.t the parameters
            print('Batch loss: ' + str(square_error))
            for param in net.parameters():
                param.data -= learning_rate*param.grad.data
            if j == 0:
                losses.append(square_error.item())

        end_ite = time.time()
        ite_time = end_ite - start_ite
        total_time = end_ite - start_training
        print('This iteration took ' + str(ite_time) + ' seconds')
        print('Total training time elapsed ' + str(total_time / 60) + ' minutes')
        L2_error = 0
        for point in random_points_in_circle():
            point_input = torch.Tensor(point).resize_(2, 1)
            L2_error += (net(point_input) - exact_solution(point_input)) ** 2
        L2_error /= 121
        l2_errors.append(L2_error.item())
        if i > 0:
            eoc = L2_error.item() / l2_errors[i - 1]
            avg_eoc += eoc
            print(eoc)
    print(avg_eoc/19)


def fitness(individual):
    return 0


def mutate(individual):
    random_chromo = random.uniform(0,1)

    if 0 <= random_chromo < 0.333: #mutate number of hidden neuron
        num_neuron = individual[0]
        num_neuron += random.randint(-1,1)
        if num_neuron == 0: num_neuron = 1
        if num_neuron == 11: num_neuron = 10
        individual[0] = num_neuron

    elif random_chromo < 0.666: #mutate initial learning rate
        ini_learn_rate = individual[1]
        ini_learn_rate += random.gauss(0, 0.0001)

        if ini_learn_rate < 0: ini_learn_rate = 0.001
        individual[1] = ini_learn_rate

    else:
        decay = individual[2]
        decay += random.gauss(0,0.05)
        if decay < 0: decay = 0.5
        individual[2] = decay


def crossover(individual1, individual2):
    return [(individual1[i] + individual2[i])/2 for i in range(3)]


population = []
train(10,0.0021,1)
