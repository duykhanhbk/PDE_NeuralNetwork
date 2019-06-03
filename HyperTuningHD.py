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
d = 10


def ut(f, input_vector):
    ut = grad(f, input_vector, create_graph=True)[0].take(torch.tensor([d - 1]))
    return ut


def laplacian(f, input_vector):
    gradient = grad(f, input_vector, create_graph=True)[0]
    lapla = 0
    for i in range(d - 1):
        uxi = gradient.take(torch.tensor([i]))
        uxixi = grad(uxi, input_vector, create_graph=True)[0].take(torch.tensor([i]))
        lapla = lapla + uxixi
    return lapla


def boundary_condition(input):
    return 0


def initial_condition(input):
    inicond = 1
    for i in range(d - 1):
        inicond = inicond * math.sin(math.pi * input.take(torch.tensor([i])))
    return inicond


def right_hand_side(input):
    rhs = 1
    rhs = rhs * (1 + 2 * math.pi * math.pi) * math.exp(input.take(torch.tensor([d - 1])))
    for i in range(d - 1):
        rhs = rhs * math.sin(math.pi * input.take(torch.tensor([i])))
    return rhs


def exact_solution(input):
    e_solu = 1
    e_solu = e_solu * math.exp(input.take(torch.tensor([d - 1])))
    for i in range(d - 1):
        e_solu = e_solu * math.sin(math.pi * input.take(torch.tensor([i])))
    return e_solu


def random_data_points(batch_size):
    Spoints = []
    Omegapoints = []
    Qpoints = []
    for i in range(batch_size):
        Qpoint = [random.uniform(0, 1) for i in range(d)]
        Qpoints.append(Qpoint)
    for i in range(batch_size):
        random_axis = random.randint(0, d - 1)
        random_bound = random.randint(0, 1)
        Spoint = [random.uniform(0, 1) for i in range(d)]
        Spoint[random_axis] = random_bound
        Spoints.append(Spoint)
    for i in range(batch_size):
        Omegapoint = [random.uniform(0, 1) for i in range(d)]
    Omegapoint[d - 1] = 0
    Omegapoints.append(Omegapoint)
    return Qpoints, Spoints, Omegapoints


def batch_loss(net, datapoints):
    Qpoints, Spoints, Omegapoints = datapoints
    G1 = G2 = G3 = 0
    for Qpoint in Qpoints:
        Qpoint_input = Variable(torch.Tensor(Qpoint).resize_(d, 1), requires_grad=True)
        Qpoint_output = net(Qpoint_input)
        G1 += (ut(Qpoint_output, Qpoint_input) - laplacian(Qpoint_output, Qpoint_input) - right_hand_side(
            Qpoint_input)) ** 2

    for Spoint in Spoints:
        Spoint_input = Variable(torch.Tensor(Spoint).resize_(d, 1), requires_grad=True)
        Spoint_output = net(Spoint_input)
        G2 += (Spoint_output - boundary_condition(Spoint_input)) ** 2

    for Omegapoint in Omegapoints:
        Omegapoint_input = Variable(torch.Tensor(Omegapoint).resize_(d, 1), requires_grad=True)
        Omegapoint_output = net(Omegapoint_input)
        G3 += (Omegapoint_output - initial_condition(Omegapoint_input)) ** 2
    G1 = G1 / len(Qpoints)
    G2 = G2 / len(Spoints)
    G3 = G3 / len(Omegapoints)
    return G1 + G2 + G3


def unit_square_grid(step):
    X = [x * step for x in range(0, 1 + math.floor(1 / step))]
    return list(itertools.product(X, repeat=d))


def random_grid(num_points):
    grid = []
    for i in range(num_points):
        point = [random.uniform(0, 1) for i in range(d)]
        grid.append(point)
    return grid


def train(no_hidden_neuron, learning_rat, num_gd):
    net = PDENetLight(d, no_hidden_neuron)
    net.init_weights()

    l2_errors = []
    losses = []
    avg_eoc = 0
    torch.set_printoptions(precision=10)
    iterations_count = 0
    start_training = time.time()
    learning_rate = learning_rat
    for i in range(20):
        print('Iteration number: ' + str(i + 1))
        sample = random_data_points(1000)  # sample space time point
        start_ite = time.time()
        for j in range(num_gd):
            print('Batch iteration number: ' + str(j + 1))
            net.zero_grad()
            square_error = batch_loss(net, sample)  # calculate square error loss
            square_error.backward()  # calculate gradient of square loss w.r.t the parameters
            print('Batch loss: ' + str(square_error))
            for param in net.parameters():
                param.data -= learning_rate * param.grad.data
            if j == 0:
                losses.append(square_error.item())
        end_ite = time.time()
        ite_time = end_ite - start_ite
        total_time = end_ite - start_training
        print('This iteration took ' + str(ite_time) + ' seconds')
        print('Total training time elapsed ' + str(total_time / 60) + ' minutes')
        L2_error = 0
        # for point in unit_square_grid(0.1):
        for point in random_grid(50000):
            point_input = torch.Tensor(point).resize_(d, 1)
            L2_error += (net(point_input) - exact_solution(point_input)) ** 2
        L2_error /= 50000

        l2_errors.append(L2_error.item())
        if i > 0:
            eoc = L2_error.item() / l2_errors[i - 1]
            avg_eoc += eoc
            print(eoc)
    print(avg_eoc / 19)


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
train(8,0.0043,5)
