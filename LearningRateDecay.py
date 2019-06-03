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

# dimension of input (x,t) \in R^d
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


net = PDENet(d, 100)
net.init_weights()

l2_errors = []
losses = []

torch.set_printoptions(precision=10)
ini_learning_rate = 0.001
decay_rate = 0.8
iterations_count = 0

for i in range(1000):
    print('Iteration number: ' + str(i + 1))
    sample = random_data_points(1000)  # sample space time point

    for j in range(4):
        print('Batch iteration number: ' + str(j + 1))
        net.zero_grad()
        square_error = batch_loss(net, sample)  # calculate square error loss
        square_error.backward()  # calculate gradient of square loss w.r.t the parameters
        print('Batch loss: ' + str(square_error))
        for param in net.parameters():
            learning_rate = ini_learning_rate/(1 + decay_rate*i)
            param.data -= learning_rate * param.grad.data
        if j == 0:
            losses.append(square_error.item())

    L2_error = 0
    # for point in unit_square_grid(0.1):
    for point in random_grid(50000):
        point_input = torch.Tensor(point).resize_(d, 1)
        L2_error = (net(point_input) - exact_solution(point_input)) ** 2
    # L2_error /= 50000

    l2_errors.append(L2_error.item())
    print('L2error = ' + str(L2_error))

    if (i + 1) % 50 == 0:
        print('Finished iteration number ' + str(i + 1) + ', saving model')
        model_path = './result/HighDimension/model' + str(i + 1) + 'th_ite'
        net.save(model_path)
        fig_path = './result/HighDimension/' + str(i + 1) + 'th_ite.png'

print('Plotting and saving convergence history')
epochs = [i + 1 for i in range(1000)]
plt.xticks(np.arange(0, 1001, 100))

plt.figure(0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, losses, color = 'blue')
plt.savefig('./result/test/loss.png')

with open('./result/test/batchloss.txt', 'w+') as f:
    for loss in losses:
        f.write(str(loss) + '\r\n')

plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('L2error')
plt.plot(epochs, l2_errors, color = 'blue')
plt.savefig('./result/test/l2error.png')
with open('./result/test/L2error.txt', 'w+') as f:
    for l2_error in l2_errors:
        f.write(str(l2_error) + '\r\n')
