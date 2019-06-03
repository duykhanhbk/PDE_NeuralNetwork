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
from PDENetLight import PDENetLight
import time

def exact_solution(spatial_time_point):
    return spatial_time_point.take(torch.tensor([0]))/(1 + spatial_time_point.take(torch.tensor([1])))


def exact_solution_scalar_value(x, y):
    return x/(1+y)


def random_data_points(batch_size):
    spatial_points = [(random.uniform(-1,1), random.uniform(0,1)) for ii in range(batch_size)]
    space_initital_points = [(random.uniform(-1,1), 0) for ii in range(batch_size)]
    boundary_points = [(2*random.randint(0,1)-1, random.uniform(0,1)) for ii in range(batch_size)]

    # i think adding the corners make performance and visual result

    boundary_points.append((-1, 0))
    boundary_points.append((-1, 1))
    boundary_points.append((1, 1))
    boundary_points.append((1, 1))

    return spatial_points, space_initital_points, boundary_points


def batch_loss(net, datapoints):
    G1 = G2 = G3 = 0
    spatial_points, space_initital_points, boundary_points = datapoints
    for spatial_point in spatial_points:
        spatial_input = Variable(torch.Tensor(spatial_point).resize_(2, 1), requires_grad=True)
        u = net(spatial_input)
        u_x = grad(u, spatial_input, create_graph=True)[0].take(torch.tensor([0]))
        u_t = grad(u, spatial_input, create_graph=True)[0].take(torch.tensor([1]))
        G1 += (u_t + u*u_x) ** 2

    for initital_point in space_initital_points:
        initital_input = Variable(torch.Tensor(initital_point).resize_(2, 1), requires_grad=True)
        initial_output = net(initital_input)
        G2 += (initial_output - initital_input.take(torch.tensor([0])))**2

    for boundary_point in boundary_points:
        boundary_input = Variable(torch.Tensor(boundary_point).resize_(2, 1), requires_grad=True)
        boundary_output = net(boundary_input)
        G3 += (boundary_output - boundary_input.take(torch.tensor([0]))/(1 + boundary_input.take(torch.tensor([1]))))**2

    G1 = G1 / len(spatial_points)
    G2 = G2 / len(space_initital_points)
    G3 = G3/ len(boundary_points)
    return G1 + G2 + G3


def unit_square_grid(step):
    X = [x * step for x in range(-1, 1 + math.floor(1/step))]
    Y = [x * step for x in range(0, 1 + math.floor(1/step))]
    return list(itertools.product(X, Y))


def plot_estimation_and_exact_solution(file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-1, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([net(torch.tensor([x, y]).resize_(2,1)).item() for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    zs = np.array([exact_solution_scalar_value(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z1 = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, color='red')
    ax.plot_surface(X, Y, Z1, color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')

    # plt.show()
    plt.savefig(file_name)
    plt.close("all")


net = PDENetLight(2, 10)
net.init_weights()

plot_estimation_and_exact_solution('./result/testnoBatchBurger/Initial.png')
l2_errors = []
losses = []

torch.set_printoptions(precision=10)
learning_rate = 0.001
iterations_count = 0
start_training = time.time()
for i in range(200):
    print('Iteration number: ' + str(i + 1))
    sample = random_data_points(1000)  # sample space time point
    start_ite = time.time()
    for j in range(1):
        print('Batch iteration number: ' + str(j + 1))
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
    for point in unit_square_grid(0.1):
        point_input = torch.Tensor(point).resize_(2,1)
        L2_error += (net(point_input) - exact_solution(point_input))**2
    L2_error /= 121

    l2_errors.append(L2_error.item())
    print('L2error = ' + str(L2_error))
    # print('Loss function = ' + str(square_error))

    print('Finished iteration number ' + str(i + 1) + ', saving model')
    # model_path = './result/testnoBatchBurger/model' + str(i+1) + 'th_ite'
    # net.save(model_path)
    fig_path = './result/testnoBatchBurger/' + str(i+1) + 'th_ite.png'
    plot_estimation_and_exact_solution(fig_path)

print('Plotting and saving convergence history')
epochs = [i + 1 for i in range(200)]
plt.xticks(np.arange(0, 201, 20))

plt.figure(0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, losses, color = 'blue')
plt.savefig('./result/testnoBatchBurger/loss.png')
with open('./result/testnoBatchBurger/batchloss.txt', 'w+') as f:
    for loss in losses:
        f.write(str(loss) + '\n')

plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('L2error')
plt.plot(epochs, l2_errors, color = 'blue')
plt.savefig('./result/testnoBatchBurger/l2error.png')
with open('./result/testnoBatchBurger/L2error.txt', 'w+') as f:
    for l2_error in l2_errors:
        f.write(str(l2_error) + '\n')
