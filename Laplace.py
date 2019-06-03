import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools
import time
from PDENetLight import PDENetLight


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


def plot_estimation_and_exact_solution(file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 3, 0.05)
    X, Y = np.meshgrid(x, y)
    condition = X ** 2 + Y ** 2 <= 6
    zs = np.array([net(torch.tensor([x, y]).resize_(2,1)).item() for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    zs = np.array([exact_solution_scalar_value(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z1 = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, color='red')
    ax.plot_surface(X, Y, Z1, color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plt.show()
    plt.savefig(file_name)
    plt.close("all")


def random_points_in_circle():
    points = []
    radius = math.sqrt(6)
    for i in range(2000):
        phi = 2 * math.pi * random.random()
        r = radius * random.random()
        points.append([r*math.cos(phi), r*math.sin(phi)])
    return points


net = PDENetLight(2, 10)
net.init_weights()

plot_estimation_and_exact_solution('./result/testnoBatch/Initial.png')
l2_errors = []
losses = []

# net = torch.load('./model3')
torch.set_printoptions(precision=10)
learning_rate = 0.001
iterations_count = 0
start_training = time.time()
for i in range(200):
    print('Iteration number: ' + str(i + 1))
    sample = random_data_points(1000)  # sample space time point
    start_ite = time.time()
    # for j in range(4):
    #     # print('Batch iteration number: ' + str(j + 1))
    #     net.zero_grad()
    #     square_error = batch_loss(net, sample)  # calculate square error loss
    #     square_error.backward()  # calculate gradient of square loss w.r.t the parameters
    #     print('Batch loss: ' + str(square_error))
    #     for param in net.parameters():
    #         param.data -= learning_rate*param.grad.data
    #     if j == 0:
    #         losses.append(square_error.item())
    net.zero_grad()
    square_error = batch_loss(net, sample)  # calculate square error loss
    square_error.backward()  # calculate gradient of square loss w.r.t the parameters
    print('Batch loss: ' + str(square_error))
    for param in net.parameters():
        param.data -= learning_rate * param.grad.data
    losses.append(square_error.item())
    end_ite = time.time()
    ite_time = end_ite - start_ite
    total_time = end_ite - start_training
    print('This iteration took ' + str(ite_time) + ' seconds')
    print('Total training time elapsed ' + str(total_time/60) + ' minutes')
    L2_error = 0
    for point in random_points_in_circle():
        point_input = torch.Tensor(point).resize_(2,1)
        L2_error += (net(point_input) - exact_solution(point_input))**2
    L2_error /= 121
    l2_errors.append(L2_error.item())

    print('L2error = ' + str(L2_error))
    # print('Loss function = ' + str(square_error))

    print('Finished iteration number ' + str(i + 1) + ', saving model')
    model_path = './result/testnoBatch/model' + str(i+1) + 'th_ite'
   # net.save(model_path)
    fig_path = './result/testnoBatch/' + str(i+1) + 'th_ite.png'
    plot_estimation_and_exact_solution(fig_path)

print('Plotting and saving convergence history')
epochs = [i + 1 for i in range(200)]
plt.xticks(np.arange(0, 201, 20))

plt.figure(0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, losses, color = 'blue')
plt.savefig('./result/testnoBatch/loss.png')

with open('./result/testnoBatch/batchloss.txt', 'w+') as f:
    for loss in losses:
        f.write(str(loss) + '\n')

plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('L2error')

plt.plot(epochs, l2_errors, color = 'blue')
plt.savefig('./result/testnoBatch/l2error.png')
with open('./result/testnoBatch/L2error.txt', 'w+') as f:
    for l2_error in l2_errors:
        f.write(str(l2_error) + '\n')
