import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as sp
from matplotlib import pyplot as plt 

import model
import helper
import itertools

# inspired by https://gist.github.com/LMescheder/b4d57e094cc8522497fdb5cf8fb44676

# Hyperparameters 
batch_size = 128
sigma = 1.0
z_dim = 64
learning_rate = 1e-4
reg_param = 10.

# Define networks 
generator = model.Generator(z_dim)
discriminator = model.Discriminator(2)

optim_gen = torch.optim.RMSprop(generator.parameters(), lr=learning_rate) 
optim_dis = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate) 

# Print Real distribution 
mu = np.vstack([np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(batch_size))
x_real = mu + sigma * np.random.normal(0.0, 0.0, [batch_size, 2]) 
helper.display_result(x_real, cmap='Reds')
plt.savefig("image/real.png")

for epoch in range(10000):
    # Getting the target distribution 
    mu = np.vstack([np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(batch_size))
    x_real = mu + sigma * np.random.normal(0.0, 0.0, [batch_size, 2]) 

    # Generate the var for generator
    z = np.random.normal(0.0, 1.0, [batch_size, z_dim])
    x_fake = generator(torch.from_numpy(z).float())
    d_out_real = discriminator(torch.from_numpy(x_real).float()) 
    d_out_fake = discriminator(x_fake.detach())

    optim_gen.zero_grad()
    optim_dis.zero_grad()

    # Define loss
    real_label = torch.ones(batch_size, 1)

    discriminator_real_loss = -torch.mean(real_label * torch.log(d_out_real))
    discriminator_fake_loss = -torch.mean(real_label * torch.log(1 - d_out_fake))
    
    d_loss = discriminator_real_loss + discriminator_fake_loss
    g_loss = -torch.mean(real_label * torch.log(discriminator(x_fake)))

    # Get Grad
    grad_d = torch.autograd.grad(d_loss, discriminator.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, generator.parameters(), create_graph=True)

    d_param = [d for d in discriminator.parameters()]
    g_param = [g for g in generator.parameters()]

    grads = grad_d + grad_g
    var = d_param + g_param

    reg = 0.5 * sum(torch.sum(g**2) for g in grads)
    Jgrads = torch.autograd.grad(reg, var)

    final_grad = [g + reg_param * j for j, g in zip(Jgrads, grads)]
    params = itertools.chain(discriminator.parameters(), generator.parameters())

    for p, g in zip(params, final_grad):
        p.grad = g

    optim_gen.step()
    optim_dis.step()

    if epoch % 500 == 0:
        print("At", epoch)
        helper.display_result(x_fake.data.numpy(), cmap='Blues')
        plt.savefig("image/" + str(epoch) + ".png")
