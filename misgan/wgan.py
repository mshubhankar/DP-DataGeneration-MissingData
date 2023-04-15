import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from matplotlib.patches import Rectangle
import pylab as plt

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Must sub-class ConvGenerator to provide transform()
class ConvGenerator(nn.Module):
    def __init__(self, output_size, latent_size=32):
        super().__init__()

        self.DIM = 16
        self.latent_size = latent_size
      
        self.block1 = nn.Sequential(
            nn.Linear(latent_size, self.DIM),
            nn.ReLU(True),
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(True),
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(True),
            nn.Linear(self.DIM, output_size)
        )

    def forward(self, input):       
        net2 = self.block1(input.clone())
        return self.transform(net2.clone())


class ConvDataGenerator(ConvGenerator):
    def __init__(self, output_size, latent_size=32):
        super().__init__(output_size=output_size, latent_size=latent_size)
        self.transform = lambda x: torch.sigmoid(x)


class ConvMaskGenerator(ConvGenerator):
    def __init__(self, output_size, latent_size=32, temperature=.66):
        super().__init__(output_size=output_size, latent_size=latent_size)
        self.transform = lambda x: torch.sigmoid(x / temperature)

class ConvCritic(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.DIM = 32
        self.main = nn.Sequential(
            nn.Linear(input_size, self.DIM),
            nn.ReLU(True),
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(True),
            nn.Linear(self.DIM, self.DIM),
            nn.ReLU(True),
        )
        self.output = nn.Linear(self.DIM, 1)

    def forward(self, input):
        net = self.main(input)
        net2 = self.output(net)
        return net2.view(-1)

class CriticUpdater:
    def __init__(self, critic, critic_optimizer, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, 1, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)

    def __call__(self, real, fake):
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()

        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()