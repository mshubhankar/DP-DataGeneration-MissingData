import torch
import torch.nn as nn

def nr(mu, sigma):
    non_zero = torch.not_equal(sigma, 0.)
    new_sigma = torch.where(non_zero, sigma, 1e-20)
    sqrt_sigma = torch.sqrt(new_sigma)

    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(-torch.square(w), 2)), (2*torch.pi)**0.5) + \
                     torch.mul(torch.div(w,2.), 1 + torch.erf(torch.div(w, 2**0.5))))
    nr_values = torch.where(non_zero, nr_values, (mu + torch.abs(mu)) / 2.)
    
    return nr_values

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, binary=True, device='cpu'):
        super(Generator, self).__init__()
        def block(inp, out, Activation, device):
            return nn.Sequential(
                nn.Linear(inp, out, bias=False),
                nn.LayerNorm(out),
                Activation(),
            ).to(device)

        self.block_0 = block(input_dim, input_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)
        self.block_1 = block(input_dim, input_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)
        self.block_2 = block(input_dim, output_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)

    def forward(self, x):
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x = self.block_2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        ).to(device)

    def forward(self, x):
        return self.model(x)

class DiscriminatorForMissing(nn.Module):
    def __init__(self, input_dim, means, weights, cov, n_distribution, device='cpu'):
        super(DiscriminatorForMissing, self).__init__()
        
        self.means = means
        self.weights = weights
        self.covs = torch.abs(cov)
        self.n_distribution = n_distribution
        self.gamma = torch.empty(1).normal_(mean=1, std=1)

        
        self.first_layer = nn.Linear(input_dim, (2 * input_dim) // 3)
        self.first_relu = nn.ReLU()

        self.model = nn.Sequential(
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        ).to(device)


    def forward(self, x):
        if torch.any(x.isnan()):
            where_isnan = torch.isnan(x)
            size = x.shape
            p_ = nn.functional.softmax(self.weights, dim=0)
            weights2 = torch.square(self.first_layer.weight)
            final_distributions = list()
            final_q = list()

            for i in range(self.n_distribution):
                data_miss = torch.where(where_isnan, self.means[i], x)
                miss_cov = torch.where(where_isnan, self.covs[i], torch.zeros(x.shape))
                layer_1_m = self.first_relu(self.first_layer(data_miss))
                layer_1_m = nr(layer_1_m, torch.matmul(miss_cov, weights2.T))
                
                norm = torch.subtract(data_miss, self.means[i])
                norm = torch.square(norm)
                q = torch.where(~where_isnan, torch.add(self.gamma, self.covs[i]), torch.ones(x.shape))
                norm = torch.div(norm, q)
                norm = torch.sum(norm)

                q = torch.log(q)
                q = torch.sum(q)

                q = torch.add(q, norm)

                norm = torch.sum(~where_isnan)
                norm = torch.mul(norm, torch.log(torch.Tensor([2*torch.pi])))
                q = torch.add(q,norm)
                q = -0.5 * q
                final_distributions.append(layer_1_m)
                final_q.append(q)

            distrib = torch.stack(final_distributions)
            log_q = torch.stack(final_q)
            log_q = torch.add(log_q, torch.log(p_))
            r = nn.functional.softmax(log_q, dim=0)

            layer_1_miss = torch.multiply(distrib, r.unsqueeze(dim=2))
            layer_1_miss = torch.sum(layer_1_miss,dim=0)
            return self.model(layer_1_miss)
        else:
            return self.model(self.first_relu(self.first_layer(x)))

