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
from .wgan import ConvDataGenerator, ConvMaskGenerator, CriticUpdater, ConvCritic
from .analysis import epsilon, noise_mult

class misgan:
    def __init__(self, params):
        self.params = params
        self.epsilon = params['epsilon']
        if self.epsilon <= 99:
            self.private = True

    def generate(self, data):

        def master_hook_adder(module, grad_input, grad_output):
            '''
            global hook

            :param module:
            :param grad_input:
            :param grad_output:
            :return:
            '''
            global dynamic_hook_function
            return dynamic_hook_function(module, grad_input, grad_output)


        def dummy_hook(module, grad_input, grad_output):
            '''
            dummy hook

            :param module:
            :param grad_input:
            :param grad_output:
            :return:
            '''
            pass

        def dp_conv_hook(module, grad_input, grad_output):
            '''
            gradient modification + noise hook

            :param module:
            :param grad_input:
            :param grad_output:
            :return:
            '''
            CLIP_BOUND = 1
            SENSITIVITY = 2
            global noise_multiplier
            ### get grad wrt. input (image)

            grad_wrt_image = grad_input[1]

            grad_input_shape = grad_wrt_image.size()
            batchsize = grad_input_shape[0]

            clip_bound_ = CLIP_BOUND / batchsize

            grad_wrt_image = grad_wrt_image.view(batchsize, -1)
            grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

            ### clip
            clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
            clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
            clip_coef = clip_coef.unsqueeze(-1)
            grad_wrt_image = clip_coef * grad_wrt_image

            ### add noise
            noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
            grad_wrt_image = grad_wrt_image + noise
            grad_input_new = [grad_wrt_image.view(grad_input_shape)]
            modified_grad_input = [None] * len(grad_input)
            modified_grad_input[1] = grad_input_new[0]
            return tuple(modified_grad_input)


        def mask_data(data, mask, tau=0):
            data = torch.nan_to_num(data)
            return mask * data + (1 - mask) * tau

        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        nz = 32   # dimensionality of the latent code
        n_critic = 5
        plot_interval = 5
        alpha = .2
        batch_size = 16
        epochs = 300
        data = torch.tensor(data.values.astype(np.float32))
        data_size = data.shape[0]
        if self.private:
            delta = float(f'1e-{len(str(data_size)) + 1}')
            gen_iterations = (epochs * (data_size // batch_size)) // n_critic
            global noise_multiplier
            noise_multiplier = noise_mult(data.shape[0], batch_size, self.epsilon, 2 * gen_iterations, delta)


        data_gen = ConvDataGenerator(output_size=data.shape[1]).to(device)
        mask_gen = ConvMaskGenerator(output_size=data.shape[1]).to(device)

        data_critic = ConvCritic(input_size=data.shape[1]).to(device)
        mask_critic = ConvCritic(input_size=data.shape[1]).to(device)

        data_noise = torch.empty(batch_size, nz, device=device)
        mask_noise = torch.empty(batch_size, nz, device=device)

        lrate = 1e-4
        data_gen_optimizer = optim.Adam(
            data_gen.parameters(), lr=lrate, betas=(.5, .9))
        mask_gen_optimizer = optim.Adam(
            mask_gen.parameters(), lr=lrate, betas=(.5, .9))

        data_critic_optimizer = optim.Adam(
            data_critic.parameters(), lr=lrate, betas=(.5, .9))
        mask_critic_optimizer = optim.Adam(
            mask_critic.parameters(), lr=lrate, betas=(.5, .9))

        update_data_critic = CriticUpdater(
            data_critic, data_critic_optimizer, batch_size)
        update_mask_critic = CriticUpdater(
            mask_critic, mask_critic_optimizer, batch_size)

        
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

        
        generator_updates = 0

        global dynamic_hook_function
        dynamic_hook_function = dummy_hook
        data_critic.main[0].register_backward_hook(master_hook_adder)
        mask_critic.main[0].register_backward_hook(master_hook_adder)

        for epoch in range(epochs):
            for real_data in data_loader:
                real_mask = ~torch.isnan(real_data)

                real_data = real_data.to(device)
                real_mask = real_mask.to(device).float()

                # Update discriminators' parameters
                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise)
                fake_mask = mask_gen(mask_noise)
                
                dynamic_hook_function = dummy_hook

                masked_fake_data = mask_data(fake_data, fake_mask)
                masked_real_data = mask_data(real_data, real_mask)

                update_data_critic(masked_real_data, masked_fake_data)
                update_mask_critic(real_mask, fake_mask)

                generator_updates += 1

                if generator_updates == n_critic:
                    generator_updates = 0

                    # Update generators' parameters
                    for p in data_critic.parameters():
                        p.requires_grad_(False)
                    for p in mask_critic.parameters():
                        p.requires_grad_(False)

                    if self.private:
                        dynamic_hook_function = dp_conv_hook

                    data_noise.normal_()
                    mask_noise.normal_()

                    fake_data = data_gen(data_noise)
                    fake_mask = mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask)

                    data_loss = -data_critic(masked_fake_data).mean()
                    data_gen.zero_grad()
                    data_loss.backward(retain_graph=True)
                    data_gen_optimizer.step()

                    data_noise.normal_()
                    mask_noise.normal_()

                    fake_data = data_gen(data_noise)
                    fake_mask = mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask)
                    
                    data_loss = -data_critic(masked_fake_data).mean()
                    mask_loss = -mask_critic(fake_mask).mean()
                    mask_gen.zero_grad()

                    total_loss = mask_loss + data_loss * alpha
                    total_loss.backward()
                    mask_gen_optimizer.step()

                    for p in data_critic.parameters():
                        p.requires_grad_(True)
                    for p in mask_critic.parameters():
                        p.requires_grad_(True)

            if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
                print(f'{epoch + 1}/{epochs} Data loss: {data_loss} Mask loss: {mask_loss}')

        data_gen.eval()
        mask_gen.eval()

        with torch.no_grad():
            full_noise = torch.empty(data.shape[0], nz, device=device)
            full_noise.normal_()
            syn_data = data_gen(full_noise)

        return syn_data