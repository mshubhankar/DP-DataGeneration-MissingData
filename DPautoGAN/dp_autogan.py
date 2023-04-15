# Code reference: https://github.com/DPautoGAN/DPautoGAN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score, explained_variance_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso
import sys
sys.path.insert(0, './DPautoGAN')

import torch
from torch import nn
import torch.nn.functional as F

from .dp_wgan import Generator, Discriminator, DiscriminatorForMissing
from .dp_autoencoder import Autoencoder, AutoencoderForMissing
from .dp_optimizer import *
from .sampling import get_data_loaders
from .analysis import epsilon, noise_mult

class DPAUTOGAN:
    def __init__(self,
                 params,
                 b1 = 0.9,
                 b2 = 0.999,
                 binary = False,
                 compress_dim = 15,
                 delta = 1e-5,
                 device = 'cpu',
                 iterations = 2000,
                 lr = 0.005,
                 l2_penalty = 0.,
                 l2_norm_clip = 0.012,
                 minibatch_size = 512,
                 microbatch_size = 1,
                 noise_multiplier = 2.5,
                 nonprivate = False,
                 alpha = 0.99,
                 gan_binary = False,
                 clip_value = 0.01,
                 d_updates = 15,
                 gan_delta = 1e-5,
                 gan_device = 'cpu',
                 gan_iterations = 2000,
                 latent_dim = 64,
                 gan_lr = 0.005,
                 gan_l2_penalty = 0.,
                 gan_l2_norm_clip = 0.022,
                 gan_minibatch_size = 128,
                 gan_microbatch_size = 1,
                 gan_noise_multiplier = 3.5,
                 n_distribution = 3,
                 enhanced = False,
                 gan_nonprivate = False):
        
        self.epsilon = params['epsilon']
        self.enhanced = enhanced
        self.n_distribution = n_distribution
        if self.epsilon > 99:
            nonprivate = True
            gan_nonprivate = True
        else:
            ae_epsilon = self.epsilon/2
            gan_epsilon = self.epsilon/2
        self.algo_name = "DPautoGAN"
        self.params = params
        self._ae_params = {
            'b1': b1,
            'b2': b2,
            'binary': binary,
            'compress_dim': compress_dim,
            'delta': delta,
            'device': device,
            'iterations': iterations,
            'lr': lr,
            'l2_penalty': l2_penalty,
            'l2_norm_clip': l2_norm_clip,
            'minibatch_size': minibatch_size,
            'microbatch_size': microbatch_size,
            'noise_multiplier': noise_multiplier,
            'ae_epsilon': ae_epsilon,
            'nonprivate': nonprivate,
        }

        self._gan_params = {
            'alpha': alpha,
            'binary': gan_binary,
            'clip_value': clip_value,
            'd_updates': d_updates,
            'delta': gan_delta,
            'device': gan_device,
            'iterations': gan_iterations,
            'latent_dim': latent_dim,
            'lr': gan_lr,
            'l2_penalty': gan_l2_penalty,
            'l2_norm_clip': gan_l2_norm_clip,
            'minibatch_size': gan_minibatch_size,
            'microbatch_size': gan_microbatch_size,
            'noise_multiplier': gan_noise_multiplier,
            'gan_epsilon': gan_epsilon,
            'nonprivate': gan_nonprivate,
        }
        

    def generate(self, X_encoded):

        if X_encoded.isnull().values.any():
            self.enhanced = True
            
        def prep_x(x_true, x_pred): #where missing make zeros for both prediction and true
            check_is_nan = torch.isnan(x_true)
            x_true = torch.where(check_is_nan, torch.zeros_like(x_true), x_true)
            x_pred = torch.where(check_is_nan, torch.zeros_like(x_pred), x_pred)
            return x_true, x_pred

        np.random.seed(0)
        torch.manual_seed(0) 
        X_encoded = torch.tensor(X_encoded.values.astype(np.float32))

        df = pd.read_csv(self.params['data_loc'])
        df = df.dropna() # Complete case analysis
        if(self.enhanced == True):
            X_complete = X_encoded[~torch.any(X_encoded.isnan(),dim=1)]
        else:
            X_encoded = X_encoded[~torch.any(X_encoded.isnan(),dim=1)]
        datatypes = [
            ('age', 'positive int'),
            ('workclass', 'categorical'),
            ('education-num', 'categorical'),
            ('education', 'categorical'),
            ('marital-status', 'categorical'),
            ('occupation', 'categorical'),
            ('relationship', 'categorical'),
            ('race', 'categorical'),
            ('sex', 'categorical binary'),
            ('capital-gain', 'positive float'),
            ('capital-loss', 'positive float'),
            ('hours-per-week', 'positive int'),
            ('fnlwgt',     'positive int'),
            ('native-country', 'categorical'),
            ('income', 'categorical binary'),
        ]


        weights, ds = [], []
        for name, datatype in datatypes:
            if 'categorical' in datatype:
                num_values = len(np.unique(df[name].dropna()))
                if num_values == 2:
                    weights.append(1.)
                    ds.append((datatype, 1))
                else:
                    for i in range(num_values):
                        weights.append(1. / num_values)
                    ds.append((datatype, num_values))
            else:
                weights.append(1.)
                ds.append((datatype, 1))
        weights = torch.tensor(weights).to(self._ae_params['device'])
        autoencoder_loss = nn.BCELoss()

        if(self.enhanced == True):
            gmm = GaussianMixture(n_components=self.n_distribution, covariance_type='diag').fit(X_complete)
            gmm_weights = torch.Tensor(np.log(gmm.weights_.reshape((-1, 1))))
            gmm_means = torch.Tensor(gmm.means_)
            gmm_covariances = torch.Tensor(gmm.covariances_)
            del gmm

        if(self.enhanced == True):
            autoencoder = AutoencoderForMissing(
                example_dim = len(X_encoded[0]),
                compression_dim  = self._ae_params['compress_dim'],
                means = gmm_means,
                weights = gmm_weights,
                cov = gmm_covariances,
                n_distribution = self.n_distribution,
                binary = self._ae_params['binary'],
                device = self._ae_params['device'],
            )
        else:
            autoencoder = Autoencoder(
                example_dim = len(X_encoded[0]),
                compression_dim  = self._ae_params['compress_dim'],
                binary = self._ae_params['binary'],
                device = self._ae_params['device'],
                )

        print(autoencoder)

        decoder_optimizer = DPAdam(
            l2_norm_clip = self._ae_params['l2_norm_clip'],
            noise_multiplier = self._ae_params['noise_multiplier'],
            minibatch_size = self._ae_params['minibatch_size'],
            microbatch_size = self._ae_params['microbatch_size'],
            nonprivate = self._ae_params['nonprivate'],
            params=autoencoder.get_decoder().parameters(),
            lr = self._ae_params['lr'],
            betas = (self._ae_params['b1'], self._ae_params['b2']),
            weight_decay = self._ae_params['l2_penalty'],
        )

        encoder_optimizer = torch.optim.Adam(
            params = autoencoder.get_encoder().parameters(),
            lr = self._ae_params['lr'] * self._ae_params['microbatch_size'] / self._ae_params['minibatch_size'],
            betas = (self._ae_params['b1'], self._ae_params['b2']),
            weight_decay = self._ae_params['l2_penalty'],
        )

        self._ae_params['noise_multiplier'] = noise_mult(len(X_encoded),
                                                        self._ae_params['minibatch_size'],
                                                        self._ae_params['ae_epsilon'],
                                                        self._ae_params['iterations'],
                                                        self._ae_params['delta']
                                                         )
        print('Achieves ({}, {})-DP'.format(
            epsilon(
                len(X_encoded),
                self._ae_params['minibatch_size'],
                self._ae_params['noise_multiplier'],
                self._ae_params['iterations'],
                self._ae_params['delta']
            ),
            self._ae_params['delta'],
        ))

        minibatch_loader, microbatch_loader = get_data_loaders(
            minibatch_size = self._ae_params['minibatch_size'],
            microbatch_size = self._ae_params['microbatch_size'],
            iterations = self._ae_params['iterations'],
            nonprivate = self._ae_params['nonprivate'],
        )

        train_losses, validation_losses = [], []

        X_encoded = X_encoded.to(self._ae_params['device'])

        for iteration, X_minibatch in enumerate(minibatch_loader(X_encoded)):
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            for X_microbatch in microbatch_loader(X_minibatch):

                decoder_optimizer.zero_microbatch_grad()
                output = autoencoder(X_microbatch)
                if self.enhanced:
                    X_microbatch, output = prep_x(X_microbatch, output)
                loss = autoencoder_loss(output, X_microbatch)
                loss.backward()
                decoder_optimizer.microbatch_step()
            
            if(self.enhanced == True):
                X_encoded_val, _ = prep_x(X_encoded, X_encoded)
                validation_loss = autoencoder_loss(autoencoder(X_encoded_val).detach(), X_encoded_val)
            else:
                validation_loss = autoencoder_loss(autoencoder(X_encoded).detach(), X_encoded)

            encoder_optimizer.step()
            decoder_optimizer.step()

            train_losses.append(loss.item())
            validation_losses.append(validation_loss.item())

            if iteration % 333 == 0:
                print ('[Iteration %d/%d] [Loss: %f] [Validation Loss: %f]' % (
                    iteration, self._ae_params['iterations'], loss.item(), validation_loss.item())
                )

        pd.DataFrame(data={'train': train_losses, 'validation': validation_losses}).plot()

        with open('./DPautoGAN/ae_eps_inf.dat', 'wb') as f:
            torch.save(autoencoder, f)

        
        with open('./DPautoGAN/ae_eps_inf.dat', 'rb') as f:
            autoencoder = torch.load(f)

        decoder = autoencoder.get_decoder()
        
        generator = Generator(
            input_dim = self._gan_params['latent_dim'],
            output_dim = autoencoder.get_compression_dim(),
            binary = self._gan_params['binary'],
            device = self._gan_params['device'],
        )

        g_optimizer = torch.optim.RMSprop(
            params = generator.parameters(),
            lr = self._gan_params['lr'],
            alpha = self._gan_params['alpha'],
            weight_decay = self._gan_params['l2_penalty'],
        )

        
        if(self.enhanced == True):
            discriminator = DiscriminatorForMissing(
            input_dim = len(X_encoded[0]),
            means = gmm_means,
            weights = gmm_weights,
            cov = gmm_covariances,
            n_distribution = self.n_distribution,
            device = self._gan_params['device'],
            )

        else:
            discriminator = Discriminator(
                input_dim = len(X_encoded[0]),
                device = self._gan_params['device'],
            )

        d_optimizer = DPRMSprop(
            l2_norm_clip = self._gan_params['l2_norm_clip'],
            noise_multiplier = self._gan_params['noise_multiplier'],
            minibatch_size = self._gan_params['minibatch_size'],
            microbatch_size = self._gan_params['microbatch_size'],
            nonprivate = self._gan_params['nonprivate'],
            params = discriminator.parameters(),
            lr = self._gan_params['lr'],
            alpha = self._gan_params['alpha'],
            weight_decay = self._gan_params['l2_penalty'],
        )

        print(generator)
        print(discriminator)
        self._gan_params['noise_multiplier'] = noise_mult(len(X_encoded),
                                                        self._gan_params['minibatch_size'],
                                                        self._gan_params['gan_epsilon'],
                                                        self._gan_params['iterations'],
                                                        self._gan_params['delta']
                                                         )
        print('Achieves ({}, {})-DP'.format(
            epsilon(
                len(X_encoded),
                self._gan_params['minibatch_size'],
                self._gan_params['noise_multiplier'],
                self._gan_params['iterations'],
                self._gan_params['delta']
            ),
            self._gan_params['delta'],
        ))

        minibatch_loader2, microbatch_loader2 = get_data_loaders(
            minibatch_size = self._gan_params['minibatch_size'],
            microbatch_size = self._gan_params['microbatch_size'],
            iterations = self._gan_params['iterations'],
            nonprivate = self._gan_params['nonprivate'],
        )

        X_train_encoded = X_encoded.to(self._gan_params['device'])

        torch.manual_seed(0)
        for iterations, X_minibatch in enumerate(minibatch_loader2(X_train_encoded)):
            d_optimizer.zero_grad()

            for real in microbatch_loader2(X_minibatch):
                z = torch.randn(real.size(0), self._gan_params['latent_dim'], device = self._gan_params['device'])
                fake = decoder(generator(z)).detach()
                d_optimizer.zero_microbatch_grad()
                # real, fake = prep_x(real, fake)
                d_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
                d_loss.backward()
                d_optimizer.microbatch_step()

            d_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.data.clamp_(-self._gan_params['clip_value'], self._gan_params['clip_value'])

            if iterations % self._gan_params['d_updates'] == 0:
                z = torch.randn(X_minibatch.size(0), self._gan_params['latent_dim'], device = self._gan_params['device'])
                fake = decoder(generator(z))

                g_optimizer.zero_grad()
                g_loss = -torch.mean(discriminator(fake))
                g_loss.backward()
                g_optimizer.step()

            if iterations % 333 == 0:
                print('[Iteration %d/%d] [D loss: %f] [G loss: %f]' % (
                    iterations, self._gan_params['iterations'], d_loss.item(), g_loss.item()
                ))
        with open('./DPautoGAN/gen_eps_inf.dat', 'wb') as f:
            torch.save(generator, f)

        with open('./DPautoGAN/gen_eps_inf.dat', 'rb') as f:
            generator = torch.load(f)
            
        with open('./DPautoGAN/ae_eps_inf.dat', 'rb') as f:
            autoencoder = torch.load(f)
        decoder = autoencoder.get_decoder()

        z = torch.randn(len(X_encoded), self._gan_params['latent_dim'], device = self._gan_params['device'])
        X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()        

        return X_synthetic_encoded
        