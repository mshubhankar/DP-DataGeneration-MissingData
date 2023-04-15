import numpy as np
import torch
from torch import optim
from torch import nn
import torch.utils.data
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
import warnings

import opacus

from snsynth.preprocessors.data_transformer import BaseTransformer
from .data_sampler import DataSampler
from ctgan.synthesizers import CTGANSynthesizer
from sklearn.mixture import GaussianMixture

def nr(mu, sigma):
    non_zero = torch.not_equal(sigma, 0.)
    new_sigma = torch.where(non_zero, sigma, 1e-20)
    sqrt_sigma = torch.sqrt(new_sigma)

    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(-torch.square(w), 2)), (2*torch.pi)**0.5) + \
                     torch.mul(torch.div(w,2.), 1 + torch.erf(torch.div(w, 2**0.5))))
    nr_values = torch.where(non_zero, nr_values, (mu + torch.abs(mu)) / 2.)
    
    return nr_values

class DiscriminatorForMissing(Module):
    def __init__(self, input_dim, discriminator_dim, loss,  means, weights, cov, n_distribution, pac=1):
        super(DiscriminatorForMissing, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pac
        #  print ('now dim is {}'.format(dim))
        self.pac = pac
        self.pacdim = dim
        self.means = means
        self.weights = weights
        self.covs = torch.abs(cov)
        self.n_distribution = n_distribution
        self.gamma = torch.empty(1).normal_(mean=1, std=1)

        seq = []
        self.first_layer = Linear(dim, discriminator_dim[0])
        self.first_relu = ReLU()
        self.first_dropout = Dropout(0.5)
        dim = discriminator_dim[0]
        for item in list(discriminator_dim[1:]):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == "cross_entropy":
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=1, lambda_=10
    ):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        
        if torch.any(input.isnan()):

            where_isnan = torch.isnan(input)
            p_ = nn.functional.softmax(self.weights, dim=0)
            weights2 = torch.square(self.first_layer.weight)
            final_distributions = list()
            final_q = list()
            
            for j in range(self.n_distribution):
                data_miss = torch.where(where_isnan, self.means[j], input)
                miss_cov = torch.where(where_isnan, self.covs[j], torch.zeros(input.shape))
                layer_1_m = self.first_relu(self.first_layer(data_miss))
                layer_1_m = nr(layer_1_m, torch.matmul(miss_cov, weights2.T))
                
                norm = torch.subtract(data_miss, self.means[j])
                norm = torch.square(norm)
                q = torch.where(~where_isnan, torch.add(self.gamma, self.covs[j]), torch.ones(input.shape))
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

            return self.seq(self.first_dropout(layer_1_miss))
        else:
            return self.seq(self.first_dropout(self.first_relu(self.first_layer(input.view(-1, self.pacdim)))))

class Discriminator(Module):
    def __init__(self, input_dim, discriminator_dim, loss, pac=1):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pac
        #  print ('now dim is {}'.format(dim))
        self.pac = pac
        self.pacdim = dim

        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == "cross_entropy":
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=1, lambda_=10
    ):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


# custom for calcuate grad_sample for multiple loss.backward()
def _custom_create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or accumulate it
    if the 'grad_sample' attribute already exists.
    This custom code will not work when using optimizer.virtual_step()
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample = param.grad_sample + grad_sample
        # param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample


class DPCTGAN(CTGANSynthesizer):
    def __init__(
        self,
        embedding_dim=32,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64),
        generator_lr=2e-3,
        generator_decay=1e-6,
        discriminator_lr=2e-3,
        discriminator_decay=1e-6,
        batch_size=120,
        discriminator_steps=1,
        log_frequency=False,
        verbose=True,
        epochs=20,
        pac=1,
        cuda=True,
        disabled_dp=False,
        delta=None,
        sigma=3,
        n_distribution=3,
        max_per_sample_grad_norm=1.0,
        epsilon=1,
        preprocessor_eps=0.00,
        loss="cross_entropy",
        category_epsilon_pct=0.3,
        enhanced=False,
    ):

        assert batch_size % 2 == 0
        self.enhanced = False
        self.n_distribution = n_distribution
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._category_epsilon_pct = category_epsilon_pct
        self.pac = pac

        # opacus parameters
        self.sigma = sigma
        self.disabled_dp = disabled_dp
        self.delta = delta
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.epsilon = epsilon - preprocessor_eps
        if self.epsilon < 0:
            raise ValueError("needs to be larger than preprocessor_eps!")
        self.preprocessor_eps = preprocessor_eps
        self.epsilon_list = []
        self.alpha_list = []
        self.loss_d_list = []
        self.loss_g_list = []
        self.verbose = verbose
        self.loss = loss

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        if self.loss != "cross_entropy":
            # Monkeypatches the _create_or_extend_grad_sample function when calling opacus
            opacus.grad_sample.utils.create_or_extend_grad_sample = (
                _custom_create_or_extend_grad_sample
            )

        if self._log_frequency:
            warnings.warn(
                "log_frequency is selected.  This may result in oversampling frequent "
                "categories, which could cause privacy leaks."
            )

    def train(
        self,
        data,
        categorical_columns=None,
        ordinal_columns=None,
        update_epsilon=None,
        transformer=BaseTransformer,
        continuous_columns_lower_upper={},
    ):
        if data.isnull().values.any():
            self.enhanced = True

        if update_epsilon:
            self.epsilon = update_epsilon - self.preprocessor_eps

        for col in categorical_columns:            
            if str(data[col].dtype).startswith("float"):
                raise ValueError(
                    "It looks like you are passing in a vector of continuous values"
                    f"to a categorical column at [{col}]."
                    "Please discretize and pass in categorical columns with"
                    "unsigned integer or string category names."
                )
        self._transformer = transformer(self.preprocessor_eps)
        self._transformer.fit(
            data,
            discrete_columns=categorical_columns,
            continuous_columns_lower_upper=continuous_columns_lower_upper,
        )
        # for tinfo in self._transformer._column_transform_info_list:
        #    if tinfo.column_type == "continuous":
        #        raise ValueError("We don't support continuous values on this synthesizer.  Please discretize values.")

        train_data = self._transformer.transform(data)

        sampler_eps = 0.0

        if categorical_columns and self._category_epsilon_pct:
            sampler_eps = self.epsilon * self._category_epsilon_pct
            per_col_sampler_eps = sampler_eps / len(categorical_columns)
            self.epsilon = self.epsilon - sampler_eps
        else:
            per_col_sampler_eps = None

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
            per_column_epsilon=per_col_sampler_eps,
        )

        spent = self._data_sampler.total_spent
        if spent > sampler_eps and not np.isclose(spent, sampler_eps):
            raise AssertionError(
                f"The data sampler used {spent} epsilon and was budgeted for {sampler_eps}"
            )

        if(self.enhanced == True):
            X_complete = train_data[~torch.any(torch.Tensor(train_data).isnan(),dim=1)]
            gmm = GaussianMixture(n_components=self.n_distribution, covariance_type='diag').fit(X_complete)
            gmm_weights = torch.Tensor(np.log(gmm.weights_.reshape((-1, 1))))
            gmm_means = torch.Tensor(gmm.means_)
            gmm_covariances = torch.Tensor(gmm.covariances_)
            del gmm


        data_dim = self._transformer.output_dimensions
        if self.enhanced:
            gmm_means = torch.cat([gmm_means, torch.zeros((3,self._data_sampler.dim_cond_vec()))], dim=1)
            gmm_covariances = torch.cat([gmm_covariances, torch.zeros((3,self._data_sampler.dim_cond_vec()))], dim=1)

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            self.pac,
        ).to(self._device)

        if(self.enhanced == True):
            discriminator = DiscriminatorForMissing(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            gmm_means,
            gmm_weights,
            gmm_covariances,
            self.n_distribution,
            self.pac,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        dataloader = torch.utils.data.DataLoader(train_data, \
            batch_size=self._batch_size) #Dummy loader (Only for Opacus)
        privacy_engine = opacus.PrivacyEngine()

        if not self.disabled_dp:
            discriminator, optimizerD, dataloader = privacy_engine.make_private(
                module=discriminator,
                optimizer=optimizerD,
                data_loader=dataloader,
                poisson_sampling=False,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm,
                clipping='flat'
            )

        one = torch.tensor(1, dtype=torch.float).to(self._device)
        mone = one * -1

        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

        assert self._batch_size % 2 == 0
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(self._epochs):
            if not self.disabled_dp:
                # if self.loss == 'cross_entropy':
                #    autograd_grad_sample.clear_backprops(discriminator)
                # else:
                for p in discriminator.parameters():
                    if hasattr(p, "grad_sample"):
                        del p.grad_sample

                if self.delta is None:
                    self.delta = 1 / (
                        train_data.shape[0] * np.sqrt(train_data.shape[0])
                    )

                epsilon = privacy_engine.get_epsilon(
                    self.delta
                )

                self.epsilon_list.append(epsilon)
                # self.alpha_list.append(best_alpha)
                if self.epsilon < epsilon:
                    if self._epochs == 1:
                        raise ValueError(
                            "Inputted epsilon and sigma parameters are too small to"
                            + " create a private dataset. Try increasing either parameter "
                            + "and rerunning."
                        )
                    else:
                        break

            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = self._data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real = self._data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm]
                    )
                    c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype("float32")).to(self._device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fakeact

                optimizerD.zero_grad()

                if self.loss == "cross_entropy":
                    y_fake = discriminator(fake_cat)

                    #   print ('y_fake is {}'.format(y_fake))
                    label_fake = torch.full(
                        (int(self._batch_size / self.pac),),
                        fake_label,
                        dtype=torch.float,
                        device=self._device,
                    )

                    #    print ('label_fake is {}'.format(label_fake))

                    error_d_fake = criterion(y_fake.squeeze(), label_fake)
                    error_d_fake.backward()
                    optimizerD.step()
                    optimizerD.zero_grad()

                    # train with real
                    label_true = torch.full(
                        (int(self._batch_size / self.pac),),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    y_real = discriminator(real_cat)
                    error_d_real = criterion(y_real.squeeze(), label_true)
                    error_d_real.backward()
                    optimizerD.step()

                    loss_d = error_d_real + error_d_fake

                else:

                    y_fake = discriminator(fake_cat)
                    mean_fake = torch.mean(y_fake)
                    mean_fake.backward(one)

                    y_real = discriminator(real_cat)
                    mean_real = torch.mean(y_real)
                    mean_real.backward(mone)

                    optimizerD.step()

                    loss_d = -(mean_real - mean_fake)

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                # pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                # pen.backward(retain_graph=True)
                # loss_d.backward()
                # optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # if condvec is None:
                cross_entropy = 0
                # else:
                #    cross_entropy = self._cond_loss(fake, c1, m1)

                if self.loss == "cross_entropy":
                    label_g = torch.full(
                        (int(self._batch_size / self.pac),),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    # label_g = torch.full(int(self.batch_size/self.pack,),1,device=self.device)
                    loss_g = criterion(y_fake.squeeze(), label_g)
                    loss_g = loss_g + cross_entropy
                else:
                    loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            self.loss_d_list.append(loss_d)
            self.loss_g_list.append(loss_g)
            if self.verbose:
                if self.disabled_dp:
                    epsilon = 'infinite'
                print(
                    "Epoch %d, Loss G: %.4f, Loss D: %.4f"
                    % (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                    flush=True,
                )
                print("epsilon is {e}".format(e=epsilon))

        return self.loss_d_list, self.loss_g_list, self.epsilon_list, self.alpha_list

    def generate(self, n, condition_column=None, condition_value=None):
        """
        TODO: Add condition_column support from CTGAN
        """
        self._generator.eval()

        # output_info = self._transformer.output_info
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._data_sampler.sample_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)
