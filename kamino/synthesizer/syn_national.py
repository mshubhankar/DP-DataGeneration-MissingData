import logging
import time

import pandas as pd
import numpy as np

from pyvacy.pyvacy import analysis
# from synthesizer.data_process import preproc_national, postproc_national
from synthesizer.util import _analyze_privacy, evaluate_data, copy_log
from synthesizer.kamino import syn_data


def syn_national():
    """
    Group sex and income in the preprocessing step. Restore back from HoloMake process.
    """
    start = time.time()

    orig_data = f'./testdata/national/national.csv'
    path_data = f'./testdata/national/MAR/national_missing_0.35.csv'
    path_ic = f'./testdata/national/national.ic'

    # path_data_preproc = preproc_national(path_data)
    path_data_preproc = path_data
    n_row, n_col = pd.read_csv(path_data_preproc).shape
    n_len = len(str(n_row)) + 1

    # ## eps=0.1
    # paras = {
    #     'reuse_embedding': True,  # set True to reuse the embedding
    #     'dp': True,  # set True to enable privacy
    #     'n_row': n_row,  # number of rows in the true dataset
    #     'n_col': n_col,  # number of columns in the true dataset
    #     'epsilon1': .01,  #
    #     'l2_norm_clip': 1.0,
    #     'noise_multiplier': 2.9,  # =1.1 for eps=1, 1.6 for eps=0.5,  for eps=0.2
    #     'minibatch_size': 6,  # batch size to sample for each iteration, default 32, = for eps=1.5, = for eps=0.2
    #     'microbatch_size': 1,  # micro batch size
    #     'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
    #     'learning_rate': 1e-4,
    #     'iterations': 1000  # =1600 for eps=1, 1300 for eps=0.5
    # }

    # ## eps=0.2
    # paras = {
    #     'reuse_embedding': True,  # set True to reuse the embedding
    #     'dp': True,  # set True to enable privacy
    #     'n_row': n_row,  # number of rows in the true dataset
    #     'n_col': n_col,  # number of columns in the true dataset
    #     'epsilon1': .08,  #
    #     'l2_norm_clip': 1.0,
    #     'noise_multiplier': 2.2,  # =1.1 for eps=1, 1.6 for eps=0.5,  for eps=0.2
    #     'minibatch_size': 10,  # batch size to sample for each iteration, default 32, = for eps=1.5, = for eps=0.2
    #     'microbatch_size': 1,  # micro batch size
    #     'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
    #     'learning_rate': 1e-4,
    #     'iterations': 1000  # =1600 for eps=1, 1300 for eps=0.5
    # }

    # ## eps=0.4
    # paras = {
    #     'reuse_embedding': True,  # set True to reuse the embedding
    #     'dp': True,  # set True to enable privacy
    #     'n_row': n_row,  # number of rows in the true dataset
    #     'n_col': n_col,  # number of columns in the true dataset
    #     'epsilon1': .1,  #
    #     'l2_norm_clip': 1.0,
    #     'noise_multiplier': 1.6,  # =1.1 for eps=1, 1.6 for eps=0.5,  for eps=0.2
    #     'minibatch_size': 15,  # batch size to sample for each iteration, default 32, = for eps=1.5, = for eps=0.2
    #     'microbatch_size': 1,  # micro batch size
    #     'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
    #     'learning_rate': 1e-4,
    #     'iterations': 1000  # =1600 for eps=1, 1300 for eps=0.5
    # }

    # ## eps=0.8
    # paras = {
    #     'reuse_embedding': True,  # set True to reuse the embedding
    #     'dp': True,  # set True to enable privacy
    #     'n_row': n_row,  # number of rows in the true dataset
    #     'n_col': n_col,  # number of columns in the true dataset
    #     'epsilon1': .1,  #
    #     'l2_norm_clip': 1.0,
    #     'noise_multiplier': 1.2,  # =1.1 for eps=1, 1.6 for eps=0.5,  for eps=0.2
    #     'minibatch_size': 20,  # batch size to sample for each iteration, default 32, = for eps=1.5, = for eps=0.2
    #     'microbatch_size': 1,  # micro batch size
    #     'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
    #     'learning_rate': 1e-4,
    #     'iterations': 1600  # =1600 for eps=1, 1300 for eps=0.5
    # }

    ### eps=1
    paras = {
        'reuse_embedding': True,  # set True to reuse the embedding
        'dp': True,  # set True to enable privacy
        'total_epsilon' : 3,
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col,  # number of columns in the true dataset
        'epsilon1': .1,  #
        'l2_norm_clip': 1.0,
        'noise_multiplier': None,  #1.1-no missing 2.6-10% 15-20%  
        'minibatch_size': 15,  # batch size to sample for each iteration
        'microbatch_size': 1,  # micro batch size
        'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
        'learning_rate': 1e-4,
        'impute' : True,
        'complete_intermediate' : False,
        'iterations': 1600  # =1600 for eps=1
        # 'iterations': 1  # testing
    }

    # ### eps=1.6
    # paras = {
    #     'reuse_embedding': True,  # set True to reuse the embedding
    #     'dp': True,  # set True to enable privacy
    #     'n_row': n_row,  # number of rows in the true dataset
    #     'n_col': n_col,  # number of columns in the true dataset
    #     'epsilon1': .4,  #
    #     'l2_norm_clip': 1.0,
    #     'noise_multiplier': 1.1,  # =1.1 for eps=1, 1.6 for eps=0.5,  for eps=0.2
    #     'minibatch_size': 50,  # batch size to sample for each iteration, default 32, = for eps=1.5, = for eps=0.2
    #     'microbatch_size': 1,  # micro batch size
    #     'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
    #     'learning_rate': 1e-4,
    #     'iterations': 1800  # =1600 for eps=1, 1300 for eps=0.5
    # }

    if paras['dp']:

        if paras['noise_multiplier'] == None:
            while(True):
                paras['noise_multiplier'] = analysis.noise_mult(N=n_row,
                                         batch_size=paras['minibatch_size'],
                                         target_eps=paras['total_epsilon']-paras['epsilon1'],
                                         iterations=paras['iterations'] * (paras['n_col'] - 1),
                                         delta=paras['delta'])
                if paras['noise_multiplier']>20:
                    paras['iterations'] = paras['iterations']//2
                    paras['minibatch_size'] = paras['minibatch_size']//2
                else:
                    break
        epsilon2 = _analyze_privacy(paras)
        # epsilon2 = 0.2
        paras['epsilon2'] = epsilon2

        gaussian_std = []
        sensitivity = 2
        std1 = np.sqrt(sensitivity * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(1):
            gaussian_std.append(std1)

        epsilon = analysis.epsilon(N=n_row,
                                   batch_size=paras['minibatch_size'],
                                   noise_multiplier=paras['noise_multiplier'],
                                   iterations=paras['iterations'] * (paras['n_col'] - 1),
                                   delta=paras['delta'], gaussian_std=gaussian_std)
        # epsilon = 0.2
        msg = f"epsilon1= {paras['epsilon1']}\t epsilon2= {'{:.2f}'.format(epsilon2)}\t delta= {paras['delta']}\n" \
              f"epsilon= {epsilon}"
        print(msg)

    syn_data(path_data_preproc, path_ic, paras)
    path_syn = paras['path_syn']

    # path_data_postproc = postproc_national(path_syn)
    path_data_postproc = path_syn
    end = time.time()
    logging.info(f'TIME_WALL= {end - start}')

    evaluate_data(orig_data, path_data_postproc, path_ic)
    copy_log(paras)


if __name__ == '__main__':
    """
    The entry point for using kamino to generate synthetic dataset
    """

    syn_national()

