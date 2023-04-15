import numpy as np
import pandas as pd
import config
import utils
import os
import subprocess
import preprocess

params = {
    'orig_data_loc': './datasets/Original',
    'dataset': 'bank',
    'epsilon': 1,
    'runs': 3,
    'missing_p': 0.2, #Percentage of missing values
    'missing_type': 'MNAR',
    'baselines' : ['PrivBayes', 'PrivBayes2'], #missing
    # 'baselines' : ['PATEGAN', 'DPCTGAN', 'PATECTGAN', 'PrivBayes',], #nomissing
    'bin_size' : 10 #Number of bins for continuous attributes
}

params['orig_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}.csv'

folders = [
# 'runs/runs/2022-03-17-01:39:33_MNAR_0.1_adult',
# 'runs/runs/2022-03-17-01:42:07_MNAR_0.2_adult',
'runs/MNAR_0.2_bank',
]

for f in folders:
    for r in range(params['runs']):
        for b in params['baselines']:
            if os.path.exists(f'{f}/{b}_run{r}.log'):
                continue
            else:
                syn_path = f'{f}/{b}_run{r}.syn'
                print(syn_path)
                utils.evaluate(params['orig_path'], syn_path)

