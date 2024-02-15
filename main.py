import numpy as np
import pandas as pd
import config
import utils
import os
import subprocess
import preprocess
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, PATECTGAN, PATEGAN, DPGAN
from snsynth.preprocessors.data_transformer import BaseTransformer
from DPautoGAN.dp_autogan import DPAUTOGAN
from misgan.misgan import misgan
from impute import impute_data
import aim 
from mbi import Dataset, Domain
import itertools
import json

params = config.params
params['epsilon_impute'] = 1-params['epsilon']
params['orig_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}.csv'
params['domain_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}.domain'
params['aim_domain_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}-domain.json'
if os.path.isfile(params['domain_path']) == False:
    utils.create_domain(params['orig_path'])

if os.path.isfile(params['aim_domain_path']) == False:
    utils.create_aim_domain(params['orig_path'])

if params['missing_p']:
    params['data_loc'] = utils.create_missing(params)
else:
    params['data_loc'] = params['orig_path']

if params['impute']:
    params = impute_data(params)

encoding1, encoding2, encoding3, all_cat_attrs = preprocess.preprocess(params, preprocess=True) #Two types of encoding. second for pategan
df, encoders = encoding1 #default encoding is 1; change for pategan
all_cols = df.columns

params['data_len'] = len(df)

baselines = {'DPCTGAN': DPCTGAN(), 'PATECTGAN': PATECTGAN(regularization='dragan'),
             'DPautoGAN': DPAUTOGAN(params=params), 'misgan': misgan(params=params)}

if params['missing_p']:
    params['results_dir'] = config.results_dir+'_'+params['missing_type']+ \
                            '_'+str(params['missing_p'])+'_'+params['dataset']
else:
    params['results_dir'] = config.results_dir+'_'+'NoMissing'+'_'+params['dataset']
    
if not os.path.exists(params['results_dir']):
    os.mkdir(params['results_dir'])

log_file = open(params['results_dir']+"/Parameter_details","w")
for key, value in params.items():
    print(key, ":", value, file=log_file)

if 'PrivBayes' in params['baselines']:
    command = [f'./PrivBayes/privbayes_new', f'{params["dataset"]}', f'{params["epsilon"]}',\
                    f'{params["runs"]}', f'{params["data_loc"]}', f'{params["domain_path"]}', \
                    f'{params["results_dir"]}/', '0', '4']

    subprocess.run(command) 

if 'PrivBayes2' in params['baselines']:
    command = [f'./PrivBayes/privbayes_new', f'{params["dataset"]}', f'{params["epsilon"]}',\
                    f'{params["runs"]}', f'{params["data_loc"]}', f'{params["domain_path"]}', \
                    f'{params["results_dir"]}/', '1', '4']
    print(' '.join(command))
    subprocess.run(command)

all_num_attrs = list(df.select_dtypes(include=np.number).columns)

attr_to_remove = list()
for col in all_num_attrs: #Make low range numerical to categorical
    if df[col].dtype == np.int64 and (df[col].max() - df[col].min()) < 50:
        attr_to_remove.append(col)
all_num_attrs = list(set(all_num_attrs) - set(attr_to_remove))
all_cols = list(df.columns)
all_cat_attrs = list(set(all_cols) - set(all_num_attrs))

for r in range(params['runs']):
    for b in params['baselines']:
            if b in baselines:

                if b.startswith('AIM'):
                    df = encoding3[0]
                    if b == 'AIM':
                        df = encoding3[0].dropna()

                    domain_config = json.load(open(params['aim_domain_path']))
                    aim_domain = Domain(domain_config.keys(), domain_config.values())
                    data = Dataset(df, aim_domain)
                    workload = list(itertools.combinations(data.domain, 2))
                    workload = [cl for cl in workload if data.domain.size(cl) <= 10000]
                    
                    workload = [(cl, 1.0) for cl in workload]
                    n_row = len(df)
                    n_len = len(str(n_row)) + 1
                    mech = aim.AIM(params['epsilon'], float(f'1e-{n_len}') , max_model_size=80)
                    synth = mech.run(data, workload)
                    syn_data = preprocess.inverse(synth.df, encoding3[1], all_cols)
                    syn_data.to_csv(params['results_dir']+f'/{b}_run{r}.syn', index=False)

                if b == 'PATEGAN' or b == 'DPautoGAN' or b == 'misgan':
                    df, encoders = encoding2
                else:
                    df, encoders = encoding1

                if b == 'DPautoGAN' or b == 'misgan':
                    syn_data = baselines[b].generate(df)
                
                else:                    
                    synth = PytorchDPSynthesizer(params['epsilon'], baselines[b])
                    # synth.fit(df, categorical_columns=pd.Index(set(df.columns)-set(all_num_attrs)))
                    synth.fit(df, categorical_columns=all_cat_attrs, transformer=BaseTransformer)
                    syn_data = synth.sample(params['data_len'])

                syn_data = preprocess.inverse(syn_data, encoders, all_cols)
                syn_path = f'{params["results_dir"]}/{b}_run{r}.syn'
                syn_data.to_csv(syn_path, index=None)

#Evaluations
for r in range(params['runs']):
    for b in params['baselines']:
        syn_path = f'{params["results_dir"]}/{b}_run{r}.syn'
        utils.evaluate(params['orig_path'], syn_path)
        

