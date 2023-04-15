import numpy as np
import pandas as pd
import os.path
import evaluation
import missing_mechanisms
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, MinMaxScaler, OneHotEncoder
import preprocess

def create_missing(params):

    data = pd.read_csv(f'{params["orig_data_loc"]}/{params["dataset"]}.csv')
    data = data.dropna()
    cols = data.columns
    p_obs = 0.5 #Percentage of attributes used to train logistic model to get MAR
    q_percentile = 0.2 #quantile for MNAR quantile 
    file_dir = f'./datasets/{params["missing_type"]}'
    data2 = data.copy()
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    file_name = file_dir + f'/{params["dataset"]}_{params["missing_p"]}.csv'

    if os.path.exists(file_name):
            return file_name

    for col in data2.columns:  #Converting int datatypes to nullable type
        if data2[col].dtype == 'int':
            data2[col] = data2[col].astype(pd.Int32Dtype())

    if params['missing_type'] == 'MCAR':
        
        mask = np.random.choice([False, True], size=data.shape, p=[params['missing_p'],\
                                 1-params['missing_p']])

        data2 = data2.where(mask)
        
    if params['missing_type'] == 'MAR':

        encoded_data, encoders = _preprocess(data)
        mask = missing_mechanisms.MAR_mask(encoded_data.to_numpy(), params['missing_p'], p_obs)
        data2 = data2.where(~mask)
        
    if params['missing_type'] == 'MNAR':

        encoded_data, encoders = _preprocess(data)
        mask = missing_mechanisms.MNAR_mask_logistic(encoded_data.to_numpy(),\
                                                         params['missing_p'])
        data2 = data2.where(~mask)

    if params['missing_type'] == 'MNARQ':

        encoded_data, encoders = _preprocess(data)
        mask = missing_mechanisms.MNAR_mask_quantiles(encoded_data.to_numpy(),\
                                                         params['missing_p'], q_percentile, p_obs)
        data2 = data2.where(~mask)

    data2.to_csv(file_name, index=None)
    return file_name

def _preprocess(df):
    
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    encoders = []

    for col in df.columns:
        attr = np.array(df[col]).reshape(-1,1)

        if col in all_num_attrs:
            enc = MinMaxScaler()
        else:
            enc = OrdinalEncoder()

        if enc:
            enc.fit(attr)
            attr = enc.transform(attr).astype(np.double)
            df[col] = attr
        encoders.append(enc)

    return (df, encoders)

def evaluate(true_path, syn_path):
    whole_log = ''
    whole_log += evaluation.validate_kway_marginal(1, true_path, syn_path)
    whole_log += evaluation.validate_kway_marginal(2, true_path, syn_path)

    whole_log += evaluation.validate_accuracy(true_path, syn_path)

    log_file = open(syn_path[:-4] + '.log',"w")
    print(whole_log, file=log_file)


def create_domain(data_path):
    df = pd.read_csv(data_path)
    domain_path = data_path[:-4] + '.domain'


    domain_file = open(domain_path,"w")
    cols = df.columns
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    attr_to_remove = list()
    for col in all_num_attrs: #Make low range numerical to categorical
        if pd.api.types.is_integer_dtype(df[col]) and df[col].max() - df[col].min() < 50:
            attr_to_remove.append(col)
    all_num_attrs = list(set(all_num_attrs) - set(attr_to_remove))
    for col in cols:
        if col in all_num_attrs:
            print(f'C {df[col].min()} {df[col].max()}', file = domain_file)
        else:
            domain = df[col].unique()
            print('D ', end='', file = domain_file)
            for d in domain:
                print(f'{d} ',  end='', file = domain_file)
            print(file = domain_file)

