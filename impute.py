import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
import torch
from opacus import PrivacyEngine
import os

def kaminoimpute(params):
    kamino_eps = params['impute'].split('_')[1]
    params['data_loc'] = params['data_loc'][:-4] + '_kaminoimpute_eps_' + kamino_eps + '.csv'

def random_impute(params):
    df = pd.read_csv(params['data_loc'])
    #randomly impute missing cells of each attribute with values from the attribute domain
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(np.random.choice(df[col].dropna()))
    params['data_loc'] = params['data_loc'][:-4] + '_randomimpute.csv'
    df.to_csv(params['data_loc'], index=False)

def mean_impute(params):
    df = pd.read_csv(params['data_loc'])
    epsilon = float(params['impute'].split('_')[1])
    params['data_loc'] = params['data_loc'][:-4] + '_meanimpute' + f'_eps_{epsilon}' + '.csv'
    # if exists skip
    if os.path.exists(params['data_loc']):
        return

    all_cols = df.columns
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    attr_to_remove = list()
    for col in all_num_attrs: #Make low range numerical to categorical
        if pd.api.types.is_integer_dtype(df[col]) and df[col].max() - df[col].min() < 50:
            attr_to_remove.append(col)
    all_num_attrs = list(set(all_num_attrs) - set(attr_to_remove))
    all_cat_attrs = list(set(all_cols) - set(all_num_attrs))
    
    #impute missing cells of each attribute with mean of the attribute
    for col in df.columns:
        if col in all_num_attrs and df[col].isnull().sum() > 0:
            # laplace noise with eps = paras['epsilon_impute']
            # sensitivity is the max absolute value of the attribute
            sensitivity = df[col].abs().max()
            mean = df[col].mean() + np.random.laplace(0, sensitivity/epsilon)
            df[col] = df[col].fillna(mean)
        if col in all_cat_attrs and df[col].isnull().sum() > 0:
            # fill up with probability of occurance
            prob = df[col].value_counts()
            # add laplace noise to pandas series
            sensitivity = 2
            prob = prob + np.random.laplace(0, sensitivity/epsilon, len(prob))
            # make all negative values 0
            prob[prob < 0] = 0
            prob = prob/prob.sum()
           
            # fill up empty cells using prob
            df[col] = df[col].fillna(pd.Series(np.random.choice(prob.index, size=len(df[col]), p=prob)))
    
    df.to_csv(params['data_loc'], index=False)

def preprocess_for_impute(df):
    all_cols = df.columns
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    all_cat_attrs = list(set(all_cols) - set(all_num_attrs))
    encoders = []
    data = []
    for col in df.columns:
        attr = np.array(df[col].dropna()).reshape(-1,1)
        if col in all_cat_attrs:
            enc = OrdinalEncoder()
            attr = np.array(enc.fit_transform(attr))
                       
        
        elif col in all_num_attrs:
            enc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
            attr = np.array(enc.fit_transform(attr))

        
        attrwmissing = df[col].astype('str').to_numpy(na_value = np.nan) #convert to numpy with nan elsewhere
        attrwmissing = np.empty((attrwmissing.shape[0], attr.shape[1]))# create np array with nan
        attrwmissing[:] = np.nan
        mask = (~pd.isnull(df[col])).to_numpy()
        attrwmissing[np.where(mask)] = attr #fill it up with whats available

        encoders.append(enc)
        data.append(attrwmissing)

    df = pd.DataFrame(np.concatenate(data,axis=1), columns=all_cols)
    return df, encoders

def inverse(encoded_data, encoders, cols):

    inverse_data = []
    encoded_data = np.array(encoded_data)
    # np.clip(encoded_data, a_min=0, a_max=None, out=encoded_data)

    count = 0
    for encoder in encoders:
        if encoder == None: #Not encoded
            inverse_data.append(encoded_data[:,\
                                count:count+1].reshape(-1,1))
            count +=1
        elif hasattr(encoder, "categories_") or hasattr(encoder, "classes_"): #Categorical
            if(hasattr(encoder,"classes_")): #LabelBinarizer
                dim = len(encoder.classes_)
                if(dim == 2):
                    dim = 1
                encoded_data[:, count:count+dim]
                inverse_data.append(encoder.inverse_transform(encoded_data[:,\
                                    count:count+dim].reshape(-1,dim)).reshape(-1,1))
                count += dim
            else: #Ordinal

                # np.clip(encoded_data, a_min=0, a_max=len(encoder.categories_),\
                #              out=encoded_data) #Limit upper bound to total features                
                inverse_data.append(encoder.inverse_transform(encoded_data[:,\
                                     count:count+1].reshape(-1,1).round()))
                count += 1
        else: #Continuous
            inverse_data.append(encoder.inverse_transform(encoded_data[:,\
                                 count:count+1].reshape(-1,1)).clip(min=0))
            count += 1
    decoded = np.concatenate(inverse_data, axis=1)

    decoded = pd.DataFrame(decoded, columns=cols)

    return decoded


# torch linear regression with differential privacy
class dpLinearRegression(torch.nn.Module):
    def __init__(self, input, epsilon=0.1):
        super(dpLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input, 1)
        self.epsilon = epsilon

    def forward(self, x):
        return self.linear(x)

    def fit(self, x, y):
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        privacy_engine = PrivacyEngine()
        dataloader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=32)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        
        self, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module = self,
            optimizer = optimizer,
            data_loader = dataloader,
            target_epsilon = self.epsilon,
            target_delta = 1e-5,
            epochs = 5,
            max_grad_norm = 1.0
        )
        
        criterion = torch.nn.MSELoss()
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = self(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        outputs = self(x)
        return outputs.detach().numpy()
        

def ml_ols_impute(params):
    df = pd.read_csv(params['data_loc'])
    df, encoders = preprocess_for_impute(df)

    all_cols = df.columns
    # if first col has missing values, randomly impute
    if df[all_cols[0]].isnull().sum() > 0:
        df[all_cols[0]] = df[all_cols[0]].fillna(np.random.choice(df[all_cols[0]].dropna()))
    # if other cols have missing values, impute using linear regression
    
    for col_index in range(1, len(all_cols)):
        x_cols = all_cols[:col_index]
        y_col = all_cols[col_index]
        if df[y_col].isnull().sum() > 0:
            # get all rows that have no missing values in y_col
            df_no_missing = df.dropna(subset=[y_col])
            x = df_no_missing[x_cols]
            y = df_no_missing[y_col]
            if params['epsilon_impute'] > 0:
                reg = dpLinearRegression(epsilon=params['epsilon_impute']/(len(all_cols)-1), input=col_index).fit(x, y)
            else:
                reg = LinearRegression().fit(x, y)
            df[y_col] = df[y_col].fillna(pd.Series(reg.predict(df[x_cols])))
    
    df_imputed = inverse(df, encoders, all_cols)
    
    params['data_loc'] = params['data_loc'][:-4] + '_mlimpute.csv'
    df_imputed.to_csv(params['data_loc'], index=False)

def impute_data(params):
    if params['impute'] == 'random':
        random_impute(params)
    if 'meanimpute' in params['impute']:
        mean_impute(params)
    if params['impute'] == 'ml':
        ml_ols_impute(params)
    if 'kaminoimpute' in params['impute']:
        kaminoimpute(params)
    return params