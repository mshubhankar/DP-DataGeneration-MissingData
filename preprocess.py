import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, MinMaxScaler, OneHotEncoder, \
                                    LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
    def inverse_transform(self, x, y=0):
        return self.encoder.inverse_transform(x).reshape(-1,1)

def preprocess(params, preprocess):
    df = pd.read_csv(params['data_loc'])
    
    if params['missing_p'] == None:
        df = df.dropna()

    if not preprocess:
        return (df, None), (df, None)
    df2 = df.copy(deep=True)
    all_cols = df.columns
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    
    attr_to_remove = list()
    for col in all_num_attrs: #Make low range numerical to categorical
        if pd.api.types.is_integer_dtype(df[col]) and df[col].max() - df[col].min() < 50:
            attr_to_remove.append(col)
    all_num_attrs = list(set(all_num_attrs) - set(attr_to_remove))
    all_cat_attrs = list(set(all_cols) - set(all_num_attrs))

    encoders = []
    data = []
    for col in df.columns:
        attr = np.array(df[col].dropna()).reshape(-1,1)
        if col == 'capital-gain' or col == 'capital-loss':
            # enc = Pipeline([
            #                 ("discretize", KBinsDiscretizer(n_bins=params['bin_size'], encode='ordinal', strategy='uniform')),
            #                 ("binarize", MyLabelBinarizer())
            #                 ])
            enc = KBinsDiscretizer(n_bins=params['bin_size'], encode='ordinal', strategy='uniform')
            attr = np.array(enc.fit_transform(attr)) #get attribute after preprocess
            attrwmissing = df2[col].astype('str').to_numpy(na_value = np.nan) #convert to numpy with nan elsewhere
            attrwmissing = np.empty((attrwmissing.shape[0], attr.shape[1]))# create np array with nan
            attrwmissing[:] = np.nan
            mask = (~pd.isnull(df2[col])).to_numpy()
            attrwmissing[np.where(mask)] = attr #fill it up with whats available

        elif col in all_num_attrs:
            enc = MinMaxScaler()
            enc.fit(attr)
            attr = enc.transform(attr)
            attrwmissing = np.float32(df2[col].astype('float').to_numpy(na_value = np.nan))
            attrwmissing[np.where(~np.isnan(attrwmissing))] = attr.reshape(-1)
            attrwmissing = attrwmissing.reshape(-1,1)


        else:
            # enc = LabelBinarizer()
            # enc.fit(attr)
            # attr = np.array(enc.transform(attr)) #get attribute after preprocess
            # attrwmissing = df2[col].astype('str').to_numpy(na_value = np.nan) #convert to numpy with nan elsewhere
            # attrwmissing = np.empty((attrwmissing.shape[0], attr.shape[1]))# create np array with nan
            # attrwmissing[:] = np.nan
            # mask = (~pd.isnull(df2[col])).to_numpy()
            # attrwmissing[np.where(mask)] = attr #fill it up with whats available
            # enc = OrdinalEncoder()
            enc = None
            attrwmissing = np.array(df[col]).reshape(-1,1)

        data.append(attrwmissing)
        encoders.append(enc)
    df = pd.DataFrame(np.concatenate(data,axis=1), columns=all_cols)

    encoders2 = []
    data2 = []
    for col in df2.columns:
        attr = np.array(df2[col].dropna()).reshape(-1,1)

        if col in all_num_attrs:
            enc = MinMaxScaler()
            enc.fit(attr)
            attr = enc.transform(attr)
            attrwmissing = np.float32(df2[col].astype('float').to_numpy(na_value = np.nan))
            attrwmissing[np.where(~np.isnan(attrwmissing))] = attr.reshape(-1)
            attrwmissing = attrwmissing.reshape(-1,1)

        else:            
            # enc = OneHotEncoder()
            enc = LabelBinarizer()
            enc.fit(attr)
            attr = np.array(enc.transform(attr)) #get attribute after preprocess
            attrwmissing = df2[col].astype('str').to_numpy(na_value = np.nan) #convert to numpy with nan elsewhere
            attrwmissing = np.empty((attrwmissing.shape[0], attr.shape[1]))# create np array with nan
            attrwmissing[:] = np.nan
            mask = (~pd.isnull(df2[col])).to_numpy()
            attrwmissing[np.where(mask)] = attr #fill it up with whats available

        data2.append(attrwmissing)
        encoders2.append(enc)
    
    df2 = pd.DataFrame(np.concatenate(data2,axis=1))
    return ((df, encoders), (df2, encoders2), all_cat_attrs)

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