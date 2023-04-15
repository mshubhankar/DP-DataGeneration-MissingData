import itertools
import config
import pandas as pd
import utils
import os 
from tqdm import tqdm
from multiprocessing import Process, Manager, Pool
import concurrent.futures
import threading, time


params = config.params
params['orig_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}.csv'
params['domain_path'] = f'{params["orig_data_loc"]}/{params["dataset"]}.domain'
if os.path.isfile(params['domain_path']) == False:
    utils.create_domain(params['orig_path'])


if params['missing_p']:
    params['data_loc'] = utils.create_missing(params)
else:
    params['data_loc'] = params['orig_path']

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def populate_attrs(params):   
    df = pd.read_csv(params['data_loc'])
    cols = df.columns
    attrs = {}

    for c in cols:
        attrs[c] = (len(df) - len(df[c].dropna()))/len(df)
    return attrs

def populate_margs(params):
    # marg_file = open('privbayes_marginal_output.txt')
    marg_file = open('PrivBayesE_marginals/Bank/bank_1_0.1.txt')
    df = pd.read_csv(params['data_loc'])
    cols = df.columns
    marginals = []
    for line in marg_file:
        line = line.strip()
        marg_attr = line.split(' ')
        marg = ''
        for attr in marg_attr:
            marg += ',' + cols[int(attr)]
        marginals.append(marg)

    return marginals


# attrs = {'A':0.2, 'B':0.2, 'C':0.2, 'D':0.2, 'E':0.2}
attrs = populate_attrs(params)

# marginals = ['A', 'B',  'BC', 'BCD']
marginals = populate_margs(params)

# print(attrs)
# print(marginals)

total_eps = 1

def find_cost(amp_cost, part, attrs):

    amplified_marg_cost = amp_cost.copy()
    for factor in part:
        amp_factor = ''.join(factor) #Possible amp factors
        amp_factor_cost = 1 #cost of amp_factor
        
        for x in attrs.keys(): #calculate the cost of each amp_factor
            if x in amp_factor:
                amp_factor_cost *= (1 - attrs[x]) #cost = \prod(all_attrs \in amp_factor)
        for marginal in amp_cost:
            if amp_factor in marginal and amplified_marg_cost[marginal]> amp_cost[marginal] * amp_factor_cost: #find best_ampfactor
                amplified_marg_cost[marginal] = amp_cost[marginal] * amp_factor_cost

    final_cost = sum(amplified_marg_cost.values())
    
    return final_cost


best_cost = total_eps
best_part = None

# futures = []
# with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
for part in tqdm(partition(list(attrs.keys()))): #Loop all partitions
    
    amp_cost = {marginal: (total_eps/len(marginals)) for marginal in marginals} #Dict to store cost
    cost = find_cost(amp_cost, part, attrs)
    if cost < best_cost:
        best_cost = cost
        best_part = part
        # futures.append(executor.submit(find_cost, amp_cost, part, attrs))

print(best_cost, best_part)
