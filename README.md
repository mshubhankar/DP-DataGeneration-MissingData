# Differentially Private Data Generation with Missing Data

This repository contains the codebase and the [full paper](full_paper.pdf) for the paper Differentially Private Data Generation with Missing Data.
All baselines except Kamino can be accessed using the ```config.py``` file. Kamino has a sub repository inside this codebase and need to run from the respective folder.

## Setup
We use python==3.7, torch>=1.7. The environment can be quickly built using conda with the command
```conda env create -f=environment.yml $ conda activate missing_syn```

## Generate Synthetic Datasets
We now describe how to run the different baselines from the paper on the Adult dataset.

**1.) Prepare PrivBayes:** 
First, build PrivBayes using ```cd PrivBayes && make clean && make```.
The script will build the C++ files for PrivBayes. Note that some of the build might need to be changed. 

**2.) Use config:** 

```config.py``` can be used to run different baselines. 

```python
params = {
    'orig_data_loc': './datasets/Original', #the location to find original datasets
    'dataset': 'adult', #ground truth dataset for synthetic data generation
    'epsilon': `, #Total privacy budget
    'runs': 1, #To repeat the experiment for multiple runs
    'missing_p': 0.2, #Percentage of missing values
    'missing_type': 'MCAR', #Missing mechanism to choose from [MCAR, MAR, MNAR]
    'baselines' : ['misgan', 'DPautoGAN', 'PrivBayes2'], #To choose from ['misgan', 'DPautoGAN', 'DPCTGAN', 'PrivBayes', 'PrivBayes2'] PrivBayesE is called PrivBayes2,
    'bin_size' : 10 #Number of bins for continuous attributes
    }
```

The missing datasets will automatically be created and stored in the respective folder in ```datasets/```. 

**3.) Generate Dataset:** 
Finally, we can generate a synthetic dataset by running ```python main.py```
This script will train the baselines from ```config.py``` and automatically evaluate the metrics by running ```evaluation.py```

## Amplified privacy

The amplified privacy cost to the ground truth data can be calculated for PrivBayesE. 
First, the marginals from PrivBayesE need to be extracted and put in the ```./PrivBayesE_marginals``` folder. Some examples are already included. 
The amplified cost can then be calculated by executed ```python opt_amp_cost.py``` and setting which marginal need to be optimized inside ```opt_amp_cost.py```.
