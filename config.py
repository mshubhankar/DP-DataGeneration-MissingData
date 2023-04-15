from datetime import datetime
results_dir = 'runs/'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
params = {
	'orig_data_loc': './datasets/Original',
	'dataset': 'adult',
	'epsilon': 3,
	'runs': 1,
	'missing_p': 0.2, #Percentage of missing values
	'missing_type': 'MCAR',
	'baselines' : ['misgan', 'DPautoGAN', 'PrivBayes2'], #PrivBayesE is called PrivBayes2
	'bin_size' : 10 #Number of bins for continuous attributes
}