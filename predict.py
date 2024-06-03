import os
import numpy as np
import torch

import eval

train_params = {}

train_params['experiment_name'] = 'demo' # This will be the name of the directory where results for this run are saved.

'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
'''
train_params['species_set'] = 'all'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = 1000

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
'''
train_params['input_enc'] = 'sin_cos'

'''
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me'
'''
train_params['loss'] = 'an_full'

#evaluate model
def evaluate(model_type):
    for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
        #model parameters
        eval_params = {}
        eval_params['exp_base'] = './experiments'
        eval_params['experiment_name'] = train_params['experiment_name']
        eval_params['eval_type'] = eval_type
        eval_params['ckp_name'] = model_type
        if eval_type == 'iucn':
            eval_params['device'] = torch.device('cpu') # for memory reasons

        #run evaluation
        cur_results = eval.launch_eval_run(eval_params)

        #save evaulation results
        np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

def main():
    model_type = 'model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt' 
    evaluate(model_type)

if __name__=="__main__": 
    main()