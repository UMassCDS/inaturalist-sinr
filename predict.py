import os
import numpy as np
import torch

import eval

train_params = {}
train_params['experiment_name'] = 'pre_trained_models' # This will be the name of the directory where results for this run are saved.

#evaluate model
def evaluate(model_type):
    for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
        #model parameters
        eval_params = {}
        eval_params['exp_base'] = '../../'
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