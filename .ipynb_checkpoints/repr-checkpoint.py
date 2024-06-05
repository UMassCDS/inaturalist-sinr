import os
import numpy as np
import torch

import models
import eval

cap=10
model_path = f'../../pre_trained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_{cap}.pt' 

train_params = torch.load(model_path, map_location='cpu')

# evaluate:
# for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
for eval_type in ['geo_feature']:
    eval_params = {}
    eval_params['exp_base'] = ''
    eval_params['experiment_name'] = ''
    eval_params['eval_type'] = eval_type
    eval_params["ckp_name"] = model_path
    if eval_type == 'iucn':
        eval_params['device'] = torch.device('cpu') # for memory reasons
    cur_results = eval.launch_eval_run(eval_params)
    np.save(os.path.join(eval_params['exp_base'], eval_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

