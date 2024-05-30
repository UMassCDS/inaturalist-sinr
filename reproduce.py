import os
import numpy as np
import torch
import shutil

import train
import eval

train_params = {}

"""
save_base
- Name of the directory to save models
"""
train_params["save_base"] = "./reproduce/"

"""
experiment_name
- Name of the directory where results for this run are saved.
"""
train_params["experiment_name"] = (
    "repr"  # This will be the name of the directory where results for this run are saved.
)


"""
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
"""
train_params["species_set"] = "all"

"""
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
"""
train_params["hard_cap_num_per_class"] = 1000  # will be overriden

"""
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
"""
train_params["num_aux_species"] = 0

"""
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
"""
train_params["input_enc"] = "sin_cos"

"""
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me'
"""
train_params["loss"] = "an_full"

# train:
# train.launch_training_run(train_params)

# load pretrain:
pre_trained_models = [
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt",
        "cap": 10,
    },
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt",
        "cap": 100,
    },
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt",
        "cap": 1000,
    },
]

pretrain_path = "./pre_trained_models/"

destination_directory = os.path.join(
    train_params["save_base"], train_params["experiment_name"]
)
os.makedirs(destination_directory, exist_ok=True)
for model in pre_trained_models:
    source_path = os.path.join(pretrain_path, model["path"])
    destination_path = os.path.join(
        destination_directory, os.path.basename(source_path)
    )
    shutil.copy2(source_path, destination_path)

print("Model files successfully transferred.")

# evaluate:
for model in pre_trained_models:
    cap = model["cap"]  # FIX no need, delete from DS
    model_name = model["path"]  # FIX this reference
    for eval_type in ["snt", "iucn", "geo_prior", "geo_feature"]:
        eval_params = {}
        eval_params["exp_base"] = train_params["save_base"]
        eval_params["experiment_name"] = train_params["experiment_name"]
        eval_params["eval_type"] = eval_type
        eval_params["ckp_name"] = model_name
        if eval_type == "iucn":
            eval_params["device"] = torch.device("cpu")  # for memory reasons
        cur_results = eval.launch_eval_run(eval_params)
        np.save(
            os.path.join(
                eval_params["exp_base"],
                train_params["experiment_name"],
                f"results_{eval_type}.npy",
            ),
            cur_results,
        )


"""
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
"""
