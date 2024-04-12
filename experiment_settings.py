"""Experimental settings

Functions
---------
get_best_models(exp_name, num_models=1, print_out=False, track_loss = "val")
get_model_losses(exp_name, track_loss)
get_settings(experiment_name)
"""
import numpy as np
import base_directories
import pickle
dir_settings = base_directories.get_directories()

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2024"

def get_best_models(exp_name, num_models=1, print_out=False, track_loss = "val"):
    dir_settings = base_directories.get_directories()
    with open(dir_settings["tuner_directory"] + exp_name + ".p", 'rb') as fp:
        tuner_results = pickle.load(fp)
    
    # Go through all tuning models
    model_num = []
    all_loss = []
    for imod in list(tuner_results.keys()):
        all_loss.append(tuner_results[imod]['results'][track_loss + '_loss'])
    top_models = np.argsort(all_loss)[:num_models]

    # Create dictionary with top models
    top_models_dict = dict()
    for itop in top_models:
        top_models_dict[itop] = tuner_results[itop]

    if print_out:
        print(top_models_dict)

    return top_models_dict

def get_model_losses(exp_name, track_loss):
    dir_settings = base_directories.get_directories()

    with open(dir_settings["tuner_directory"] + exp_name + ".p", 'rb') as fp:
        tuner_results = pickle.load(fp)

    all_loss = []
    for imod in list(tuner_results.keys()):
        all_loss.append(tuner_results[imod]['results'][track_loss + '_loss'])

    return all_loss

### TYPICAL TUNING SPECS
TUNE_LAYS = [[],[1], [2], [5], [10], [20], [50], [100],
            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]]

# Contains a dictionary will all the simulations

def get_settings(experiment_name):
    experiments = {
        # ---------------------- MAIN SIMULATIONS ---------------------------

        "chosen10": {
            "save_model": True,
            "n_models": 1,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,),
            "hiddens3": (5,5,5,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .1, 
            "ridge_param3": 0, 
            "learning_rate": .0001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 8319,
            "init_seed": 8,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen5": {
            "save_model": True,
            "n_models": 1,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01, 
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 3770,
            "init_seed": 7,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen1": {
            "save_model": True,
            "n_models": 1,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01,
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 3770,
            "init_seed": 6,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },


        "chosen10_245": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "245",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (15, 8, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,),
            "hiddens3": (5,5,5,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .1, 
            "ridge_param3": 0, 
            "learning_rate": .0001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen5_245": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "245",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (15, 8, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01, 
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen1_245": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "245",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (15, 8, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01,
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen10_370": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,),
            "hiddens3": (5,5,5,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .1, 
            "ridge_param3": 0, 
            "learning_rate": .0001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen5_370": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01, 
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "chosen1_370": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": (100,100,100,),
            "hiddens3": (50,50,),
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": .01,
            "ridge_param3": 0, 
            "learning_rate": .001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": 'relu',
            "act_fun3": 'relu',
            "n_epochs": 25_000,
            "patience": 50,
            },

        "top10": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": None,
            "hiddens3": None,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": None, 
            "ridge_param3": 0, 
            "learning_rate": None,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": None,
            "act_fun3": None,
            "n_epochs": 25_000,
            "patience": 50,
            },

        "top5": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": None,
            "hiddens3": None,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": None, 
            "ridge_param3": 0, 
            "learning_rate": None,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": None,
            "act_fun3": None,
            "n_epochs": 25_000,
            "patience": 50,
            },

        "top1": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": None,
            "hiddens3": None,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": None, 
            "ridge_param3": 0, 
            "learning_rate": None,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun1": 'gelu',
            "act_fun2": None,
            "act_fun3": None,
            "n_epochs": 25_000,
            "patience": 50,
            },

        # Keep this one for some Supp figures
        "oneyearmeans": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 5, 5),
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1850, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ["timemean=10_spatialmean=global", "timemean=1",],

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens": [25, 25],
            "dropout_rate": 0.,
            "ridge_param": [.1, 0],
            "learning_rate": 0.001,  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":0.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": None,
            "act_fun": ["elu", "elu"],
            "n_epochs": 25_000,
            "patience": 50,
            },

        # ---------------------- TUNE SIMULATIONS ---------------------------

        "basetune10": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": TUNE_LAYS,
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":[0.1, 0.01, 0.001],
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun2": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun3": ["elu", "relu", 'tanh', 'gelu'],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "basetune5": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980,2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980,2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": TUNE_LAYS,
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":[0.1, 0.01, 0.001],
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun2": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun3": ["elu", "relu", 'tanh', 'gelu'],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "basetune1": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "ERA5",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980,2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980,2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": TUNE_LAYS,
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":[0.1, 0.01, 0.001],
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun2": ["elu", "relu", 'tanh', 'gelu'],
            "act_fun3": ["elu", "relu", 'tanh', 'gelu'],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "basetune10_singlearch": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": list(range(1000)),
            "act_fun1": "gelu",
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "basetune5_singlearch": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": list(range(1000)),
            "act_fun1": "gelu",
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "basetune1_singlearch": {
            "save_model": True,
            "n_models": 30,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0),
            "no_test":True,
            "baseline_yr_bounds": (1980, 2010),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1980, 2010),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": list(range(1000)),
            "act_fun1": "gelu",
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "tune10": {
            "save_model": True,
            "n_models": 100,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 10,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": 'gelu',
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "tune5": {
            "save_model": True,
            "n_models": 100,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 5,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate":.01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": 'gelu',
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        "tune1": {
            "save_model": True,
            "n_models": 100,  # the number of networks to train
            "ssp": "370",  # [options: '126' or '370']
            "ensemble_data" : False, # True or False to use ensemble members
            "gcmsub": "ALL",  # [options: 'ALL' or 'UNIFORM'
            "obsdata": "BEST",  # [options: 'BEST' or 'GISTEMP'
            "smooth": False,
            "model_type" : "static_window",
            "len_window": 1,
            "n_train_val_test": (20, 10, 0,),
            "no_test": True,
            "baseline_yr_bounds": (1850, 1899),
            "training_yr_bounds": (1951, 2080),
            "anomaly_yr_bounds": (1951, 1980),
            "remove_sh": False,
            "anomalies": True,  # [options: True or False]
            "remove_map_mean": False,  # [options: False or "weighted" or "raw"]

            "input_fields" : ("timemean=10_spatialmean=global", "timemean=1",),

            "network_type": 'shash2',  # [options: "reg" or "shash2"]
            "hiddens1": (2,2,2,),
            "hiddens2": TUNE_LAYS,
            "hiddens3": TUNE_LAYS,
            "dropout_rate": 0.,
            "ridge_param1": 0, 
            "ridge_param2": [.1, .01, .001, 0], 
            "ridge_param3": 0, 
            "learning_rate": [0.01, 0.001, 0.0001],  # reg->0.0001, shash2->.00005 or .00001
            "baseline_learning_rate": .01,
            "batch_size": 64,
            "rng_seed": 0,
            "seed": 0,
            "act_fun1": 'gelu',
            "act_fun2": ["elu", "relu", 'tanh',],
            "act_fun3": ["elu", "relu", 'tanh',],
            "n_epochs": 25_000,
            "patience": 50,
            },

        }
    
    # Make adjustments to the experimental settings above. Set to False if no adjustments. Useful for tuning.
    adjust_settings = True

    if adjust_settings:
        tunelist = ["tune1","tune5","tune10",]
        for tunename in tunelist:
            for iseed in range(10):
                experiments[tunename + '_' + str(iseed)] = experiments[tunename].copy()
                experiments[tunename + '_' + str(iseed)]['seed'] = iseed

        
        testtoplist = [("top1","tune1"),("top5","tune5"),("top10","tune10"),]

        try:
            for keyname, tunename in testtoplist:

                for_mean = []
                for i in np.arange(5):
                    losses = get_model_losses(tunename + '_' + str(i), track_loss = 'test')
                    for_mean.append(losses)
                best_model_archs = np.argsort(np.mean(np.array(for_mean), axis=0))
                alldict = get_best_models(tunename + '_0', num_models=100, track_loss = "test")
                topdict = dict()
                topdict[best_model_archs[0]] = alldict[best_model_archs[0]].copy() # Create dictionary for best model
                topdict = topdict[list(topdict.keys())[0]].copy()

                # Add best specs from tuning to the experiments to repeat training + get more metrics
                experiments[keyname + '_best'] = experiments[keyname].copy()
                for koi in ["hiddens2", "hiddens3", "ridge_param2", "learning_rate", "act_fun2", "act_fun3",]:
                    if "hiddens" in koi:
                        res = topdict[koi].strip('][').split(', ')
                        if res == ['']:
                            experiments[keyname + '_best'][koi] = []
                        else:
                            experiments[keyname + '_best'][koi] = res
                    else:
                        experiments[keyname + '_best'][koi] = topdict[koi]
                experiments[keyname + '_best']['seed'] = None
                                
        except:
            print('no tune results to add to experiments in experiment_settings.py')

        for keyname in ["top1_best", "top5_best", "top10_best"]:
            for init_seed in np.arange(10):
                experiments[keyname + "_" + str(init_seed)] = experiments[keyname].copy()
                experiments[keyname + "_" + str(init_seed)]["init_seed"] = int(init_seed)
                experiments[keyname + "_" + str(init_seed)]["n_models"] = 10

        exp_dict = experiments[experiment_name]
        exp_dict['exp_name'] = experiment_name

    return exp_dict
