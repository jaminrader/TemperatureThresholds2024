"""Training script.

Trains a model given parameters for a simulation in experiment_settings.py
"""

import sys, imp, os, time

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import LearningRateScheduler
import pandas as pd

import experiment_settings
import file_methods, plots, custom_metrics, network, data_processing, base_directories

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2023"

### MAKE SOME DECISIONS
EXP_NAME_LIST = ["chosen1_245", "chosen5_245", "chosen10_245"]
RETRAIN_MODELS = True
OVERWRITE_MODEL = True

dir_settings = base_directories.get_directories()

MODEL_DIRECTORY = dir_settings['models']       
PREDICTIONS_DIRECTORY = dir_settings['predictions']   
DATA_DIRECTORY = dir_settings['data']   
DIAGNOSTICS_DIRECTORY = dir_settings['diagnostics']   
FIGURE_DIRECTORY = dir_settings['figure']   

base_directories.make_directories()

imp.reload(data_processing)
imp.reload(file_methods)
imp.reload(experiment_settings)
imp.reload(network)

make_plots = False
need_rng_seed = False

for EXP_NAME in EXP_NAME_LIST:

    if not RETRAIN_MODELS and EXP_NAME in os.listdir(MODEL_DIRECTORY):
        continue

    settings = experiment_settings.get_settings(EXP_NAME)
    print(settings)

    # make a directory for the time run
    start_time = time.localtime()
    time_dir = time.strftime("%Y-%m-%d_%H%M", start_time) + '/'
    
    # define early stopping callback (cannot be done elsewhere)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_RegressLossExpSigma',
                                                       patience=settings['patience'],
                                                       verbose=1,
                                                       mode='auto',
                                                       restore_best_weights=True)    
    
    def scheduler(epoch, learning_rate):
        # Set the learning rate to 0 for the first epoch, and then to the specified learning_rate
        return 0.0 if epoch < 1 else learning_rate
    learning_rate_scheduler = LearningRateScheduler(lambda epoch: scheduler(epoch, settings["learning_rate"]))

    if settings["seed"] is None:
        # define random number generator
        need_rng_seed=True
        rng = np.random.default_rng(settings["rng_seed"])

    for iloop in np.arange(settings['n_models']):
        if need_rng_seed:
            seed = rng.integers(low=1_000,high=10_000,size=1)[0]
            settings["seed"] = int(seed)
        tf.random.set_seed(settings["seed"])
        np.random.seed(settings["seed"])

        # get model name
        model_name = file_methods.get_model_name(settings)
        if os.path.exists(MODEL_DIRECTORY + model_name + "_model") and OVERWRITE_MODEL==False:
            print(model_name + 'exists. Skipping...')
            print("================================\n")
            continue   
            
        # load observations for diagnostics plotting and saving predictions
        da_obs, x_obs, global_mean_obs = data_processing.get_observations(DATA_DIRECTORY, settings)
        N_TRAIN, N_VAL, N_TEST, ALL_MEMBERS = data_processing.get_members(settings)            

        # get the data
        (x_train, 
         x_val, 
         x_test, 
         y_train, 
         y_val, 
         y_test, 
         onehot_train, 
         onehot_val, 
         onehot_test, 
         y_yrs_train, 
         y_yrs_val, 
         y_yrs_test, 
         target_years, 
         map_shape,
         settings) = data_processing.get_cmip_data(DATA_DIRECTORY, settings)
        
        #x_val = x_val[np.logical_and(y_val > .8, y_val < 2.2)]
    

        ## determine how many GCMs are being used for later re-shaping
        N_GCMS = len(file_methods.get_cmip_filenames(settings, verbose=0))

        #----------------------------------------        
        tf.keras.backend.clear_session()                

        try:

            full_model, baseline_model, delta_model = network.compile_full_model(x_train, y_train, settings)

            print('TRAINING BASELINE MODEL')
            network.set_trainable(delta_model, full_model, settings, False, learning_rate=settings["baseline_learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping],
                                verbose=0,                        
                                )
            
            # # Then train the delta model
            print('TRAINING DELTA MODEL')
            network.set_trainable(baseline_model, full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model, full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("delta_layers"), full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, settings, False, learning_rate=settings["learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping,learning_rate_scheduler],
                                verbose=0,                        
                                )
            
            network.set_trainable(baseline_model, full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model, full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("delta_layers"), full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, settings, True, learning_rate=settings["learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping,learning_rate_scheduler],
                                verbose=0,                        
                            )
        except: # Wrapped and repeated -- due to some strange miniforge issues that occaisionally kill training

            full_model, baseline_model, delta_model = network.compile_full_model(x_train, y_train, settings)

            print('TRAINING BASELINE MODEL')
            network.set_trainable(delta_model, full_model, settings, False, learning_rate=settings["baseline_learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping],
                                verbose=0,                        
                                )
            
            # # Then train the delta model
            print('TRAINING DELTA MODEL')
            network.set_trainable(baseline_model, full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model, full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("delta_layers"), full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, settings, False, learning_rate=settings["learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping,learning_rate_scheduler],
                                verbose=0,                        
                                )
            
            network.set_trainable(baseline_model, full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model, full_model, settings, True, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("delta_layers"), full_model, settings, False, learning_rate=settings["learning_rate"])
            network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, settings, True, learning_rate=settings["learning_rate"])

            history = full_model.fit(x_train, onehot_train, 
                                epochs=settings['n_epochs'], 
                                batch_size = settings['batch_size'], 
                                shuffle=True,
                                validation_data=[x_val, onehot_val],
                                callbacks=[early_stopping,learning_rate_scheduler],
                                verbose=0,                        
                            )
        
        #----------------------------------------
        # save the tensorflow model
        if settings["save_model"]:
            model_save_loc = MODEL_DIRECTORY + EXP_NAME + '/'
            model_save_dir = model_save_loc + time_dir
            os.system('mkdir ' + model_save_loc)
            os.system('mkdir ' + model_save_dir)
            file_methods.save_tf_model(full_model, model_name, model_save_dir, settings)
