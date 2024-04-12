"""Tuning functions.

Functions
---------
random_select(settings, trial_num)
check_autosave(exp_name, autosave_dir)
make_json_friendly(specs_orig)
build_and_train_model(inputs, outputs, y_train, trial_specs, tunername, init_model=None)
    includes: scheduler(epoch, learning_rate)
tune(exp_name, seed=0, ntrials=30)
get_best_models(exp_name, num_models=1, print_out=False, track_loss = "val")
get_model_losses(exp_name, track_loss)
"""

# Tune the neural network for a given problem
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import silence_tensorflow.auto
import pickle

import experiment_settings
import network, data_processing, base_directories

import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2024"


def random_select(settings, trial_num):

    rng = np.random.default_rng(seed=int(trial_num))

    # Keep a hold of the specifications this trial is going to use
    trial_settings = dict()
    for key in list(settings.keys()):
        if type(settings[key]) == type([]): # if list, choose one randomly
            trial_settings[key] = rng.choice(settings[key])
        else: # if not, it's a fixed param
            trial_settings[key] = settings[key]

    # Make sure seed is of type int
    trial_settings["seed"] = int(trial_settings["seed"])

    return trial_settings

def check_autosave(exp_name, autosave_dir):

    # Open the autosaved pickle file
    with open(autosave_dir + exp_name + ".p", 'rb') as fp:
        saved_tuner_results = pickle.load(fp)

    # Find index of the last autosaved model
    autosaved_at = max(list(saved_tuner_results.keys()))

    return saved_tuner_results, autosaved_at

def make_json_friendly(specs_orig):
    specs = specs_orig.copy()
    # Removes numpy objects from dictionary, and turns lists into strings
    for imod in specs.keys():
        for key in specs[imod].keys():
            if type(specs[imod][key]) == np.ndarray:
                specs[imod][key] = specs[imod][key].tolist()
            if type(specs[imod][key]) == list:
                specs[imod][key] = str(specs[imod][key])
            if type(specs[imod][key]) == np.int64:
                specs[imod][key] = int(specs[imod][key])
    return specs

def build_and_train_model(inputs, outputs, y_train, trial_specs, tunername, init_model=None):
    x_train, x_val = inputs
    onehot_train, onehot_val = outputs

    ### Set seed for one model
    tf.keras.utils.set_random_seed(trial_specs['seed'])

    ### Set up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_RegressLossExpSigma',
                                                       patience=trial_specs['patience'],
                                                       verbose=1,
                                                       mode='auto',
                                                       restore_best_weights=True)   
    
    def scheduler(epoch, learning_rate):
        # Set the learning rate to 0 for the first epoch, and then to the specified learning_rate
        return 0.0 if epoch < 1 else learning_rate
    learning_rate_scheduler = LearningRateScheduler(lambda epoch: scheduler(epoch, trial_specs["learning_rate"]))

    ### Build the model
    tf.keras.backend.clear_session()   

    full_model, baseline_model, delta_model = network.compile_full_model(x_train, y_train, trial_specs)
    ### Train the model
    # Train the baseline model first

    if (tunername.startswith('basetune') or tunername.startswith('tune')) and (init_model == None):
        print('TRAINING BASELINE MODEL')
        network.set_trainable(delta_model, full_model, trial_specs, False, learning_rate=trial_specs["baseline_learning_rate"])

        history = full_model.fit(x_train, onehot_train, 
                            epochs=trial_specs['n_epochs'], 
                            batch_size = trial_specs['batch_size'], 
                            shuffle=True,
                            validation_data=[x_val, onehot_val],
                            callbacks=[early_stopping,],
                            verbose=0,                        
                            )
        
        if tunername.startswith('tune'): # Only save the baseline model for the second stage of tuning
            init_model, __, __ = network.compile_full_model(x_train, y_train, trial_specs)
            init_model.get_layer("baseline_model").set_weights(baseline_model.get_weights())

    else:
        baseline_model.set_weights(init_model.get_layer("baseline_model").get_weights())
        
    if tunername.startswith('tune'):
        # Then train the delta model
        print('TRAINING DELTA MODEL')
        network.set_trainable(baseline_model, full_model, trial_specs, False, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model, full_model, trial_specs, True, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model.get_layer("delta_layers"), full_model, trial_specs, True, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, trial_specs, False, learning_rate=trial_specs["learning_rate"])

        history = full_model.fit(x_train, onehot_train, 
                            epochs=trial_specs['n_epochs'], 
                            batch_size = trial_specs['batch_size'], 
                            shuffle=True,
                            validation_data=[x_val, onehot_val],
                            callbacks=[early_stopping,learning_rate_scheduler,],
                            verbose=0,                        
                            )
        
        network.set_trainable(baseline_model, full_model, trial_specs, False, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model, full_model, trial_specs, True, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model.get_layer("delta_layers"), full_model, trial_specs, False, learning_rate=trial_specs["learning_rate"])
        network.set_trainable(delta_model.get_layer("epsilon_layers"), full_model, trial_specs, True, learning_rate=trial_specs["learning_rate"])

        history = full_model.fit(x_train, onehot_train, 
                            epochs=trial_specs['n_epochs'], 
                            batch_size = trial_specs['batch_size'], 
                            shuffle=True,
                            validation_data=[x_val, onehot_val],
                            callbacks=[early_stopping,learning_rate_scheduler,],
                            verbose=0,                        
                            )
    
    return full_model, init_model, trial_specs
    
    

def tune(exp_name, seed=0, ntrials=30):
    
    tf.keras.utils.set_random_seed(seed) # doesn't really do anything now

    # Get the specs for the experiment
    settings = experiment_settings.get_settings(exp_name)

    if ntrials==None:
        ntrials = settings['n_models']

    # Get the directory names
    dir_settings = base_directories.get_directories()

    # Prep the results dictionary
    os.system('mkdir ' + dir_settings['tuner_directory'])

    # Set frequency to autosave (hardcode only)
    autosave_every = 5

    # Check autosave and initialize tuner_results dictionary
    try:
        os.system('mkdir ' + dir_settings['tuner_autosave_directory'])
        tuner_results, autosaved_at = check_autosave(exp_name, dir_settings['tuner_autosave_directory'])
    except:
        print('No autosave found. Starting tuner from trial 0.')
        tuner_results = dict()
        autosaved_at = -1 # not autosaved, will tune like normal

    # Start the tuning
    first_loop = True
    for itune in range(ntrials):
        # Bypass training new model if autosave has already done this model
        if itune <= autosaved_at:
            continue
        else:
            # Randomly select from all possible choices
            tuner_results[itune] = random_select(settings, itune)

        trial_specs = tuner_results[itune]

        # Build the data
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
            trial_specs) = data_processing.get_cmip_data(dir_settings['data'], trial_specs)
        

        ### Build and Train the model
        # Wrapped with try/except to reduce issues with gradient tape crashes
        if first_loop: #and type(settings['seed']) == int:
            init_model = None # train baseline_model only once
            first_loop = False
        try:
            full_model, init_model, trial_specs = \
                build_and_train_model((x_train, x_val), 
                                    (onehot_train, onehot_val), y_train,
                                    trial_specs, exp_name, init_model = init_model)
        except:
            full_model, init_model, trial_specs = \
                build_and_train_model((x_train, x_val), 
                                    (onehot_train, onehot_val), y_train,
                                    trial_specs, exp_name, init_model=init_model)
                
        ### Get the observations for validation

        def calc_y_obs(global_mean_map = None, global_mean_vals = None):
            assert global_mean_map is not None or global_mean_vals is not None, "One of the inputs to calc_y_obs must not be None"
            if global_mean_map != None:
                weights = np.cos(np.deg2rad(da_obs.lat))
                weights.name = "weights"
                da_obs_weighted = (global_mean_map).weighted(weights)
                global_mean_vals = da_obs_weighted.mean(["lat", "lon"])

            BEST_2023_mean = 1.54
            ann_mean_v_anom_diff = BEST_2023_mean - global_mean_vals[2023-end_year-1]
            ann_mean_anom_obs = global_mean_vals + ann_mean_v_anom_diff
            roll = settings['len_window']
            max_T_anom_obs = ann_mean_anom_obs.rolling(min_periods=roll, center=False, time=roll).max().shift(shifts={"time":-roll})
            return max_T_anom_obs, ann_mean_anom_obs

        da_obs, x_obs, global_mean_obs = data_processing.get_observations(dir_settings['data'], trial_specs)

        # params
        end_year = da_obs.time.dt.year.values.max()
        obs_metric_start_year = 1980
        obs_yrs = np.arange(end_year-x_obs.shape[0]+1, end_year+1)

        # Calculations
        y_obs, ann_mean_obs = calc_y_obs(global_mean_vals = global_mean_obs)
        obs_baseline = ann_mean_obs.values[np.logical_and(obs_yrs<=settings['anomaly_yr_bounds'][1], obs_yrs>=settings['anomaly_yr_bounds'][0])].mean() # To adjust network predictions
        y_obs = y_obs.values - obs_baseline

        # Boolean arrays for selecting observed years
        years_bool = np.logical_and(obs_yrs >= obs_metric_start_year, obs_yrs <= end_year - trial_specs["len_window"])
        early_years_bool = np.logical_and(obs_yrs >= obs_metric_start_year, obs_yrs <= 1999 - trial_specs["len_window"])
        no_pinatubo_bool = np.logical_or(obs_yrs < 1991-trial_specs["len_window"], obs_yrs >= 1996)
        no_pinatubo_years_bool = np.logical_and(no_pinatubo_bool,years_bool)
        no_pinatubo_early_years_bool = np.logical_and(no_pinatubo_bool,early_years_bool)
        y_1_2_bool = np.logical_and(y_test > 1, y_test < 2)

        ### Evaluate on the validation set
        if exp_name.startswith('tune'):
            trial_metrics_baseline = init_model.evaluate(x_val, y_val)
            trial_metrics_test_baseline = init_model.evaluate(x_test, y_test)
            trial_metrics_test12_baseline = init_model.evaluate(x_test[y_1_2_bool], y_test[y_1_2_bool])
            trial_metrics_obs_baseline = init_model.evaluate(x_obs[years_bool], y_obs[years_bool])
            trial_metrics_obs_nopinatubo_baseline = init_model.evaluate(x_obs[no_pinatubo_years_bool], y_obs[no_pinatubo_years_bool])
            if trial_specs['len_window'] < 15:
                trial_metrics_obs_20C_baseline = init_model.evaluate(x_obs[early_years_bool],  y_obs[early_years_bool])
            if trial_specs['len_window'] <= 5:
                trial_metrics_obs_20C_nopinatubo_baseline = init_model.evaluate(x_obs[no_pinatubo_early_years_bool], y_obs[no_pinatubo_early_years_bool])

        trial_metrics = full_model.evaluate(x_val, y_val)
        trial_metrics_test = full_model.evaluate(x_test, y_test)
        trial_metrics_test12 = full_model.evaluate(x_test[y_1_2_bool], y_test[y_1_2_bool])
        trial_metrics_obs = full_model.evaluate(x_obs[years_bool], y_obs[years_bool])
        trial_metrics_obs_nopinatubo = full_model.evaluate(x_obs[no_pinatubo_years_bool], y_obs[no_pinatubo_years_bool])
        if trial_specs['len_window'] < 15:
            trial_metrics_obs_20C = full_model.evaluate(x_obs[early_years_bool], y_obs[early_years_bool])
        if trial_specs['len_window'] <= 5:    
            trial_metrics_obs_20C_nopinatubo = full_model.evaluate(x_obs[no_pinatubo_early_years_bool], y_obs[no_pinatubo_early_years_bool])
         
        # Update the current trial dictionary
        tuner_results[itune]['results'] = dict()

        tuner_results[itune]['results']['val_loss'] = float(trial_metrics[1])
        tuner_results[itune]['results']['obs_loss'] = float(trial_metrics_obs[1])
        tuner_results[itune]['results']['test_loss'] = float(trial_metrics_test[1])
        tuner_results[itune]['results']['test12_loss'] = float(trial_metrics_test12[1])
        tuner_results[itune]['results']['obs_nopinatubo_loss'] = float(trial_metrics_obs_nopinatubo[1])

        tuner_results[itune]['results']['val_mae'] = float(trial_metrics[2])
        tuner_results[itune]['results']['obs_mae'] = float(trial_metrics_obs[2])
        tuner_results[itune]['results']['test_mae'] = float(trial_metrics_test[2])
        tuner_results[itune]['results']['test12_mae'] = float(trial_metrics_test12[2])
        tuner_results[itune]['results']['obs_nopinatubo_mae'] = float(trial_metrics_obs_nopinatubo[2])


        if exp_name.startswith('tune'):
            tuner_results[itune]['results']['bl_val_loss'] = float(trial_metrics_baseline[1])
            tuner_results[itune]['results']['bl_obs_loss'] = float(trial_metrics_obs_baseline[1])
            tuner_results[itune]['results']['bl_test_loss'] = float(trial_metrics_test_baseline[1])
            tuner_results[itune]['results']['bl_test12_loss'] = float(trial_metrics_test12_baseline[1])
            tuner_results[itune]['results']['bl_obs_nopinatubo_loss'] = float(trial_metrics_obs_nopinatubo_baseline[1])

            tuner_results[itune]['results']['bl_val_mae'] = float(trial_metrics_baseline[2])
            tuner_results[itune]['results']['bl_obs_mae'] = float(trial_metrics_obs_baseline[2])
            tuner_results[itune]['results']['bl_test_mae'] = float(trial_metrics_test_baseline[2])
            tuner_results[itune]['results']['bl_test12_mae'] = float(trial_metrics_test12_baseline[2])
            tuner_results[itune]['results']['bl_obs_nopinatubo_mae'] = float(trial_metrics_obs_nopinatubo_baseline[2])

            if trial_specs['len_window'] < 15:
                tuner_results[itune]['results']['obs20C_loss'] = float(trial_metrics_obs_20C[1])
                tuner_results[itune]['results']['obs20C_mae'] = float(trial_metrics_obs_20C[2])
                tuner_results[itune]['results']['bl_obs20C_loss'] = float(trial_metrics_obs_20C_baseline[1])
                tuner_results[itune]['results']['bl_obs20C_mae'] = float(trial_metrics_obs_20C_baseline[2])
            
            if trial_specs['len_window'] <= 5:
                tuner_results[itune]['results']['obs20C_nopinatubo_loss'] = float(trial_metrics_obs_20C_nopinatubo[1])
                tuner_results[itune]['results']['obs20C_nopinatubo_mae'] = float(trial_metrics_obs_20C_nopinatubo[2])
                tuner_results[itune]['results']['bl_obs20C_nopinatubo_loss'] = float(trial_metrics_obs_20C_nopinatubo_baseline[1])
                tuner_results[itune]['results']['bl_obs20C_nopinatubo_mae'] = float(trial_metrics_obs_20C_nopinatubo_baseline[2])


        print('Model ' + str(itune) + ' trained.')

        # Autosave data every 'autosave_every' trained models
        if (itune%autosave_every == autosave_every-1):
            os.system('mkdir ' + dir_settings['tuner_autosave_directory'])
            with open(dir_settings['tuner_autosave_directory'] + exp_name + ".json", 'w') as fp:
                json.dump(make_json_friendly(tuner_results), fp)
            with open(dir_settings['tuner_autosave_directory'] + exp_name + ".p", 'wb') as fp:
                pickle.dump(tuner_results, fp)

    # Save final data
    with open(dir_settings['tuner_directory'] + exp_name + ".json", 'w') as fp:
        json.dump(make_json_friendly(tuner_results), fp)
    with open(dir_settings['tuner_directory'] + exp_name + ".p", 'wb') as fp:
        pickle.dump(tuner_results, fp)

    print('Finished tuning experiment ' + exp_name + ".")

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
