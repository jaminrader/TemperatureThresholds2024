# TemperatureThresholds2024

Jamin K. Rader, Elizabeth A. Barnes, Noah S. Diffenbaugh
April 12, 2024

These scripts were run using the attached environment.yml file.

To Train a network:
    1) Use an experiment in experiment_settings.py, or specify and new experiment. The experiments used in the main text are entitled 'chosen1' 'chosen5' and 'chosen10'. All others were used for comparing SSPs or for tuning.
    2) Run train_model.py, and be sure to specify the experiments you want to run in the variable EXP_NAME_LIST.  

To Tune a network:
    1) Specify an experiment in experiment_settings.py. For hyperparameters that you want to tune, give them a list of values. Make sure to use another data type (e.g. int, float, string, or tuple) in the cases you are not using random search to select hyperparameters.
    2) The procedure is to first run the 'basetune' experiments, use eval_tune.ipynb to identify the best architectures, then run the 'tune' experiments.

To Make Figures:
    Code for making the figures can be found in final_figures.ipynb. Note: if you have trained the neural networks used in the paper, but you have not done any tuning, skip the cells that create plots for tuning diagnostics.

Description of scripts:
    base_directories.py - specify directories for figures and models to save to
    custom_metrics.py - custom metrics to assist with training
    data_processing.py - takes climate model data and observations and turns them into arrays for training
    eval_tune.ipynb - brief code for looking at tuning results
    experiment_settings.py - specify the experiments, including hyperparameters, regions, and input fields
    file_methods.py - additional functions for working with the CMIP6 data
    final_figures.ipynb - script for making the plots seen in the main and supp texts
    make_directories.py - creates directories from base_directories
    network.py - contains the Tensorflow neural network architecture, weights freezing, and compiling functions
    shash_tfp.py - uses tensorflow probability to create the loss function
    train_model.py - trains models specified in experiment_settings.py
    tune_model.py - used for tuning the network using random search, given a tunable experiment in experiment_settings.py
    tune.py - holds functions for tuning and analysis