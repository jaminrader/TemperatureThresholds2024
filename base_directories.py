""" Define default settings."""

import socket
import os

def get_directories():

    dir_settings = {
        "data": 'data/',
        "figure" : 'figures/',
        "predictions" : 'saved_predictions/',
        "diagnostics" : 'model_diagnostics/',
        "models" : 'saved_models/',
        "tuner_directory" : 'tuning_results/',
        "tuner_autosave_directory": 'tuning_results/autosave/',
    }

    return dir_settings

def make_directories():
    dir_settings = get_directories()
    for dir in dir_settings:
        os.system('mkdir ' + dir_settings[dir])

