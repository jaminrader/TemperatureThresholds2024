"""Tuning script.

Tunes a model given parameters for a simulation in experiment_settings.py
"""

import tune
import numpy as np

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2024"

# Specify a list of experiments to tune
EXP_NAMES = ["basetune1_singlearch", "basetune5_singlearch", "basetune10_singlearch"]


NTRIALS = None # None reverts to experiment dictionary

if __name__ == "__main__":

    for exp_name in EXP_NAMES:

        tune.tune(exp_name, seed=0, ntrials=NTRIALS)