"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
get_single_model_netcdf_da(directory, f)
get_da(directory, f, settings)
save_pred_obs(pred_vector, filename)
load_tf_model(model_name, directory)
load_settings_model(model_name, directory)
save_tf_model(model, model_name, directory, settings)
convert_to_cftime(da, orig_time)
get_single_model_names(settings, directory)
get_cmip_single_filenames(settings, directory)
get_cmip_ensemble_filenames(settings, verbose=0)
get_cmip_filenames(settings, directory=None, verbose=0)
"""

import xarray as xr
import json
import pickle
import tensorflow as tf
import custom_metrics
import os
import time
import network

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2024"


def get_model_name(settings):
    # model_name = (settings["exp_name"] + '_' +
    #               'ssp' + settings["ssp"] + '_' +
    #               str(settings["target_temp"]) + '_' +
    #               'gcmsub' + settings["gcmsub"] + '_' +
    #               settings["network_type"] + 
    #               '_rng' + str(settings["rng_seed"]) + 
    #               '_seed' + str(settings["seed"])
    #              )
    model_name = (settings["exp_name"] +
                  '_seed' + str(settings["seed"])
                  )

    return model_name

def get_netcdf_da(filename):
    try:
        da = xr.open_dataarray(filename)
    except: # if not single variable
        da = xr.open_dataset(filename).tas
    return da

def get_single_model_netcdf_da(directory, f):
    data_all = None
    if len(f) > 1:
        nc_filename = f[0]
        da_hist = get_netcdf_da(directory + nc_filename)
        nc_filename = f[1]
        da_ssp = get_netcdf_da(directory + nc_filename)
        da = xr.concat([da_hist, da_ssp], dim="year")
    else:
        da = get_netcdf_da(directory + f[0])

    da = convert_to_cftime(da, orig_time="year")
    da = da.expand_dims(dim = ['gcm'], axis=[0])
    return da

def get_da(directory, f, settings):
    if settings['ensemble_data']:
        da = get_netcdf_da(directory + f)
    else: # single model data
        da = get_single_model_netcdf_da(directory, f)

    return da

def save_pred_obs(pred_vector, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(pred_vector, f)


def load_tf_model(model_name, directory):
    # loading a tf model
    model = tf.keras.models.load_model(
        directory + model_name,
        compile=False,
        custom_objects={
            "RegressLossExpSigma": network.RegressLossExpSigma,
            "InterquartileCapture": custom_metrics.InterquartileCapture(),
            "SignTest": custom_metrics.SignTest(),
            "CustomMAE": custom_metrics.CustomMAE()
            },
        )
    return model

def load_settings_model(model_name, directory):
    # loading the .json file with settings
    with open(directory + model_name, 'r') as file:
        data = json.load(file)
    return data


def save_tf_model(model, model_name, directory, settings):

    # save the tf model
    tf.keras.models.save_model(model, directory + model_name + "_model", overwrite=True)

    # save the meta data
    with open(directory + model_name + '_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(settings))

def convert_to_cftime(da, orig_time):

    da = da.rename({orig_time: "time"})
    dates = xr.cftime_range(start="1850", periods=da.shape[0], freq="YS", calendar="noleap")
    da = da.assign_coords({"time": ("time", dates, {"units": "years since 1850-01-01"})})

    return da

def get_single_model_names(settings, directory):
    model_names = []
    fns = os.listdir(directory)
    for fn in fns:
        if ('_ssp' + settings['ssp'] + '_' in fn) \
        and (fn.endswith('.nc')) \
        and ('-' not in fn.split('_r')[1].split('i')[0]):
            fnbeg = fn.split('_ann_mean')[0]
            modname = fnbeg.split('ssp' + settings['ssp'] + '_')[-1]
            model_names.append(modname)
    return model_names

def get_cmip_single_filenames(settings, directory):
    cmpi6_model_names = get_single_model_names(settings, directory)
    model_filenames = []

    for model_name in cmpi6_model_names:
        # combine historical and ssp
        if settings['ssp'] == '370':
            nc_filename_hist = "tas_Amon_historical_" + model_name + "_ann_mean_2pt5degree.nc"
            nc_filename_ssp = "tas_Amon_ssp" + settings['ssp'] + "_" + model_name + "_ann_mean_2pt5degree.nc"
            model_filenames.append([nc_filename_hist, nc_filename_ssp])
        elif settings['ssp'] in ['126','245']:
            model_filenames.append(["tas_Amon_historical_ssp" + settings['ssp'] + "_" + model_name + "_ann_mean_2pt5degree.nc"])

    return model_filenames

def get_cmip_ensemble_filenames(settings, verbose=0):
    if settings["ssp"] == '370' and settings["gcmsub"] == 'ALL':
        filenames = ('tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_CESM2-LE2-smbb_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif settings["ssp"] == '245' and settings["gcmsub"] == 'ALL':
        filenames = (
            'tas_Amon_historical_ssp245_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp245_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp245_CNRM-ESM2-1_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp245_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp245_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp245_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc',
            )
    elif settings["ssp"] == '370' and settings["gcmsub"] == 'UNIFORM':
        filenames = ('tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     # 'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'ALL') and (settings["target_temp"] == 2.0)):
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )

    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'ALL7')):
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'ALL10')):
        filenames = (
            'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
            )
    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'noHIGH10')):
        filenames = (
            'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
            )
    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'noHIGH7')):
        filenames = (
            'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
            )
    elif ((settings["ssp"] == '126' and settings["gcmsub"] == 'noHIGH5')):
        filenames = (
            'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
            #  'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
            )

    elif settings["ssp"] == '126' and settings["gcmsub"] == 'ALL':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'noM6':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     # 'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',                     
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'FORCE':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'EXTEND':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )

    elif settings["ssp"] == '126' and settings["gcmsub"] == 'UNIFORM':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )
    elif settings["gcmsub"] == 'OOS':
        filenames = (
            'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_MRI-ESM2-0_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-CM6-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_CNRM-ESM2-1_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_GISS-E2-1-G_r1-5_ncecat_ann_mean_2pt5degree.nc',
            'tas_Amon_historical_ssp126_IPSL-CM6A-LR_r1-5_ncecat_ann_mean_2pt5degree.nc',
            )

    # elif settings["ssp"] == '126' and settings["gcmsub"] == 'MIROC':
    #     filenames = (
    #                  'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                  'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',            
    #                 )
    # elif settings["ssp"] == '370' and settings["gcmsub"] == 'MIROC':
    #     filenames = (
    #                  # 'tas_Amon_historical_ssp370_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
    #                  'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',    
    #                 )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'MAX':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     )


    else:
        raise NotImplementedError('no such SSP')

    if verbose != 0:
        print(filenames)

    return filenames

def get_cmip_filenames(settings, directory=None, verbose=0):
    if settings['ensemble_data']:
        return get_cmip_ensemble_filenames(settings, verbose)
    else:
        return get_cmip_single_filenames(settings, directory)