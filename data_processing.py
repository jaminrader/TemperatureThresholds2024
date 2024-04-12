"""Build the split and scaled training, validation and testing data.

Functions
---------
get_members(settings)
get_observations(directory, settings)
compute_global_mean(da)
get_cmip_data(directory, rng, settings)
get_labels(da, settings, verbose=1)
add_input(data, input_name, prev = None)
preprocess_data(da, MEMBERS, settings)
make_data_split(da, data, f_labels, f_years, labels, years, settings)
"""
import numpy as np
import pandas as pd
import file_methods
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import xarray as xr


__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2023"


def get_members(settings):
    n_train = settings["n_train_val_test"][0]
    n_val   = settings["n_train_val_test"][1]
    n_test  = settings["n_train_val_test"][2]
    all_models = np.arange(0,n_train+n_val+n_test)

    return n_train, n_val, n_test, all_models

def get_observations(directory, settings):
    if settings["obsdata"] == "BEST":
        nc_filename_obs = 'Land_and_Ocean_LatLong1_185001_202312_anomalies_ann_mean_2pt5degree.nc'
    elif settings["obsdata"] == 'GISS': 
        nc_filename_obs = 'gistemp1200_GHCNv4_ERSSTv5_188001_202312_ann_mean_2pt5degree.nc'
    elif settings["obsdata"] == "NCEP":
        nc_filename_obs = 'NCEP_R1_air_surface_mon_mean_194801_202112_ann_mean_2pt5degree.nc' 
    elif settings["obsdata"] == 'ERA5':
        nc_filename_obs = 'ERA5_t2m_mon_194001-202402_194001_202312_ann_mean_2pt5degree.nc'
    else:
        raise NotImplementedError('no such obs data')

    da_obs = file_methods.get_netcdf_da(directory + nc_filename_obs)
    global_mean_obs = compute_global_mean(da_obs)

    data_obs = preprocess_data(da_obs, settings=settings) 

    x_obs = data_obs.values.reshape((data_obs.shape[0],np.product(data_obs.shape[1:-1], dtype=int), -1))
    if settings["anomalies"]:
        print('observations: filling NaNs with zeros')
        x_obs = np.nan_to_num(x_obs)

    print('np.shape(x_obs) = ' + str(np.shape(x_obs)))
    
    return data_obs, x_obs, global_mean_obs

def compute_global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    temp_weighted = da.weighted(weights)
    global_mean = temp_weighted.mean(("lon", "lat"), skipna=True)
    
    return global_mean

def get_cmip_data(directory, settings, verbose=1):
    data_train, data_val, data_test = None, None, None
    labels_train, labels_val, labels_test = None, None, None
    years_train, years_val, years_test = None, None, None
    target_years = []
    
    N_TRAIN, N_VAL, N_TEST, ALL_MODELS = get_members(settings)
    
    rng_cmip = np.random.default_rng(settings["seed"])
    train_models = rng_cmip.choice(ALL_MODELS, size=N_TRAIN, replace=False)
    val_models   = rng_cmip.choice(np.setdiff1d(ALL_MODELS,train_models), size=N_VAL, replace=False)
    if 'no_test' in list(settings.keys()):
        if settings['no_test']:
            test_models = val_models
        else:
            test_models = rng_cmip.choice(np.setdiff1d(ALL_MODELS,np.append(train_models[:],val_models)), size=N_TEST, replace=False)
    else:
        test_models = rng_cmip.choice(np.setdiff1d(ALL_MODELS,np.append(train_models[:],val_models)), size=N_TEST, replace=False)
    if verbose == 1:
        print(train_models, val_models, test_models)
    
    # save the meta data
    settings['train_models'] = train_models.tolist()
    settings['val_models'] = val_models.tolist()
    settings['test_models'] = test_models.tolist()
    
    # loop through and get the data
    filenames = file_methods.get_cmip_filenames(settings, directory, verbose=0)

    for imodel,f in enumerate(filenames):
        if verbose == 1:
            print(f)
        da = file_methods.get_da(directory, f, settings)

        f_labels, f_years, f_target_year = get_labels(da, settings,verbose=verbose)

        # create sets of train / validaton / test
        target_years = np.append(target_years,f_target_year)

        if imodel in train_models:

            data_train, labels_train, years_train = make_data_split(da, 
                                                                    data_train, 
                                                                    f_labels, 
                                                                    f_years, 
                                                                    labels_train,
                                                                    years_train,
                                                                    settings,
                                                                )
            if 'shuffle' in settings:
                if settings['shuffle']:
                    traina, trainb = data_train[..., 0], data_train[..., 1]
                    np.random.shuffle(trainb)
                    data_train = np.stack([traina, trainb], axis=-1)
        elif imodel in val_models:
            data_val, labels_val, years_val       = make_data_split(da, 
                                                                    data_val, 
                                                                    f_labels, 
                                                                    f_years, 
                                                                    labels_val,
                                                                    years_val,
                                                                    settings,
                                                                )
            
        if imodel in test_models:
            data_test, labels_test, years_test    = make_data_split(da, 
                                                                    data_test, 
                                                                    f_labels, 
                                                                    f_years, 
                                                                    labels_test,
                                                                    years_test,
                                                                    settings,
                                                                )

    YEARS_UNIQUE = np.unique(years_train)
    if verbose == 1:
        print('---------------------------')                
        print('data_train.shape = ' + str(np.shape(data_train)))
        print('data_val.shape = ' + str(np.shape(data_val)))
        print('data_test.shape = ' + str(np.shape(data_test)))

    x_train = data_train.reshape((np.product(data_train.shape[0:2], dtype=int),np.product(data_train.shape[2:-1], dtype=int), -1))
    x_val   = data_val.reshape((np.product(data_val.shape[0:2], dtype=int),np.product(data_val.shape[2:-1], dtype=int), -1))
    x_test  = data_test.reshape((np.product(data_test.shape[0:2], dtype=int),np.product(data_test.shape[2:-1], dtype=int), -1))

    y_train = labels_train.reshape((data_train.shape[0]*data_train.shape[1],))
    y_val   = labels_val.reshape((data_val.shape[0]*data_val.shape[1],))
    y_test  = labels_test.reshape((data_test.shape[0]*data_test.shape[1],))

    y_yrs_train = years_train.reshape((data_train.shape[0]*data_train.shape[1],))
    y_yrs_val   = years_val.reshape((data_val.shape[0]*data_val.shape[1],))
    y_yrs_test  = years_test.reshape((data_test.shape[0]*data_test.shape[1],))
    if verbose == 1:
        print(x_train.shape, y_train.shape, y_yrs_train.shape)
        print(x_val.shape, y_val.shape, y_yrs_val.shape)
        print(x_test.shape, y_test.shape, y_yrs_test.shape)  
    
    # make onehot vectors for training
    if settings["network_type"] == 'shash2':
        onehot_train = np.zeros((x_train.shape[0],2))
        onehot_train[:,0] = y_train.astype('float32')
        onehot_val = np.zeros((x_val.shape[0],2))    
        onehot_val[:,0] = y_val.astype('float32')
        onehot_test = np.zeros((x_test.shape[0],2))    
        onehot_test[:,0] = y_test.astype('float32')
    else:
        onehot_train = np.copy(y_train)
        onehot_val = np.copy(y_val)
        onehot_test = np.copy(y_test)    
    
    map_shape = np.shape(data_train)[2:]
    
    return x_train, x_val, x_test, y_train, y_val, y_test, onehot_train, onehot_val, onehot_test, y_yrs_train, y_yrs_val, y_yrs_test, target_years, map_shape, settings

# Get the labels (maximum annual global temperature)
def get_labels(da, settings, verbose=1):
    # select region
    if 'target_region' in settings:
        reglats, reglons = settings['target_region']
        da = da.sel(lat=slice(min(reglats), max(reglats)), lon=slice(min(reglons), max(reglons)))
    # compute the ensemble mean, global mean temperature
    # these computations should be based on the training set only
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    temp_weighted = da.weighted(weights)

    global_mean = temp_weighted.mean(("lon", "lat"))

    # compute the baseline temperature for each member individually (regardless of train/test set)
    new_change = False
    if not new_change:
        baseline = global_mean.sel(time=slice(str(settings["baseline_yr_bounds"][0]),
                                            str(settings["baseline_yr_bounds"][1]))).mean('time') # Shape: members
    else:
        baseline = global_mean # I think this is right.

    if new_change:
        labels = global_mean.rolling(min_periods=settings['len_window'], center=False, time=settings['len_window'],).max().shift(shifts={"time":-settings['len_window']}) - baseline
        labels = labels.values
        target_year = np.array([np.nan])
    
    else:

        if settings['model_type'] == 'static_threshold':

            largerbool = global_mean.values > baseline.values + settings["target_temp"]

            # The threshold must always be exceeded
            assert np.all(np.any(largerbool, axis=1)), "Threshold must be exceeded by each climate model member"
            
            # Get the years that the temperature exceeds the target threshold for each member
            iwarmer = np.argmax(largerbool, axis=-1)
            target_year = global_mean["time.year"][iwarmer]
            
            # define the labels
            if verbose == 1:
                print('TARGET_YEAR = ' + str(target_year.values) + ', TARGET_TEMP = ' + str(settings["target_temp"]))        
            labels = np.stack([target_year]*da['time.year'].shape[0]) - da['time.year'].values[..., None]

        elif settings['model_type'] == 'static_window':
            labels = (global_mean - baseline).rolling(min_periods=settings['len_window'], center=False, time=settings['len_window'],).max().shift(shifts={"time":-settings['len_window']})
            labels = labels.values
            target_year = np.array([np.nan])

    return labels, da['time.year'].values, target_year

### Add an input field for making predictions
def add_input(data, input_name, prev = None):
    if input_name == 'none':
        return (data.copy()*0.0).expand_dims(dim = ['field'], axis=[-1]), False
    # Determine window and offset (roll)
    assert 'timemean=' in input_name or 'timevar=' in input_name, "Must have timemean= or timevar= in input fields"
    ret_dat = data.copy()
    if 'latrange=' in input_name:
        latrange = np.array(input_name.split('latrange=')[1].split('_')[0].split(",")).astype(float)
        latmax, latmin = latrange.max(), latrange.min()
        ret_dat = ret_dat.sel(lat=slice(latmin,latmax))
    if 'spatialmean=' in input_name:
        varmean = input_name.split('spatialmean=')[1].split('_')[0]
        if varmean == 'global':
            weights = np.cos(np.deg2rad(ret_dat.lat))
            weights.name = "weights"
            temp_weighted = ret_dat.weighted(weights)
            global_temp = temp_weighted.mean(["lat", "lon"])
            global_temp = global_temp.expand_dims(dim = ['lon', 'lat'], axis=[-1, -2])
            ret_dat[:] = global_temp
        else:
            assert False, "Only spatial mean is global right now"
    if 'timemean=' in input_name:
        windowmean = int(input_name.split('timemean=')[1].split('_')[0])
        ret_dat = ret_dat.rolling(min_periods=windowmean, center=False, time=windowmean).mean("time")
    elif 'timevar=' in input_name:
        windowmean = int(input_name.split('timevar=')[1].split('_')[0])
        ret_dat = ret_dat.rolling(min_periods=windowmean, center=False, time=windowmean).std("time")
    if 'offset=' in input_name:
        offset = int(input_name.split('offset=')[1].split('_')[0])
        ret_dat = ret_dat.shift(shifts={"time":offset}, fill_value=np.nan)

    # add field axis
    ret_dat = ret_dat.expand_dims(dim = ['field'], axis=[-1])

    # take difference from previous field
    if 'diff=' in input_name:
        if bool(input_name.split('diff=')[1].split('_')[0]):
            ret_dat = ret_dat - prev

    drop_bool = False
    if 'drop=' in input_name:
        drop_bool = input_name.split('drop=')[1][:4] == 'True'
    return ret_dat, drop_bool

def preprocess_data(new_data, settings):
    
    #new_data = add_input(new_data, "global_timemean=5")
    input_fields_predrop = []
    drop_fields = []
    for ifield, input_field_name in enumerate(settings["input_fields"]):
        if ifield == 0:
            input_to_add, drop_bool = add_input(new_data, input_field_name)
        else:
            input_to_add, drop_bool = add_input(new_data, input_field_name, prev=input_fields_predrop[ifield-1])
        input_fields_predrop.append(input_to_add)
        drop_fields.append(drop_bool)

    # Drop fields that were used for taking a difference, but aren't to be included in the neural network
    input_fields = []
    for input_to_add, drop_bool in zip(input_fields_predrop, drop_fields):
        if not drop_bool:
            input_fields.append(input_to_add)   
    

    ## Testing importance of tendency
    #input_fields = [input_fields[0] - input_fields[1], input_fields[2]]

    new_data = xr.concat(input_fields, dim='field')


    #new_data = new_data.mean(("lat", "lon")).expand_dims(dim=["lat", "lon"], axis=[-2, -1])


    if settings["anomalies"] is True:
        new_data = new_data - new_data.sel(time=slice(str(settings["anomaly_yr_bounds"][0]),str(settings["anomaly_yr_bounds"][1]))).mean('time')
    if settings["anomalies"] == 'Baseline':
        new_data = new_data - new_data.sel(time=slice(str(settings["baseline_yr_bounds"][0]),str(settings["baseline_yr_bounds"][1]))).mean('time')
        new_data = new_data - new_data.sel(time=slice(str(settings["anomaly_yr_bounds"][0]),str(settings["anomaly_yr_bounds"][1]))).mean('time')
        
    if settings["remove_map_mean"]  == 'raw':
        new_data = new_data - new_data.mean(("lon","lat"))
    elif settings["remove_map_mean"] == 'weighted':
        weights = np.cos(np.deg2rad(new_data.lat))
        weights.name = "weights"
        new_data_weighted = new_data.weighted(weights)
        new_data = new_data - new_data_weighted.mean(("lon","lat"))
    
    if settings["remove_sh"] == True:
        # print('removing SH')
        i = np.where(new_data["lat"]<=-50)[0]
        if(len(new_data.shape)==3):
            new_data[:,i,:] = 0.0
        else:
            new_data[:,:,i,:] = 0.0
        
    return new_data


def make_data_split(da, data, f_labels, f_years, labels, years, settings):

    # process the data, i.e. compute anomalies, subtract the mean, etc.
    new_data = preprocess_data(da, settings)    
    n_members = new_data.shape[0]

    # only train on certain samples
    iyears = np.where((f_years >= settings["training_yr_bounds"][0]) & (f_years <= settings["training_yr_bounds"][1]))[0]    
    f_years = f_years[iyears]

    f_labels = f_labels[:, iyears] # grab just years for training
    new_data = new_data[:,iyears,...]
    
    if data is None:
        data = new_data.values
        labels = f_labels
        years = np.tile(f_years,(n_members,1))
    else:
        data = np.concatenate((data,new_data.values),axis=0)        
        labels = np.concatenate((labels, f_labels), axis=0)      
        years = np.concatenate((years,np.tile(f_years,(n_members,1))),axis=0)    

    return data, labels, years