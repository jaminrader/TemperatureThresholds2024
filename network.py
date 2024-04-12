"""Neural network functions.

Classes
---------
Exponentiate(keras.layers.Layer)


Functions
---------
RegressLossExpSigma(y_true, y_pred)
compile_model(x_train, y_train, settings)
build_baseline_model(x_train, y_train, settings)
build_mu_sigma_delta_model(in_layers, baseline_input, settings, name)
build_delta_model(x_train, y_train, settings)
compile_full_model(x_train, y_train, settings)
set_trainable(model, full_model, settings, trainable, learning_rate)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax, Flatten, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import Callback
import custom_metrics

__author__ = "Jamin K. Rader, Elizabeth A. Barnes and Noah S. Diffenbaugh"
__version__ = "12 April 2024"

print(tf.__version__)
print(tfp.__version__)


class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)

def RegressLossExpSigma(y_true, y_pred):
    # network predictions
    mu = y_pred[:,0]
    sigma = y_pred[:,1]
    
    # normal distribution defined by N(mu,sigma)
    norm_dist = tfp.distributions.Normal(mu,sigma)

    # compute the log as the -log(p)
    loss = -norm_dist.log_prob(y_true[:,0])    

    return tf.reduce_mean(loss, axis=-1)    

def build_baseline_model(x_train, y_train, settings):

    if "init_seed" in list(settings.keys()):
        init_seed = settings["init_seed"]
    else:
        init_seed = settings["seed"]

    inputs = Input(shape=x_train.shape[1:]) 

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    layers = normalizer(inputs)

    layers = tf.reduce_mean(layers, axis=-1) # Turns global mean into one value
    layers = tf.expand_dims(layers, axis=-1)
    
    layers_mu = tf.keras.layers.Layer()(layers)
    layers_sigma = tf.keras.layers.Layer()(layers)
    
    activation = settings["act_fun1"]
    ridge = settings["ridge_param1"]

    for hidden in settings["hiddens1"]:

        layers_mu = Dense(hidden, activation=activation,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=init_seed),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=init_seed))(layers_mu)
        layers_sigma = Dense(hidden, activation=activation,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=init_seed),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=init_seed))(layers_sigma)
        ridge = 0 # only first layer has ridge
        
    y_avg = np.mean(y_train)
    y_std = np.std(y_train)

    mu_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=init_seed+100),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=init_seed+100),
        name="mu_z_unit",
    )(layers_mu)
    
    mu_unit = tf.keras.layers.Rescaling(
        scale=y_std,
        offset=y_avg,
        name="mu_unit",
    )(mu_z_unit)
    
    # sigma_unit. The network predicts the log of the scaled sigma_z, then
    # the resclaing layer scales it up to log of sigma y, and the custom
    # Exponentiate layer converts it to sigma_y.
    log_sigma_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_sigma_z_unit",
    )(layers_sigma)

    log_sigma_unit = tf.keras.layers.Rescaling(
        scale=1.0,
        offset=np.log(y_std),
        name="log_sigma_unit",
    )(log_sigma_z_unit)

    sigma_unit = Exponentiate(
        name="sigma_unit",
    )(log_sigma_unit)

    outputs = mu_unit, sigma_unit

    baseline_model = Model(inputs, outputs, name="baseline_model")

    return baseline_model

def build_mu_sigma_delta_model(in_layers, baseline_input, settings, name):

    if "init_seed" in list(settings.keys()):
        init_seed = settings["init_seed"]
    else:
        init_seed = settings["seed"]

    init = tf.keras.initializers.RandomNormal(seed=init_seed)

    inputs = Input(shape=in_layers.shape[1:])
    layers_delta = Dropout(rate=settings["dropout_rate"],
                     seed=init_seed)(inputs) 
    activation = settings["act_fun2"]
    ridge = settings["ridge_param2"]
    for hidden in settings["hiddens2"]:
        layers_delta = Dense(hidden, activation=activation,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                        bias_initializer=init,
                        kernel_initializer=init,)(layers_delta)
        ridge = 0 # only first layer has ridge

    # Incorporating baseline knowledge
    layers_delta = Concatenate()([layers_delta, baseline_input])
    activation = settings["act_fun3"]
    ridge = settings["ridge_param3"]
    for hidden in settings["hiddens3"]:
        layers_delta = Dense(hidden, activation=activation,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                        bias_initializer=init,
                        kernel_initializer=init,)(layers_delta)
        ridge = 0 # only first layer has ridge

    layers_delta = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="mu_z_unit",
    )(layers_delta)

    mu_sigma_delta_model = Model([inputs, baseline_input], layers_delta, name=name)



    return mu_sigma_delta_model

def build_delta_model(x_train, y_train, settings):

    if "init_seed" in list(settings.keys()):
        init_seed = settings["init_seed"]
    else:
        init_seed = settings["seed"]

    inputs = Input(shape=x_train.shape[1:]) 
    baseline_input_layer = Input(shape=(2,)) # SHASH2

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    layers = normalizer(inputs)
    layers = Flatten()(layers)

    mu_delta_model = build_mu_sigma_delta_model(layers, baseline_input_layer, settings, "delta_layers")
    sigma_delta_model = build_mu_sigma_delta_model(layers, baseline_input_layer, settings, "epsilon_layers")
    delta_z_unit = mu_delta_model([layers, baseline_input_layer])
    log_epsilon_unit = sigma_delta_model([layers, baseline_input_layer])

    y_avg = np.mean(y_train)
    y_std = np.std(y_train)
    
    delta_unit = tf.keras.layers.Rescaling(
        scale=y_std, # rescale to standard deviation of y
        offset=0.0, # No offset, because this is a delta mu value
        name="mu_unit",
    )(delta_z_unit)

    epsilon_unit = Exponentiate(
        name="epsilon_unit",
    )(log_epsilon_unit)

    outputs = delta_unit, epsilon_unit

    delta_model = Model([inputs, baseline_input_layer], outputs, name="delta_model")

    return delta_model


def compile_full_model(x_train, y_train, settings):

    LOSS = RegressLossExpSigma
    metrics = [
                LOSS,
                custom_metrics.CustomMAE(name="custom_mae"),
                custom_metrics.InterquartileCapture(name="interquartile_capture"),
                custom_metrics.SignTest(name="sign_test"),
                ]

    input_layer = Input(shape=x_train.shape[1:], name="input_layer")

    long_input = input_layer[..., 0]
    short_input = input_layer[..., 1:]

    baseline_model = build_baseline_model(x_train[..., 0], y_train, settings)
    delta_model = build_delta_model(x_train[..., 1:], y_train, settings)

    base_mu, base_sigma = baseline_model(long_input)
    baseline_knowledge_input_layer = tf.keras.layers.concatenate([base_mu, base_sigma], axis=1)
    delta, epsilon = delta_model([short_input, baseline_knowledge_input_layer])

    mu = base_mu + delta
    sigma = base_sigma * epsilon
    
    output_layer = tf.keras.layers.concatenate([mu, sigma], axis=1)
        
    # Constructing the model
    full_model = Model(input_layer, output_layer, name="full_model")

    full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["baseline_learning_rate"]), 
                  loss=LOSS, 
                  metrics=metrics,
                 )
        
        
    full_model.summary()
    
    return full_model, baseline_model, delta_model

def set_trainable(model, full_model, settings, trainable, learning_rate):

    LOSS = RegressLossExpSigma
    metrics = [
                LOSS,
                'mae',
                custom_metrics.CustomMAE(name="custom_mae"),
                custom_metrics.InterquartileCapture(name="interquartile_capture"),
                custom_metrics.SignTest(name="sign_test"),
                ]

    model.trainable = trainable

    full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss=LOSS, 
                  metrics=metrics,
                 )