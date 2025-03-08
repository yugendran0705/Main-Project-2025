from keras import layers
from keras import models
from keras.layers import InputSpec
import keras.optimizers as opts
import keras.regularizers as regularizers
from tensorflow.keras import backend as K
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from preprocessing.file_utils import LoadprofileGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.custom_layers import AdamWithWeightnorm
import pandas as pd
import numpy as np
import os
import gc
import types
import preprocessing.config as cfg
import tensorflow as tf

class CNNKerasRegressor(KerasRegressor):
    """ScikitLearn wrapper for keras models which incorporates
    batch-generation on top. This Class wraps CNN topologies."""

    def __init__(self, xZero=1,xOne=1, **kwargs):
        super().__init__(**kwargs)
        self.xZero = xZero
        self.xOne = xOne

    def save(self, uid):
        path = os.path.join(cfg.data_cfg['model_dump_path'],'cnn',uid)
        # self.model.save(path + '.h)  # everything saved
        self.model.save_weights(path + '.weights.h5')
        with open(path + '_arch.json', 'w') as f:
            f.write(self.model.to_json())

    def fit(self, x, y, **kwargs):
        assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame),\
            f'{self.__class__.__name__} needs pandas DataFrames as input'

        p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
        # print('p_id_col',p_id_col)
        window_size = kwargs.pop('window_size', None)
        data_cache = kwargs.pop('data_cache', {})
        cache = data_cache.get('data_cache', None)
        batch_size = kwargs.pop('batch_size', None)

        if cache is not None:
            # subsequent conduct iteration
            seq_tra = cache['seq_tra']
            kwargs['validation_data'] = cache['seq_val']
        else:
            # first conduct iteration
            seq_tra = self._generate_batches(x, y, p_id_col=p_id_col,
                                             batch_size=batch_size,
                                             window_size=window_size)
            x_val, y_val = kwargs.pop('validation_data')
            seq_val = self._generate_batches(x_val, y_val, p_id_col=p_id_col,
                                             batch_size=batch_size,
                                             window_size=window_size)

            kwargs['validation_data'] = seq_val
            new_cache = {'seq_tra': seq_tra, 'seq_val': seq_val}
            data_cache.update({'data_cache': new_cache})
        import numpy as np

        x_sample, y_sample = next(iter(seq_tra))  # Get one batch
        print(f"X shape: {np.array(x_sample).shape}")
        print(f"Y shape: {np.array(y_sample).shape}")


        return self.fit_generator(seq_tra, **kwargs)



    def fit_generator(self, seq, **kwargs):
        """Constructs a new model with `build_cnn_model(x_shape=(self.xZero,self.xOne))` & fits the model to Sequence via `fit()`.

        # Arguments
            seq : Sequence object
            **kwargs: dictionary arguments for `Sequential.fit()`

        # Returns
            history : Training history object
        """
        fit_args = kwargs.copy()

        if build_cnn_model(x_shape=(self.xZero,self.xOne)) is None:
            self.model = self.__call__(**kwargs)
        elif (not isinstance(build_cnn_model(x_shape=(self.xZero,self.xOne)), types.FunctionType) and
            not isinstance(build_cnn_model(x_shape=(self.xZero,self.xOne)), types.MethodType)):
            self.model = build_cnn_model(x_shape=(self.xZero,self.xOne))
        else:
            self.model = build_cnn_model(x_shape=(self.xZero,self.xOne))

        # Remove arguments not needed for fit()
        fit_args.pop('workers', None)
        fit_args.pop('use_multiprocessing', None)
        
        history = self.model.fit(seq, **fit_args)

        return history

    def predict(self, x, **kwargs):
        """Use this func to get a prediction for x. Return value has its
        dummy values removed that were due to batch size"""
        assert isinstance(x, pd.DataFrame), \
            f'{self.__class__.__name__} needs pandas DataFrames as input'
        p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
        batch_size = kwargs.pop('batch_size', None)
        window_size = kwargs.pop('window_size', None)
        seq = self._generate_batches(x,
                                     None,
                                     p_id_col=p_id_col, batch_size=batch_size,
                                     window_size=window_size, shuffle=False)

        yhat = self.predict_generator(seq, **kwargs)
        # yhat = np.squeeze(seq[0][1])
        return yhat

    def predict_generator(self, seq, **kwargs):
        """Returns predictions for the given test data.

        # Arguments
            x: Sequence object
            **kwargs: dictionary arguments
                Legal arguments are the arguments of
                `Sequential.predict_generator`.

        # Returns
            preds: array-like, shape `(n_samples,)` Predictions.
        """
        # kwargs = self.filter_sk_params(Sequential.predict_generator, kwargs)
        return np.squeeze(self.model.predict(seq, **kwargs))

    def score(self, x, y, **kwargs):
        """This score function returns the loss"""

        if kwargs.pop('score_directly', False):
            if np.any(np.isnan(y)):
                return 9999  # Return a constant loss for NaN values
            
            loss_fn = tf.keras.losses.get(self.model.loss)  # Convert string to function
            
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            
            loss = np.mean(loss_fn(y_tensor, x_tensor).numpy())

            print(f'Loss: {loss:.6f} K²')
            return loss
        else:
            raise NotImplementedError()
            # todo: make the below code work
            # sample weight needed
            p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
            batch_size = kwargs.get('batch_size', None)
            x, sample_weights = self._generate_batches(x, p_id_col=p_id_col,
                                                       batch_size=batch_size)
            kwargs['sample_weight'] = sample_weights
            n_dummy = batch_size - np.count_nonzero(sample_weights[-batch_size:, 0])
            y = np.vstack((y.values, np.zeros((n_dummy, y.shape[1]))))

            return super().score(x, y, **kwargs)

    @staticmethod
    def _generate_batches(_df_x, _df_y, p_id_col, batch_size, window_size,
                          shuffle=True):
        """Batches are sequences of length window_size, which are shuffled
        independently from each other. One original unique sequence produces
        sequence_length different sub-samples that are randomly mixed
        into batches of size batch_size. The sequence is prepadded with zeros in
        order to make up for the window (or lookback) to start way before
        the first element in the sequence such that the networks first output
        will start at the first element of the target sequence. Here,
        no explicit downsampling is elaborated since CNNs do it implicitly
        within the window.
        """

        if _df_y is not None:
            # target vectors shall not contain p_id_col since they aren't
            # needed to build the sequences from loadprofile generator
            if p_id_col in _df_y:
                _df_y.drop([p_id_col], axis=1, inplace=True)
            _df = pd.concat([_df_x, _df_y], axis=1)
        else:
            _df = _df_x

        p_ids = _df[p_id_col].unique().tolist()

        profile_dfs_l = [_df.loc[_df[p_id_col] == int(p), :] for p in p_ids]

        # prepad with zeros ( these can't be masked by sample_weight :( )
        zero_df = pd.DataFrame(0,
                               index=np.arange(window_size-1),
                               columns=_df.columns)

        # prepad and make starting values for temperatures more plausible
        prepadded_profile_dfs_l = \
            [pd.concat([zero_df.assign(**{c: df[c].iloc[0] for c in df.columns
                                          if c.startswith(('ambient',
                                                           'coolant')) and not
                                          '_ew_rolling_std_' in c}), df],
                       ignore_index=True)
                 .assign(**{p_id_col: df[p_id_col].unique().tolist()[0]})
             for df in profile_dfs_l]

        if _df_y is None:
            target = [np.zeros(len(df)).astype(np.int8)
                      for df in prepadded_profile_dfs_l]
        else:
            target = [df.loc[:, _df_y.columns].values for df in
                      prepadded_profile_dfs_l]
        samples = [df.loc[:, [c for c in _df_x.columns if c != p_id_col]].values
                   for df in prepadded_profile_dfs_l]
        del prepadded_profile_dfs_l
        gc.collect()
        # print('samples',samples)
        seq = LoadprofileGenerator(data=samples, targets=target,
                                   length=window_size,
                                   shuffle=shuffle, batch_size=batch_size)

        return seq


def build_cnn_model(x_shape=(256, 100),  # window_size x n_features
                    arch='plain', n_layers=2, n_units=64,
                    activation='relu', kernel_size=3, dilation_start_rate=1,
                    regularization_rate=1e-6, dropout_rate=0.5,
                    optimizer='adam', lr_rate=1e-4, loss='mse'):

    assert arch in ['plain', 'res'], "Specified architecture not recognized!"

    opts_map = {'adam': opts.Adam, 'nadam': opts.Nadam,
                'adamax': opts.Adamax, 'sgd': opts.SGD,
                'rmsprop': opts.RMSprop,
                'adam_weightnorm': AdamWithWeightnorm,}

    cnn_cfg = {'x_shape': x_shape,
               'arch': arch,
               'n_filters': int(n_units),
               'kernel_size': int(kernel_size),
               'dilation_start_rate': int(dilation_start_rate),
               'n_layers': int(n_layers),
               'activation': activation,
               'dropout_rate': dropout_rate,
               'reg_rate': regularization_rate,
               'batchnorm': '_weightnorm' not in optimizer and
                            activation != 'selu',
               }

    model = cnn_network(**cnn_cfg)

    opt = opts_map[optimizer](learning_rate=lr_rate)
    model.compile(optimizer=opt, loss=loss)
    model.summary()
    return model


def cnn_network(x_shape, arch, n_filters, kernel_size, dilation_start_rate,
                n_layers, activation, dropout_rate=0.5, reg_rate=1e-8,
                batchnorm=True):

    regs = {'kernel_regularizer': regularizers.l2(reg_rate),
            'bias_regularizer': regularizers.l2(reg_rate),
            'activity_regularizer': regularizers.l2(reg_rate)}


    dropout_layer = SpatialAlphaDropout1D if activation == 'selu' else \
        layers.SpatialDropout1D

    def add_common_layers(z):
        if batchnorm:
            z = layers.BatchNormalization()(z)
        z = layers.Activation(activation)(z)
        z = dropout_layer(dropout_rate)(z)
        return z

    def residual_block(z, dilation_start_rate):
        """optional block to use for automatic model building"""
        short_cut = z

        z = layers.Conv1D(n_filters, kernel_size, padding='causal',
                      dilation_rate=dilation_start_rate, activation=None,
                      input_shape=x_shape, **regs)(z)
        z = add_common_layers(z)

        z = layers.Conv1D(n_filters, kernel_size, padding='causal',
                          dilation_rate=dilation_start_rate*2,
                          activation=None,
                          input_shape=x_shape, **regs)(z)
        z = add_common_layers(z)

        short_cut = layers.Conv1D(n_filters, kernel_size=1, padding='causal',
                      dilation_rate=dilation_start_rate, activation=None,
                      input_shape=x_shape, **regs)(short_cut)

        z = layers.add([short_cut, z])

        return z

    x = layers.Input(shape=x_shape)
    y = x
    for i in range(n_layers):
        dilation_rate = dilation_start_rate * (2 ** i)
        if i % 2 == 0 and arch == 'res':  # every two layers
            shortcut = y

        y = layers.Conv1D(n_filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate,
                          activation=None,
                          input_shape=x_shape, **regs)(y)
        y = add_common_layers(y)

        if i % 2 == 1 and arch == 'res':  # every two layers (anti-cyclic)
            shortcut = layers.Conv1D(n_filters, kernel_size=1,
                                     padding='causal',
                                     dilation_rate=dilation_rate,
                                     activation=None,
                                     input_shape=x_shape, **regs)(shortcut)
            y = layers.add([shortcut, y])

    y = layers.GlobalMaxPooling1D()(y)
    y = layers.Dense(len(cfg.data_cfg['Target_param_names']))(y)

    model = models.Model(inputs=x, outputs=y)
    return model


class SpatialAlphaDropout1D(layers.AlphaDropout):

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape


def get_predefined_model():
    # input shape: window_size x len(x_cols)
    x = layers.Input((33, 91), name='input_62')
    y = layers.Conv1D(filters=126, kernel_size=2, padding='causal',
                      name='conv1d_154')(x)
    #y = layers.BatchNormalization(name='batch_normalization_118')(y)
    y = layers.Activation('relu', name='activation_118')(y)
    y = layers.Conv1D(filters=126, kernel_size=2, padding='causal',
                      dilation_rate=2, name='conv1d_155')(y)
    #y = layers.BatchNormalization(name='batch_normalization_119')(y)
    y = layers.Activation('relu', name='activation_119')(y)

    shortcut = layers.Conv1D(filters=126, kernel_size=1, padding='causal',
                             dilation_rate=2, name='conv1d_156')(x)
    y = layers.add([shortcut, y], name='add_37')

    y = layers.MaxPool1D(pool_size=33, name='global_max_pooling1d_62')(y)
    y = layers.Flatten()(y)
    y = layers.Dense(units=len(cfg.data_cfg['Target_param_names']),
                     name='dense_62')(y)

    model = models.Model(inputs=x, outputs=y)
    model.compile(optimizer=opts.Adam(lr=1e-9), loss='mse')
    return model


