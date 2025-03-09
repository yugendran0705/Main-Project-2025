import numpy as np
import pandas as pd
import os

from keras import models

import keras.callbacks
from tensorflow.keras import models, regularizers, initializers
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Input, Dense, TimeDistributed, GaussianNoise, Add, Flatten
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, SGD, RMSprop
from tensorflow.keras.utils import get_custom_objects
from keras import backend as K
from scikeras.wrappers import KerasRegressor
from sklearn.utils.validation import check_array
import tensorflow as tf
import preprocessing.config as cfg
from preprocessing.custom_layers import AdamWithWeightnorm, LayerNormalization


class RNNKerasRegressor(KerasRegressor):
    """ScikitLearn wrapper for keras models which incorporates
    batch-generation on top. This Class wraps RNN topologies."""

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        self.xZero = 69
        self.xOne = 128
        self.xTwo = 91

    def reset_states(self):
        self.model.reset_states()

    def save(self, uid):
        path = os.path.join(cfg.data_cfg['model_dump_path'], 'rnn', uid)
        # self.model.save(path + '.h)  # everything saved
        self.model.save_weights(path + '.weights.h5')
        try:
            with open(path + '_arch.json', 'w') as f:
                f.write(self.model.to_json())
        except TypeError as err:
            print(err)
            print('Model architecture will not be saved.')

    # def fit(self, x, y, **kwargs):
    #     # assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame), \
    #     #     f'{self.__class__.__name__} needs pandas DataFrames as input'

    #     p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
    #     data_cache = kwargs.pop('data_cache', {})
    #     cache = data_cache.get('data_cache', None)
    #     downsample_rate = kwargs.pop('downsample_rate', None)
    #     tbptt_len = kwargs.pop('tbptt_len', None)

    #     if cache is not None:
    #         # Load from cache for efficiency
    #         x, y = cache['x'], cache['y']
    #         kwargs['sample_weight'] = cache['sample_weight']
    #         kwargs['validation_data'] = (cache['x_val'], cache['y_val'], cache['val_sample_weight'])
    #     else:
    #         # Generate batches for the first iteration
    #         batch_generation_cfg = {'p_id_col': p_id_col,
    #                                 'batch_size': kwargs.get('batch_size', 32),
    #                                 'downsample_rate': downsample_rate,
    #                                 'tbptt_len': tbptt_len}

    #         x, sample_weights = self._generate_batches(x, **batch_generation_cfg)
    #         y, _ = self._generate_batches(y, **batch_generation_cfg)
    #         kwargs['sample_weight'] = sample_weights

    #         if 'validation_data' in kwargs:
    #             x_val, y_val = kwargs.pop('validation_data')
    #             x_val, val_sample_weights = self._generate_batches(x_val, **batch_generation_cfg)
    #             y_val, _ = self._generate_batches(y_val, **batch_generation_cfg)
    #             kwargs['validation_data'] = (x_val, y_val, val_sample_weights)

    #             new_cache = {'x': x, 'y': y, 'sample_weight': sample_weights,
    #                             'x_val': x_val, 'y_val': y_val,
    #                             'val_sample_weight': val_sample_weights}
    #             data_cache.update({'data_cache': new_cache})

    #     # Train model
    #     # x = x.reshape(-1,x.shape[2])
    #     # y = y.reshape(-1,y.shape[2])
    #     print(x.shape, y.shape)
        

    #     history = self.fit(x, y, **kwargs)
    #     self.model.reset_states()
    #     return history
    def fit(self, x, y, **kwargs):
        assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame), \
            f'{self.__class__.__name__} needs pandas DataFrames as input'
        # Extract known arguments from kwargs
        p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
        window_size = kwargs.pop('window_size', None)
        data_cache = kwargs.pop('data_cache', {})
        cache = data_cache.get('data_cache', None)
        batch_size = kwargs.pop('batch_size', None)
        downsample_rate = kwargs.pop('downsample_rate', None)
        tbptt_len = kwargs.pop('tbptt_len', None)

        if cache is not None:
            # Use cached data for training
            seq_tra = cache['seq_tra']
            kwargs['validation_data'] = cache['seq_val']
        else:
            # Generate training sequences
            seq_tra = self._generate_batches(x, y, 
                                            p_id_col=p_id_col,
                                            batch_size=batch_size,
                                            downsample_rate=downsample_rate,
                                            tbptt_len=tbptt_len,
                                            )
            
            x_val, y_val = kwargs.pop('validation_data')
            seq_val = self._generate_batches(x_val, y_val, 
                                            p_id_col=p_id_col,
                                            batch_size=batch_size,
                                            downsample_rate=downsample_rate,
                                            tbptt_len=tbptt_len,
                                            )

            kwargs['validation_data'] = seq_val
            new_cache = {'seq_tra': seq_tra, 'seq_val': seq_val}
            data_cache.update({'data_cache': new_cache})

        import numpy as np
        x_sample, y_sample = next(iter(seq_tra))  # Get one batch
        print(f"X shape: {np.array(x_sample).shape}")
        print(f"Y shape: {np.array(y_sample).shape}")

        return self.fit_generator(seq_tra, **kwargs)


    def fit_generator(self, seq, **kwargs):
        """Constructs a new model with `build_rnn_model(x_shape=(self.xZero,self.xOne))` & fits the model to Sequence via `fit()`.
        
        # Arguments
            seq : Sequence object
            **kwargs: dictionary arguments for `Sequential.fit()`
        
        # Returns
            history : Training history object
        """
        fit_args = kwargs.copy()

        # Ensure the correct model is used
        if build_rnn_model(x_shape=(self.xZero, self.xOne, self.xTwo)) is None:
            self.model = self.__call__(**kwargs)
        elif (not isinstance(build_rnn_model(x_shape=(self.xZero, self.xOne)), types.FunctionType) and
            not isinstance(build_rnn_model(x_shape=(self.xZero, self.xOne)), types.MethodType)):
            self.model = build_rnn_model(x_shape=(self.xZero, self.xOne))
        else:
            self.model = build_rnn_model(x_shape=(self.xZero, self.xOne))

        # Remove unnecessary arguments
        fit_args.pop('workers', None)
        fit_args.pop('use_multiprocessing', None)

        # Train the RNN model
        history = self.model.fit(seq, **fit_args)

        return history


    def predict(self, x, **kwargs):
        """Use this func to get a prediction for x. """
        assert isinstance(x, pd.DataFrame), \
            f'{self.__class__.__name__} needs pandas DataFrames as input'
        p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
        downsample_rate = kwargs.pop('downsample_rate', None)
        tbptt_len = kwargs.pop('tbptt_len', None)
        batch_size = kwargs['batch_size']
        batch_generation_cfg = {'p_id_col': p_id_col,
                                'batch_size': batch_size,
                                'downsample_rate': downsample_rate,
                                'tbptt_len': tbptt_len}
        x, sample_weights = self._generate_batches(x, **batch_generation_cfg)
        yhat = super().predict(x, **kwargs)

        if len(yhat.shape) < 3:
            if tbptt_len == 1:
                third_dim = 1 if len(yhat.shape) == 1 else yhat.shape[1]
                yhat = yhat.reshape(yhat.shape[0], 1, third_dim)
            elif len(yhat.shape) == 2:
                # single target model
                yhat = yhat.reshape(yhat.shape[0], yhat.shape[1], 1)
            else:
                raise ValueError('Something wrong with _yhat shape!')

        original_batch_size = batch_size // downsample_rate
        # check if prediction was on replicated data due to batch_size
        # this check relies on x's last feature being different in each profile
        is_replicated, num_profiles = \
            self.is_replicated(x[:original_batch_size, :, :])
        n_dummies = self.get_dummies_from_w_matrix(sample_weights, batch_size)

        # return yhat as 2-dim matrix
        # 3d due to tbptt length -> 2d


        profiles = []
        # revert breakdown due to tbptt
        for idx_b, n_dummy in enumerate(n_dummies):
            profile = np.vstack(yhat[idx_b + n, :, :] for n in
                                range(0, yhat.shape[0], batch_size))
            if n_dummy != 0:
                profile = profile[:-n_dummy, :]
            profiles.append(profile)

        assert len(profiles) == batch_size, \
            f'ping! {len(profiles)} != {batch_size}'

        non_downsampled_profiles = []
        # revert downsampling (by zipping appropriate profiles together)
        for p_i_sample in range(num_profiles):
            samples_list = \
                [profiles[p_i_sample+d_i] for d_i in
                          range(0, batch_size, original_batch_size)]
            max_len = len(max(samples_list, key=lambda x: len(x)))
            samples_stack = np.dstack(np.pad(s,
                   ((0, max_len - s.shape[0]), (0, 0)),
                   mode='constant', constant_values=np.nan)
                            for s in samples_list).transpose((2, 0, 1))
            samples_stack = samples_stack.reshape((-1, samples_stack.shape[2]),
                                                  order='F')
            samples_stack = samples_stack[~np.isnan(samples_stack)]\
                .reshape(-1, samples_stack.shape[1])
            non_downsampled_profiles.append(samples_stack)

        yhat = np.vstack(non_downsampled_profiles)
        return yhat

    def score(self, x, y, **kwargs):
        """This score func will return the loss"""
        # sample weight needed

        if kwargs.pop('score_directly', False):
            #  x = actual, y = prediction
            if np.any(np.isnan(y)):
                loss = 9999  # NaN -> const.
            else:
                loss = np.mean(K.eval(
                    self.model.loss_functions[0](K.cast(x, np.float32),
                                                 K.cast(y, np.float32))))
            print(f'Loss: {loss:.6} K²'),
            return loss
        else:
            p_id_col = kwargs.pop('p_id_col', 'p_id_col_not_found')
            downsample_rate = kwargs.pop('downsample_rate', None)
            tbptt_len = kwargs.pop('tbptt_len', None)
            batch_size = kwargs['batch_size']
            batch_generation_cfg = {'p_id_col': p_id_col,
                                    'batch_size': batch_size,
                                    'downsample_rate': downsample_rate,
                                    'tbptt_len': tbptt_len}
            x, sample_weights = \
                self._generate_batches(x, **batch_generation_cfg)
            y, _ = \
                self._generate_batches(y, **batch_generation_cfg)
            kwargs['sample_weight'] = sample_weights

            return super().score(x, y, **kwargs)

    @staticmethod
    def _generate_batches(x, y, p_id_col, batch_size, downsample_rate, tbptt_len):
        """Generates batches of data from the DataFrame."""
        if not isinstance(p_id_col, str):
            raise TypeError(f"Expected p_id_col to be a string, but got {type(p_id_col)}")

        if y is not None:
            # target vectors shall not contain p_id_col since they aren't
            # needed to build the sequences from loadprofile generator
            if p_id_col in y:
                y.drop([p_id_col], axis=1, inplace=True)
            _df = pd.concat([x, y], axis=1)
        else:
            _df = x


        p_ids = _df[p_id_col].astype(int).unique().tolist()
        
        # Filter clean profile IDs (assumes noise IDs are 3+ digits)
        clean_p_ids = [p_id for p_id in p_ids if len(str(p_id)) < 3]

        profile_dfs_l = [_df.loc[_df[p_id_col].astype(int) == int(p), :]
                            .reset_index(drop=True)
                            .drop(p_id_col, axis=1)
                        for p in clean_p_ids]

        max_len = len(max(profile_dfs_l, key=lambda x: len(x)))

        downsampled_max_len = np.ceil(max_len // downsample_rate)
        assert downsampled_max_len >= tbptt_len, \
            f"tbptt_len ({tbptt_len}) must be smaller than {downsampled_max_len}"

        div_factor = downsample_rate * tbptt_len
        max_len += (div_factor - (max_len % div_factor))
        max_len = int(max_len)

        original_batch_size = batch_size // downsample_rate
        if len(profile_dfs_l) < original_batch_size:
            profile_dfs_l = profile_dfs_l * original_batch_size
            profile_dfs_l = profile_dfs_l[:original_batch_size]

        assert len(profile_dfs_l) == original_batch_size, 'Batch size mismatch!'

        arr = np.concatenate([
            np.pad(p.values, ((0, max_len - len(p)), (0, 0)), 
                mode='constant', constant_values=np.nan)
            [np.newaxis, :] for p in profile_dfs_l
        ])

        arr = arr.reshape((batch_size, -1, arr.shape[2]), order='F')

        if max_len > tbptt_len:
            arr = np.concatenate([arr[:, n:n + tbptt_len, :] for n in
                                range(0, arr.shape[1], tbptt_len)])

        assert arr.shape[1] % tbptt_len == 0, 'Shape error after tbptt split!'

        nan_mask = np.isnan(arr[:, :, 0].reshape(arr.shape[:2]))
        _sample_weights = np.ones(nan_mask.shape)
        _sample_weights[nan_mask] = 0

        x = np.nan_to_num(arr).astype(np.float32)
        sample_weights = _sample_weights.astype(np.uint8)
        print('x_shape',x.shape)
        print('y.shape',y.shape)
        return x, y

    @staticmethod
    def is_replicated(batch_of_yhat):
        """Given one batch of yhat, check whether there are identical rows,
        which indicates a replicated dataset. This is the case for
        predictions on valset and testset, as these are usually fewer than
        batch_size. Note, that for RNNs batch_size = num train profiles."""
        first_row = batch_of_yhat[0, :, -1]  # first row, last feature
        is_replicated = False
        for i, row in enumerate(batch_of_yhat[1:, :, -1]):
            if np.array_equal(first_row, row):
                is_replicated = True
                break
        else:
            i += 1
        num_profiles = i + 1
        return is_replicated, num_profiles

    @staticmethod
    def get_dummies_from_w_matrix(weights, batch_size):
        """Scan weight matrix for zeros which denote the padded zeros that
        need to be chopped off at the end. Return List of dummies for each
        profile which may be downsampled"""
        max_profile_len = weights.shape[0] * weights.shape[1] // batch_size
        n_dummies_within_batch = \
            max_profile_len - np.sum(np.count_nonzero(weights[n:n+batch_size, :],
                                                      axis=1) for
                                     n in range(0, weights.shape[0], batch_size))
        return n_dummies_within_batch.astype(int)


# def build_rnn_model(x_shape=(100, 1, 10),
#                       arch='lstm',
#                     #   n_layers=1,
#                     #   n_units=64,
#                     #   kernel_reg=1e-9,
#                     #   activity_reg=1e-9,
#                     #   recurrent_reg=1e-9,
#                     #   bias_reg=1e-9,
#                     #   dropout_rate=0.5,
#                       optimizer='adam',
#                     #   gauss_noise_std=1e-3,
#                     #   lr_rate=1e-5,
#                       batch_size=128,
#                     #   clipnorm=1.,
#                     #   clipvalue=.5,
#                     #   loss='mse',
#                       ):
#     """Build function for a keras RNN model"""

#     arch_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    
#     opts_map = {'adam': Adam, 'nadam': Nadam,
#                 'adamax': Adamax, 'sgd': SGD,
#                 'rmsprop': RMSprop}

#     rnn_cfg = {
#         'units': int(n_units),
#         'input_shape': x_shape[1:],
#         # 'kernel_regularizer': regularizers.l2(kernel_reg),
#         # 'activity_regularizer': regularizers.l2(activity_reg),
#         # 'recurrent_regularizer': regularizers.l2(recurrent_reg),
#         # 'bias_regularizer': regularizers.l2(bias_reg),
#         # 'dropout': dropout_rate,
#         # 'recurrent_dropout': dropout_rate,
#         # 'return_sequences': (n_layers > 1),
#     }

#     # layernorm seems to not improve performance and in addition can't be
#     # serialized -> would need to be fixed in save_model()
#     layer_norm = False

#     def lstm_chrono_bias_initializer(_, *args, **kwargs):
#         """Chrono init for bias vector in LSTMs"""
#         f_init = \
#             ChronoInitializer(cfg.keras_cfg['max_time_step'])((int(n_units),),
#                                                               *args, **kwargs)
#         i_init = -tf.identity(f_init)

#         return tf.concat([
#             i_init,
#             f_init,
#             initializers.Zeros()((int(n_units) * 2,), *args, **kwargs),
#         ], axis=0)

#     def gru_chrono_bias_initializer(_, *args, **kwargs):
#         pass

#     if True:  # n_layers > 1:  # todo: always return sequences?
#         rnn_cfg['return_sequences'] = True
    
#     ANN = arch_dict[arch]

#     if arch == 'gru':
#         # compatibility with CUDNN_GRU
#         rnn_cfg['reset_after'] = True
#         rnn_cfg['recurrent_activation'] = 'sigmoid'
#     else:
#         rnn_cfg['implementation'] = 2  # gru would produce NANs with this

#     if arch == 'gru':
#         # bias init
#         # todo: Does GRU've been found to benefit from unit_forget_bias?
#         # is chrono init then helpful at all?
#         # rnn_cfg['bias_initializer'] = gru_chrono_bias_initializer
#         pass
#     elif arch == 'lstm':
#         rnn_cfg['unit_forget_bias'] = False
#         rnn_cfg['bias_initializer'] = lstm_chrono_bias_initializer

#     # create model
#     x = Input(shape=x_shape[1:])
#     x_before = x
#     if layer_norm:
#         x = LayerNormalization()(x)
#     y = ANN(**rnn_cfg)(x)
#     y = GaussianNoise(gauss_noise_std)(y)

#     x_dense = Dense(n_units, activation='relu')(x_before)

#     if 'res_' in arch:
#         y = Add([x_dense, y])

#     if n_layers > 1:
#         for i in range(n_layers-1):
#             rnn_cfg.pop('batch_input_shape', None)
#             y_before = y
#             if layer_norm:
#                 y = LayerNormalization()(y)
#             y = ANN(**rnn_cfg)(y)
#             y = GaussianNoise(gauss_noise_std)(y)

#             if 'res_' in arch:
#                 y = Add([y_before, y])
#     y = TimeDistributed(
#         Dense(len(cfg.data_cfg['Target_param_names'])))(y)

#     model = models.Model(outputs=y, inputs=x_before)

#     opt = opts_map[optimizer](learning_rate=lr_rate,
#                             #   clipnorm=clipnorm,  # grad normalization
#                               clipvalue=clipvalue)  # grad clipping
#     model.compile(optimizer=opt, loss=loss)
#     model.summary()
#     return model

from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, SimpleRNN, TimeDistributed, Dropout, 
    Add, LayerNormalization, GaussianNoise, Lambda
)
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, SGD, RMSprop
from tensorflow.keras import regularizers, models
import tensorflow as tf

def build_rnn_model(
    x_shape=(100, 1, 10),
    arch="lstm",
    n_layers=1,
    n_units=64,
    kernel_reg=1e-9,
    activity_reg=1e-9,
    recurrent_reg=1e-9,
    bias_reg=1e-9,
    dropout_rate=0.5,
    optimizer="adam",
    gauss_noise_std=1e-3,
    lr_rate=1e-5,
    batch_size=128,
    clipnorm=1.0,
    clipvalue=0.5,
    loss="mse",
):
    """Build function for a Keras RNN model"""

    arch_dict = {"lstm": LSTM, "gru": GRU, "rnn": SimpleRNN}
    opts_map = {"adam": Adam, "nadam": Nadam, "adamax": Adamax, "sgd": SGD, "rmsprop": RMSprop}

    RNNLayer = arch_dict.get(arch, LSTM)

    rnn_cfg = {
        "units": n_units,
        "kernel_regularizer": regularizers.l2(kernel_reg),
        "activity_regularizer": regularizers.l2(activity_reg),
        "recurrent_regularizer": regularizers.l2(recurrent_reg),
        "bias_regularizer": regularizers.l2(bias_reg),
        "dropout": dropout_rate,
        "recurrent_dropout": dropout_rate,
        "return_sequences": (n_layers > 1),
    }

    x = Input(shape=x_shape[1:])
    y = x

    # Stack RNN layers
    for i in range(n_layers):
        y = RNNLayer(**rnn_cfg)(y)

    # ✅ Apply Gaussian Noise before TimeDistributed (optional)
    y = GaussianNoise(gauss_noise_std)(y)

    # ✅ Ensure `y` has at least 3D shape before applying `TimeDistributed`
    if n_layers == 1:  
        y = Lambda(lambda x: tf.expand_dims(x, axis=1))(y)  # ✅ Use Lambda Layer instead of `tf.expand_dims()`

    # ✅ Correct `TimeDistributed` application
    y = TimeDistributed(Dense(x_shape[-1]))(y)

    model = models.Model(inputs=x, outputs=y)

    opt = opts_map[optimizer](learning_rate=lr_rate, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss=loss)
    model.summary()
    
    return model


def get_predefined_model():
    x = Input(shape=(1, 91), name='input_161')
    x_before = x
    y = LSTM(units=4, name='lstm_91')(x)
    x_dense = Dense(4, activation='relu', name='dense_321')(x_before)
    y = Add([x_dense, y], name='add_76')
    y = Flatten()(y)
    y = Dense(len(cfg.data_cfg['Target_param_names']), name='dense_322')(y)
    model = models.Model(outputs=y, inputs=x_before)

    model.compile(optimizer='adam', loss='mse')
    return model


class StateResetter(keras.callbacks.Callback):
    def __init__(self, p_start_list=[]):
        super().__init__()
        self.profile_start_indices = p_start_list

    def on_batch_begin(self, batch, logs={}):
        if batch in self.profile_start_indices:
            self.model.reset_states()


class NaNCatcher(keras.callbacks.Callback):
    NAN_CONST = float(9999)

    def on_epoch_end(self, epoch, logs=None):
        if np.any(np.isnan(logs.get('loss', np.NaN))):
            self._stop_training(logs)

        val_loss = logs.get('val_loss', None)
        if val_loss is not None:
            if np.any(np.isnan(val_loss)):
                self._stop_training(logs)

    def _stop_training(self, _logs):
        self.model.stop_training = True
        _logs['loss'] = self.NAN_CONST
        if 'val_loss' in _logs:
            _logs['val_loss'] = self.NAN_CONST
        print('Stop training due to nan output')
        self.model.history.history['nan_output'] = True


class ChronoInitializer(initializers.RandomUniform):
    """
    Chrono Initializer from the paper :
    [Can recurrent neural networks warp time? ](https://openreview.net/forum?id=SJcKhk-Ab)

    Source: https://github.com/titu1994/Keras-just-another-network-JANET/blob/master/chrono_initializer.py
    """

    def __init__(self, max_timesteps, seed=None):
        super().__init__(1., max_timesteps - 1, seed)
        self.max_timesteps = max_timesteps

    def __call__(self, shape, dtype=None):
        values = super().__call__(shape, dtype=dtype)
        return tf.math.log(values)

    def get_config(self):
        config = {
            'max_timesteps': self.max_timesteps
        }
        base_config = super(ChronoInitializer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def reshape_input_for_batch_train(x_set):
    assert isinstance(x_set, pd.DataFrame)
    x = x_set.values
    return np.reshape(x, (x.shape[0], 1, x.shape[1]))


get_custom_objects().update({'ChronoInitializer': ChronoInitializer,
                             'chrono_initializer': ChronoInitializer})