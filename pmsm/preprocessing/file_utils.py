import sqlite3
import uuid
import numpy as np
import pandas as pd
import time
import sys
import ast
import os
import re
from random import shuffle as shuffle_list
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import model_from_json
import preprocessing.config as cfg

sns.set()  # nicer graphics


def measure_time(func):
    """time measuring decorator"""
    def wrapped(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('took {:.3} seconds'.format(end_time-start_time))
        return ret
    return wrapped


def get_available_gpus():

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class LoadprofileGenerator(TimeseriesGenerator):
    """This is a customized version of keras TimeseriesGenerator. Its
    intention is to neglect strides and sampling rates but to incorporate
    iteration through several timeseries or loadprofiles of arbitrary length."""
    def __init__(self, data, targets, length, start_index=0,
                 shuffle=False, reverse=False, batch_size=128):
        # print(data)
        super().__init__(data, targets, length, start_index=start_index,
                         shuffle=shuffle, reverse=reverse,
                         end_index=len(data[0]), batch_size=batch_size)
        assert isinstance(data, list), 'data must be list of timeseries'
        if any(isinstance(i, pd.DataFrame) for i in self.data):
            self.data = [i.values for i in self.data]
        if any(isinstance(i, pd.DataFrame) for i in self.targets):
            self.targets = [i.values for i in self.targets]
        if self.shuffle:
            zippd = list(zip(self.data, self.targets))
            shuffle_list(zippd)  # inplace operation
            self.data, self.targets = list(zip(*zippd))
        # start index is the same for each profile
        # for each profile there's a different end_index
        self.end_index = [len(d)-1 for d in self.data]

        batches_per_profile = [(e - self.start_index + self.batch_size)//
                               self.batch_size for e in self.end_index]
        self.data_len = sum(batches_per_profile)
        self.batch_cumsum = np.cumsum(batches_per_profile)

    def __len__(self):
        return self.data_len

    def _empty_batch(self, num_rows):
        # shape of first profile suffices
        samples_shape = [num_rows, self.length]
        samples_shape.extend(self.data[0].shape[1:])
        targets_shape = [num_rows]
        targets_shape.extend(self.targets[0].shape[1:])
        return np.empty(samples_shape), np.empty(targets_shape)

    def __getitem__(self, index):
        # index is the enumerated batch index starting at 0
        # find corresponding profile

        p_idx = np.nonzero(index < self.batch_cumsum)[0][0]
        prev_sum = 0 if p_idx == 0 else self.batch_cumsum[p_idx-1]

        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index[p_idx] + 1,
                size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * (index - prev_sum)
            rows = np.arange(i, min(i + self.batch_size,
                                    self.end_index[p_idx] + 2))
            # +2 to get the last element, too
        samples, targets = self._empty_batch(len(rows))
        for j, row in enumerate(rows):
            indices = range(row - self.length, row)
            samples[j] = self.data[p_idx][indices]
            targets[j] = self.targets[p_idx][row-1]
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets


class Report:
    """Summary of an experiment/trial"""
    TARGET_SCHEME = cfg.data_cfg['db_target_scheme']
    TABLE_SCHEMES = \
        {'predictions': ['id text', 'idx int'] +
                        ['{} real' for _ in range(len(TARGET_SCHEME))] +
                        ['{} real' for _ in range(len(TARGET_SCHEME))],
         'meta_experiments': ['id text', 'target text', 'testset text',
                              'score real', 'loss_metric text', 'seed text',
                              'scriptname text', 'start_time text',
                              'end_time text', 'config text']
         }

    def __init__(self, uid, seed,
                 score=None, yhat=None, actual=None, history=None,
                 used_loss=None, model=None):
        self.score = score
        self.yhat_te = yhat
        self.actual = actual
        self.history = history
        self.uid = uid
        self.seed = seed
        self.yhat_tr = None
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.used_loss = used_loss
        self.model = model
        self.cfg_blob = {}

    def save_to_db(self):
        if cfg.data_cfg['save_predictions'] and not cfg.debug_cfg['DEBUG']:
            cols = self.yhat_te.columns.tolist()
            assert all(t in self.TARGET_SCHEME for t in cols), \
                'config has unknown target specified'
            # fill up missing targets up to TARGET_SCHEME
            df_to_db = self.yhat_te.copy()
            df_to_db = df_to_db.assign(**{t: 0 for t in self.TARGET_SCHEME
                                          if t not in cols})
            df_to_db = df_to_db.loc[:, self.TARGET_SCHEME]  # reorder cols

            gtruth_to_db = self.actual.copy()
            gtruth_to_db = gtruth_to_db.assign(**{t: 0 for t in
                                                  self.TARGET_SCHEME
                                                  if t not in cols})
            gtruth_to_db = gtruth_to_db.loc[:, self.TARGET_SCHEME]\
                .rename(columns={t:t+'_gtruth' for t in gtruth_to_db.columns})

            df_to_db = pd.concat([df_to_db, gtruth_to_db], axis=1)

            try:
                with sqlite3.connect(cfg.data_cfg['db_path']) as con:
                    print("Connected to database successfully!")  # ✅ Connection check
                    # predictions
                    table_name = 'predictions'
                    table_scheme = self.TABLE_SCHEMES[table_name]
                    query = "CREATE TABLE IF NOT EXISTS " + \
                            "{}{}".format(table_name, tuple(table_scheme))\
                                .replace("'", "")
                    query = query.format(*df_to_db.columns)
                    con.execute(query)

                    df_to_db['id'] = self.uid
                    df_to_db['idx'] = self.yhat_te.index

                    entries = [tuple(x) for x in np.roll(df_to_db.values,
                                                        shift=2, axis=1)]
                    query = f'INSERT INTO {table_name} ' + \
                            'VALUES ({})'.format(
                                ', '.join('?' * len(df_to_db.columns)))
                    con.executemany(query, entries)

                    # meta experiments
                    table_name = 'meta_experiments'
                    table_scheme = self.TABLE_SCHEMES[table_name]
                    query = "CREATE TABLE IF NOT EXISTS " + \
                            "{}{}".format(table_name, tuple(table_scheme))\
                                .replace("'", "")
                    con.execute(query)

                    config_blob = {**cfg.data_cfg, **cfg.keras_cfg, **cfg.lgbm_cfg}
                    if hasattr(self.model, 'sk_params'):
                        config_blob['sk_params'] = self.model.sk_params

                    entry = (self.uid,
                        str(cfg.data_cfg['Target_param_names']),
                        str(cfg.data_cfg['testset']),
                        str(self.score),
                        cfg.data_cfg['loss'],
                        str(self.seed),
                        os.path.basename(sys.argv[0]),
                        self.start_time,
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        str(config_blob),
                        )
                    query = f'INSERT INTO {table_name} VALUES {entry}'
                    con.execute(query)
                    print(f'Predictions and meta of model with uuid {self.uid} saved to db.')

            except sqlite3.Error as e:
                print("Database connection failed:", e)  # ❌ Error handling


    def save_model(self):
        if not cfg.debug_cfg['DEBUG'] and self.model is not None:
            self.model.save(self.uid)
            print(f'Model arch and weights dumped for {self.uid}.')

    def load_model(self):
        path = os.path.join(cfg.data_cfg['model_dump_path'],
                            self.uid+'_arch.json')
        with open(path, 'r') as f:
            self.model = model_from_json(f.read())
        self.model.compile(optimizer='adam', loss='mse')
        self.model.load_weights(os.path.join(cfg.data_cfg['model_dump_path'],
                                             self.uid+'_weights.h5'))
        return self

    @classmethod
    def load(clf, uid, truncate_at=None):
        """Return a Report object from uid. Uid must exist in database."""

        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            query = """SELECT * FROM predictions WHERE id=?"""
            pred_table = pd.read_sql_query(query, con, params=(uid,))
            query = """SELECT * FROM meta_experiments WHERE id=?"""
            meta_table = pd.read_sql_query(query, con, params=(uid,))
        cfg.data_cfg['Target_param_names'] = \
            ast.literal_eval(meta_table.target[0])  # str of list -> list
        cfg.data_cfg['testset'] = ast.literal_eval(meta_table.testset[0])
        target_cols = cfg.data_cfg['Target_param_names']
        yhat = pred_table.loc[:, target_cols]
        actual = pred_table.loc[:, [t+'_gtruth' for t in target_cols]]
        actual = actual.rename(columns=lambda c: c.replace('_gtruth', ''))

        score = meta_table.score[0]
        seed = meta_table.seed[0]
        used_loss = meta_table.loss_metric[0]
        cfg_blob = meta_table.config[0]

        if truncate_at is not None:
            actual = actual.iloc[:truncate_at, :]
            yhat = yhat.iloc[:truncate_at, :]

        report = clf(uid, seed, score, yhat, actual, used_loss=used_loss)
        # fix the wrongly placed mse func obj in the cfg blob with string
        report.cfg_blob = eval(re.sub('<[A-Za-z0-9_]+(?:\s+[a-zA-Z0-9_]+)*>',
                                      "'mse'", cfg_blob))
        try:
            report.load_model()
        except FileNotFoundError:
            print(f'Couldnt load model {uid}. '
                  f'Weight or architecture file not found.')
        return report

    import matplotlib.pyplot as plt

    def plot(self, show=True):
        linestyles = ['-', '--', ':', '-.']
        
        # Set up figure with appropriate size
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
        
        plot_idx = 0

        # Plot Training/Validation Loss
        if self.history is not None:
            history = self.history.history
            axes[plot_idx].plot(history['loss'], label='Train Loss', linestyle='-')
            axes[plot_idx].plot(history['val_loss'], label='Validation Loss', linestyle='--')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel(f'{self.used_loss} in K²')
            axes[plot_idx].set_title(f'Training/Validation Score over Epochs (Experiment {self.uid})')
            axes[plot_idx].legend(loc='upper right', fontsize=8)
            plot_idx += 1

        # Plot Test Set Predictions vs Ground Truth
        for i, c in enumerate(self.actual):
            axes[plot_idx].plot(self.actual[c], alpha=0.6, color='darkorange', label=f'Ground Truth {c}', linestyle=linestyles[i])
        for i, c in enumerate(self.yhat_te):
            axes[plot_idx].plot(self.yhat_te[c], lw=2, color='navy', label=f'Predicted {c}', linestyle=linestyles[i])
        
        axes[plot_idx].set_xlabel('Time (s)')
        axes[plot_idx].set_ylabel('Temperature (°C)')
        axes[plot_idx].set_title(f'Prediction vs Ground Truth (Experiment {self.uid})')
        axes[plot_idx].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plot_idx += 1

        # Plot Prediction Error
        for i, c in enumerate(self.actual):
            axes[plot_idx].plot(self.yhat_te[c] - self.actual[c], color='red', label=f'Prediction Error {c}', linestyle=linestyles[i])
        
        axes[plot_idx].set_xlabel('Time (s)')
        axes[plot_idx].set_ylabel('Temperature (K)')
        axes[plot_idx].set_title(f'Prediction Error (Experiment {self.uid})')
        axes[plot_idx].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

        # Adjust layout for better visibility
        plt.tight_layout()

        # Plot Performance on Trainset (Separate Figure)
        if self.yhat_tr is not None:
            y_tr, yhat_tr = self.yhat_tr
            plt.figure(figsize=(8, 4))
            plt.plot(y_tr, alpha=0.6, color='darkorange', label='Ground Truth')
            plt.plot(yhat_tr, lw=2, color='navy', label='Prediction')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Training Set Performance (Experiment {self.uid})')
            plt.legend()

        # Show the plots
        if show:
            plt.show()

    def paper_1_plot_testset_performance(self):
        sns.set_context('paper')
        sns.set_style('whitegrid')

        cols_to_plot = cfg.data_cfg['Target_param_names']  
        linestyles = ['-', '--', ':', '-.']
        colors = ['green', 'navy', 'red', 'magenta', 'darkorange', 'yellow']

        param_map = {
            'pm': '{PM}',
            'stator_tooth': '{ST}',
            'stator_yoke': '{SY}',
            'stator_winding': '{SW}'
        }

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Function to format the plots
        def _format_plot(ax):
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Temperature (°C)')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(-1000, np.around(len(self.actual), -3) + 300)
            tcks = np.arange(0, np.around(len(self.actual), -3), 7200)
            ax.set_xticks(tcks)
            ax.set_xticklabels(tcks // 7200)

        # **Plot Predictions vs Ground Truth**
        ax1 = axes[0]
        ax1.set_title('Prediction vs Ground Truth')

        for i, c in enumerate(self.actual):
            ax1.plot(self.actual[c], alpha=0.6, color='green', label=rf'$\theta_{{{param_map[c]}}}$', linestyle=linestyles[i])
        for i, c in enumerate(self.yhat_te):
            ax1.plot(self.yhat_te[c], lw=2, color='navy', label=rf'$\hat \theta_{{{param_map[c]}}}$', linestyle=linestyles[i])

        _format_plot(ax1)

        # **Plot Prediction Error**
        ax2 = axes[1]
        ax2.set_title('Prediction Error')

        for i, c in enumerate(self.actual):
            ax2.plot(self.yhat_te[c] - self.actual[c], color=colors[i], label=fr'Prediction Error $\theta_{{{param_map[c]}}}$')

        _format_plot(ax2)

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()

    def presentation_plot_testset_performance(self, trunc=True):
        sns.set_context('talk')
        sns.set_style('whitegrid')

        # Truncate data if required (avoid modifying original)
        if trunc:
            truncate_at = 40092
            yhat_te_trunc = self.yhat_te.iloc[:truncate_at, :].copy()
            actual_trunc = self.actual.iloc[:truncate_at, :].copy()
        else:
            yhat_te_trunc = self.yhat_te
            actual_trunc = self.actual

        param_map = {
            'pm': '{PM}',
            'stator_tooth': '{ST}',
            'stator_yoke': '{SY}',
            'stator_winding': '{SW}'
        }

        n_targets = len(actual_trunc.columns)
        fig, axes = plt.subplots(n_targets, 2, figsize=(12, 1.8 * n_targets))

        def _format_plot(ax, y_lbl='temp', x_lbl=True):
            """Helper function to format plots."""
            ax.set_xlabel('Time in h' if x_lbl else '')
            
            if y_lbl in param_map:
                ax.set_ylabel(r'$\theta_{}$ in °C'.format(param_map[y_lbl]))
            elif y_lbl == 'motor_speed':
                ax.set_ylabel('Motor speed in 1/min')
            elif y_lbl.startswith('i_'):
                ax.set_ylabel('Current in A')
            else:
                ax.set_ylabel('Temperature in °C')

            ax.set_xlim(-1000, np.around(len(actual_trunc), -3) + 300)
            tcks = np.arange(0, np.around(len(actual_trunc), -3), 7200)
            ax.set_xticks(tcks)
            ax.set_xticklabels(tcks // 7200 if x_lbl else [])

        for i, c in enumerate(actual_trunc):
            diff = yhat_te_trunc[c] - actual_trunc[c]
            
            # **Plot Prediction vs Ground Truth**
            ax1 = axes[i, 0] if n_targets > 1 else axes[0]
            ax1.set_title('Prediction and Ground Truth') if i == 0 else None
            ax1.plot(actual_trunc[c], color='green', label=rf'$\theta_{{{param_map[c]}}}$', linestyle='-')
            ax1.plot(yhat_te_trunc[c], lw=2, color='navy', label=rf'$\hat \theta_{{{param_map[c]}}}$', linestyle='-')
            _format_plot(ax1, y_lbl=c, x_lbl=i >= 5)
            ax1.legend(loc='upper left', fontsize=8)

            # Display Mean Squared Error (MSE)
            mse_text = f'MSE: {(diff ** 2).mean():.2f} K²'
            ax1.text(0.6, 0.9, mse_text, bbox={'facecolor': 'white'}, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='center')

            # **Plot Prediction Error**
            ax2 = axes[i, 1] if n_targets > 1 else axes[1]
            ax2.set_title('Prediction Error') if i == 0 else None
            ax2.plot(diff, color='red', label=rf'Prediction Error $\theta_{{{param_map[c]}}}$')
            _format_plot(ax2, y_lbl=c, x_lbl=i >= 5)
            ax2.legend(loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.show()

    def paper_0_plot_testset_performance(self, testset_x, trunc=True):
        sns.set_context('paper')
        sns.set_style('whitegrid')

        # Truncate data if needed
        if trunc:
            truncate_at = 40092
            yhat_te_trunc = self.yhat_te.iloc[:truncate_at, :].copy()
            actual_trunc = self.actual.iloc[:truncate_at, :].copy()
        else:
            yhat_te_trunc = self.yhat_te
            actual_trunc = self.actual

        param_map = {'pm': '{PM}',
                    'stator_tooth': '{ST}',
                    'stator_yoke': '{SY}',
                    'stator_winding': '{SW}'}

        input_param_map = {'motor_speed': 'Motor speed',
                        'coolant': 'Coolant temperature',
                        'i_q': 'q-Axis current',
                        'i_d': 'd-Axis current'}

        n_targets = len(actual_trunc.columns)
        n_inputs = len(input_param_map)
        
        fig, axes = plt.subplots(n_targets + 2, 2, figsize=(12, 1.5 * (n_targets + 2)))

        def _format_plot(ax, y_lbl='temp', x_lbl=True):
            """Helper function to format plots."""
            ax.set_xlabel('Time in h' if x_lbl else '')

            if y_lbl in param_map:
                ax.set_ylabel(r'$\theta_{}$ in °C'.format(param_map[y_lbl]))
            elif y_lbl == 'motor_speed':
                ax.set_ylabel('Motor speed in 1/min')
            elif y_lbl.startswith('i_'):
                ax.set_ylabel('Current in A')
            else:
                ax.set_ylabel('Temperature in °C')

            ax.set_xlim(-1000, np.around(len(actual_trunc), -3) + 300)
            tcks = np.arange(0, np.around(len(actual_trunc), -3), 7200)
            ax.set_xticks(tcks)
            ax.set_xticklabels(tcks // 7200 if x_lbl else [])

        for i, c in enumerate(actual_trunc):
            diff = yhat_te_trunc[c] - actual_trunc[c]
            
            # **Plot Prediction vs Ground Truth**
            ax1 = axes[i, 0]
            ax1.set_title('Prediction and Ground Truth') if i == 0 else None
            ax1.plot(actual_trunc[c], color='green', label=rf'$\theta_{{{param_map[c]}}}$', linestyle='-')
            ax1.plot(yhat_te_trunc[c], lw=2, color='navy', label=rf'$\hat \theta_{{{param_map[c]}}}$', linestyle='-')
            _format_plot(ax1, y_lbl=c, x_lbl=i >= n_targets - 2)
            ax1.legend(loc='lower right', fontsize=8)

            # Display Mean Squared Error (MSE)
            mse_text = f'MSE: {(diff ** 2).mean():.2f} K²'
            ax1.text(0.6, 0.9, mse_text, bbox={'facecolor': 'white'}, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='center')

            # **Plot Prediction Error**
            ax2 = axes[i, 1]
            ax2.set_title('Prediction Error') if i == 0 else None
            ax2.plot(diff, color='red', label=rf'Prediction Error $\theta_{{{param_map[c]}}}$')
            _format_plot(ax2, y_lbl=c, x_lbl=i >= n_targets - 2)
            ax2.legend(loc='lower center', fontsize=8)

            # Display Maximum Absolute Error (L∞ norm)
            linf_text = r'$L_{\infty}$: {:.2f} K'.format(diff.abs().max())
            ax2.text(0.5, 0.9, linf_text, bbox={'facecolor': 'white'}, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='center')

        # **Plot Input Parameters**
        for i, (c, title) in enumerate(input_param_map.items()):
            y_lbl = 'temp' if c in ['ambient', 'coolant'] else c
            ax = axes[n_targets + i, 0]
            ax.set_title(title)
            ax.plot(testset_x[c], color='g')
            _format_plot(ax, y_lbl=y_lbl, x_lbl=i < 2)

        plt.tight_layout()
        plt.show()

class TrialReports:
    """Manages a list of reports"""
    def __init__(self, seed=0, reports=None):
        if reports is not None:
            assert isinstance(reports, list), 'ping!'
            self.reports = reports
        else:
            self.reports = []
        self.seed = seed
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.end_time = '-1'
        self.ensemble_score = -1.0
        self.data_cache = {}

    def __add__(self, report):
        assert isinstance(report, Report), 'ping!'
        self.reports.append(report)
        return self

    def __str__(self):
        self.print()
        return '\n'

    def get_scores(self):
        return [r.score for r in self.reports]

    def get_mean_score(self):
        return np.mean(self.get_scores())

    def get_uids(self):
        return [r.uid for r in self.reports]

    def get_predictions_te(self):
        return [r.yhat_te for r in self.reports]

    def conduct(self, n_trials):
        """Generator function to conduct trials sequentially."""
        for i in range(n_trials):
            trial_seed = self.seed + i
            model_uuid = str(uuid.uuid4())[:6]
            print('model uuid: {}, seed: {}'.format(model_uuid, trial_seed))
            np.random.seed(trial_seed)
            tf.random.set_seed(trial_seed)
            report = Report(uid=model_uuid, seed=trial_seed)

            yield report
            report.save_to_db()
            report.save_model()
            self.reports.append(report)
        # get ensemble performance
        self.ensemble_score = \
            mean_squared_error(self.reports[0].actual,
                               np.mean(np.dstack([r.yhat_te for r in self.reports]), axis=2))
        self.print()

    def print(self, plot=True):
        """Print summary of trial performances.
        Expects a list of dictionaries with details about the conducted trials.
        If configured, best model will be plotted as well.

        Confidence Interval of 95% for Metric Mean is constructed by
        t-distribution instead of normal, since sample size is most of the time
        < 30
        """
        # summary statistic
        scores = np.asarray(self.get_scores())
        bmr = best_model_report = self.reports[np.argmin(scores)]

        print('')
        print('#'*20)

        print(
f"""Performance Report
# trials: {len(scores)}
mean MSE: {scores.mean():.6} K² +- {stats.t.ppf(1-0.025, len(scores)):.3} K²
std MSE: {scores.std():.6} K²

Best Model: uuid {bmr.uid}; seed {bmr.seed}; score {bmr.score:.6} K²
Ensemble Score: {self.ensemble_score:.6} K²
"""
        )

        print('#'*20)

        # plot best model performance
        if cfg.plot_cfg['do_plot'] and plot:
            try:
                bmr.plot()
            except Exception:
                print("Plotting failed..")

    @staticmethod
    def conduct_step(model_func, seed, init_params, fit_params,
                      predict_params, inverse_params, dm):

        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}, seed: {}'.format(model_uuid, seed))
        np.random.seed(seed)
        tf.random.set_seed(seed)
        report = Report(uid=model_uuid, seed=seed)

        model = model_func(**init_params)

        report.history = model.fit(**fit_params)
        report.model = model

        if report.history.history.get('nan_output', False):
            # training failed
            report.actual = dm.inverse_transform(**inverse_params).iloc[:2, :]
            report.yhat_te = report.actual.copy()
            report.score = float(report.history.history['loss'][-1])

        else:
            # training successful
            try:
                report.yhat_te = model.predict(**predict_params)
                report.actual = dm.inverse_transform(**inverse_params)

                inverse_params_copy = inverse_params.copy()  # important!
                inverse_params_copy['df'] = report.yhat_te
                report.yhat_te = dm.inverse_transform(**inverse_params_copy)

                report.score = model.score(report.actual, report.yhat_te,
                                           score_directly=True)
                report.save_model()
            except ValueError:
                print('ValueError on this config:\n {}'.format({
                    **init_params, **fit_params, **predict_params}))
                raise

        report.save_to_db()
        report.print()
        return report


class HyperparameterSearchReport:
    """Manages a list of TrialReports. This class loads tables from a sqlite
     database with a certain scheme and performs some analysis. More
     specifically, the list of loaded trial-reports pertain to those generated
     during a certain hyperparameter search with hp_tune_xx.py."""

    bayes_col_filter = ['n_iter', 'model_uids', 'mean_score', 'best_score',
                        'start_time', 'end_time']

    def __init__(self):
        self.hp_searches = {}
        print('Reading', cfg.data_cfg['db_path'], '..')
        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            query = """SELECT * FROM meta_experiments"""
            self.meta_tab = pd.read_sql_query(query, con)
            query = """SELECT * FROM bayes_opt_results"""
            self.bayes_tab = pd.read_sql_query(query, con)

        assert not self.meta_tab.id.duplicated().any(), \
            'Duplicated ID found! -> {}. 6 digits too few?'.format(
                self.meta_tab.id[self.meta_tab.id.duplicated()])

    def read_search(self, hp_search_uid, verbose=True):
        tab = (self.bayes_tab
               .loc[self.bayes_tab.bayes_search_id == hp_search_uid,
                    self.bayes_col_filter]
               .sort_values(by='n_iter', axis=0)
               .reset_index(drop=True))

        # get runtime
        time_format = "%Y-%m-%d %H:%M"
        runtime = (pd.to_datetime(tab.end_time, format=time_format) -
                   pd.to_datetime(tab.start_time, format=time_format)).sum()
        if verbose:
            print(f'Runtime of experiment {hp_search_uid}: {runtime}')

        # get model ids and their score std
        model_uids = (tab['model_uids'].astype('object')
                      .apply(eval).apply(pd.Series).stack()
                      .reset_index(drop=True, level=-1)
                      .rename("id").rename_axis('n_iter').reset_index())

        # merge corresponding single scores from meta_table
        model_uids = pd.merge(model_uids, self.meta_tab, how='left', on='id')

        # get id of best model
        best_model_id = model_uids.at[model_uids.score.idxmin(), 'id']

        # get std per iter
        grp = model_uids[['n_iter', 'score', 'id']].groupby('n_iter')
        grp_std = grp['score'].std().reset_index()\
                   .rename(columns={'score': 'std_score'})
        grp_best_model_id = grp['id'].min().reset_index()\
            .rename(columns={'id': 'best_model_id'})

        tab = pd.merge(tab, grp_std, how='left', on='n_iter')
        tab = pd.merge(tab, grp_best_model_id, how='left', on='n_iter')

        self.hp_searches[hp_search_uid] = tab
        return tab

    def get_best_model_and_score(self, hp_search_uid, verbose=True):
        assert hp_search_uid in self.hp_searches, \
            f'please load hp search {hp_search_uid} first'

        tab = self.hp_searches[hp_search_uid]
        best_model_id = tab.at[tab.best_score.idxmin(), 'best_model_id']
        best_score = tab.best_score.min()
        if verbose:
            print(f'In HP Search {hp_search_uid}: Best model: {best_model_id} '
                  f'with score: {best_score}')
        return best_model_id, best_score

    def plot_convergence(self, hp_search_uid, title=''):
        assert hp_search_uid in self.hp_searches, \
            f'please load hp search {hp_search_uid} first'
        tab = self.hp_searches[hp_search_uid]
        plt.plot(tab.mean_score, color='navy', label='mean')
        plt.fill_between(np.arange(len(tab)), tab.mean_score - tab.std_score,
                         tab.mean_score + tab.std_score,
                         alpha=0.4, label='standard deviation', color='#99ff99')
        plt.plot(tab.best_score, '--', color='navy', label='min (best)')

        _, best_score = self.get_best_model_and_score(hp_search_uid, verbose=0)

        # mark best iter in red
        plt.plot(np.argmin(tab.best_score.values), best_score, 'Xr')

        plt.text(0.7, 0.9, bbox={'facecolor': 'white'},
                 transform=plt.gca().transAxes,
                 s='Global best MSE: ' + f'{best_score:.2f} ' +
                   r'$\mathrm{K^2}$', verticalalignment='top',
                 horizontalalignment='center')
        tcks = np.arange(len(tab))[::10]
        plt.xticks(tcks, tcks)
        plt.yscale('log')
        plt.ylim(0.8, 200)
        plt.xlim(-0.5, len(tab.mean_score) + 1)
        plt.xlabel('search iteration')
        plt.ylabel(r'MSE in $\mathrm{K^2}$')
        plt.title(f'Bayesian Optimization Over {title} Temperatures')
        plt.tight_layout()
        plt.legend(loc='upper left', frameon=True)

    def plot_best_models_performance(self, uid_rot, uid_sta):
        from preprocessing.data import LightDataManager

        model_uid_rot, _ = self.get_best_model_and_score(uid_rot)
        model_uid_sta, _ = self.get_best_model_and_score(uid_sta)

        print('plot performance of (rotor)', model_uid_rot, 'and (stator)',
              model_uid_sta)
        truncate_at = None  # 40092  # oder None
        report_rot = Report.load(model_uid_rot, truncate_at=truncate_at)
        report_sta = Report.load(model_uid_sta, truncate_at=truncate_at)

        if report_rot.model is not None:
            print(f'rotor model parameters: {report_rot.model.count_params()}')
        if report_sta.model is not None:
            print(f'stator model parameters: {report_sta.model.count_params()}')

        report = Report('xxx', 1,
                        score=np.average([report_rot.score,
                                          report_sta.score],
                                         weights=[1, 3]),
                        yhat=pd.concat([report_rot.yhat_te,
                                        report_sta.yhat_te],
                                       axis=1),
                        actual=pd.concat([report_rot.actual,
                                          report_sta.actual],
                                         axis=1))
        dm = LightDataManager(cfg.data_cfg['file_path'], standardize=False)
        test_set_x = dm.tst_df[dm.x_cols]
        report.paper_0_plot_testset_performance(test_set_x, trunc=True)