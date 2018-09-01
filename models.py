# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
from typing import Callable, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm

import config
import preprocessing
import utils


def get_y_train():
    return pd.read_csv(config.path + 'application_train.csv.zip').TARGET


def run_autoencoder(X_train_scaled, X_test_scaled,
                    units_1=1000, units_2=1000, units_3=None, learning_rate=0.001, epochs=300, batch_size=128,
                    noise_begin=0.6, noise_end=0.1, noise_reduce_epoch=200, noise_reduce_end_epoch=290,
                    verbose_eval=50):
    """Run a scheduled denoising Autoencoder. The rate of noise is decreased during training. The type of noise is
    InputSwapNoise as described in https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629.
    Frequently evaluates the activations using a logistic regressiong each `verbose_eval` epoch.

    :param X_train_scaled: Scaled training data.
    :param X_test_scaled: Scaled test data.
    :param units_1: Number of units in first layer.
    :param units_2: Number of units in second layer.
    :param units_3: Number of units in third layer.
    :param learning_rate: Learning rate.
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :param noise_begin: Noise rate at the beginning.
    :param noise_end: Noise rate at the end.
    :param noise_reduce_epoch: Epoch where reducing the noise starts.
    :param noise_reduce_end_epoch: Epoch where reducing the noise ends.
    :param verbose_eval: Number of epochs between the current status of the Autoencoder is printing.
    :return: Tuple with training and test activations and scores of the logistic regressiong.
    """
    assert noise_reduce_end_epoch > noise_reduce_epoch, (
        'noise_reduce_end_epoch should be larger than noise_reduce_epoch')
    assert epochs > noise_reduce_epoch, 'Number of epochs should be larger than noise_reduce_end_epoch.'

    lr = LogisticRegression(penalty='l2', dual=False, tol=1e-1, C=0.005,
                            fit_intercept=True, intercept_scaling=1, class_weight=None,
                            random_state=1, solver='sag', max_iter=100,
                            multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)

    X_scaled = pd.concat([X_train_scaled, X_test_scaled]).reset_index(drop=True)

    units_layer_1 = units_1
    units_layer_2 = units_2
    units_layer_3 = units_3

    inputswapnoise_ratio = noise_begin
    noise_reduce_per_epoch = (noise_begin - noise_end) / (noise_reduce_end_epoch - noise_reduce_epoch)
    print('noise reduce after epoch {}: {:0.3f}'.format(noise_reduce_epoch, noise_reduce_per_epoch))

    X_true = tf.placeholder(dtype=tf.float32, shape=[None, X_train_scaled.shape[1]])
    print('Shape: ', X_train_scaled.shape)

    X = tf.placeholder(dtype=tf.float32, shape=[None, X_train_scaled.shape[1]])
    b_train_autoencoder = tf.placeholder(dtype=tf.bool)

    DAE_layer_1 = tf.layers.Dense(units=units_layer_1, activation=tf.nn.leaky_relu)(X)
    DAE_layer_2 = tf.layers.Dense(units=units_layer_2, activation=tf.nn.leaky_relu)(DAE_layer_1)
    if units_layer_3 is None:
        DAE_layer_final = tf.layers.Dense(units=X_train_scaled.shape[1])(DAE_layer_2)
    else:
        DAE_layer_3 = tf.layers.Dense(units=units_layer_3, activation=tf.nn.leaky_relu)(DAE_layer_2)
        DAE_layer_final = tf.layers.Dense(units=X_train_scaled.shape[1])(DAE_layer_3)

    DAE_loss = tf.losses.mean_squared_error(labels=X_true, predictions=DAE_layer_final)
    DAE_optimizer = tf.train.AdamOptimizer(learning_rate)
    DAE_train = DAE_optimizer.minimize(DAE_loss)

    lst_scores_logreg = {}

    init = tf.global_variables_initializer()

    print('Starting training...')
    with tf.Session() as session:

        session.run(init)
        print('Initialization finished...\n')

        for epoch in range(epochs):

            if epoch >= noise_reduce_epoch and epoch < noise_reduce_end_epoch:
                inputswapnoise_ratio -= noise_reduce_per_epoch

            # Shuffle and add InputSwapNoise
            X_shuffled = X_scaled.loc[:, DAE_features].sample(frac=1)
            X_noise = X_shuffled.copy()
            for col in X_noise.columns:
                srs_mask = np.random.rand((X_noise.shape[0])) < inputswapnoise_ratio
                srs_sample = X_noise.loc[:, col].sample(frac=1, replace=True)
                X_noise.loc[srs_mask, col] = srs_sample.iloc[srs_mask].values

            lst_loss_values = []

            for batch in range(X_noise.shape[0] // batch_size):
                X_noise_batch = X_noise.iloc[batch * batch_size:(batch + 1) * batch_size, :]
                X_true_batch = X_shuffled.iloc[batch * batch_size:(batch + 1) * batch_size, :]

                _, loss = session.run((DAE_train, DAE_loss), feed_dict={X: X_noise_batch, X_true: X_true_batch,
                                                                        b_train_autoencoder: True})
                lst_loss_values.append(loss)

            del X_noise
            gc.collect()

            if (epoch % verbose_eval == 0) or (epoch == epochs - 1) or (epoch < 2):

                train_activations_1 = session.run(DAE_layer_1, feed_dict={X: X_train_scaled,
                                                                          b_train_autoencoder: False})

                train_activations_2 = session.run(DAE_layer_2, feed_dict={X: X_train_scaled,
                                                                          b_train_autoencoder: False})
                if units_layer_3 is None:
                    scores_cv_lr = cross_val_score(lr,
                                                   np.concatenate([train_activations_1, train_activations_2], axis=1),
                                                   y,
                                                   groups=None, scoring='roc_auc', cv=None,
                                                   n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
                else:
                    train_activations_3 = session.run(DAE_layer_3, feed_dict={X: X_train_scaled,
                                                                              b_train_autoencoder: False})
                    scores_cv_lr = cross_val_score(lr, np.concatenate(
                        [train_activations_1, train_activations_2, train_activations_3], axis=1), y,
                                                   groups=None, scoring='roc_auc', cv=None,
                                                   n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
                score_logreg = np.mean(scores_cv_lr)

                del train_activations_1, train_activations_2
                if units_layer_3 is not None:
                    del train_activations_3
                gc.collect()

                lst_scores_logreg[epoch] = score_logreg
                print('Epoch {} \tMSE: {:0.5f}, layer_1_2 score:{:0.4f}, noise: {:0.2f}' \
                      .format(epoch, np.mean(lst_loss_values), score_logreg, inputswapnoise_ratio))

        X_train_autoencoded = session.run(DAE_layer_final, feed_dict={X: X_train_scaled,
                                                                      b_train_autoencoder: False})
        X_activations_layer_1 = session.run(DAE_layer_1, feed_dict={X: X_train_scaled,
                                                                    b_train_autoencoder: False})
        X_activations_layer_2 = session.run(DAE_layer_2, feed_dict={X: X_train_scaled,
                                                                    b_train_autoencoder: False})
        if units_layer_3 is not None:
            X_activations_layer_3 = session.run(DAE_layer_3, feed_dict={X: X_train_scaled,
                                                                        b_train_autoencoder: False})

        X_test_autoencoded = session.run(DAE_layer_final, feed_dict={X: X_test_scaled,
                                                                     b_train_autoencoder: False})
        X_test_activations_layer_1 = session.run(DAE_layer_1, feed_dict={X: X_test_scaled,
                                                                         b_train_autoencoder: False})
        X_test_activations_layer_2 = session.run(DAE_layer_2, feed_dict={X: X_test_scaled,
                                                                         b_train_autoencoder: False})
        if units_layer_3 is not None:
            X_test_activations_layer_3 = session.run(DAE_layer_3, feed_dict={X: X_test_scaled,
                                                                             b_train_autoencoder: False})
    if units_layer_3 is None:
        df_train_activations = pd.DataFrame(np.concatenate([X_activations_layer_1, X_activations_layer_2],
                                                           axis=1))
        df_test_activations = pd.DataFrame(np.concatenate([X_test_activations_layer_1, X_test_activations_layer_2],
                                                          axis=1))
    else:
        df_train_activations = pd.DataFrame(np.concatenate([X_activations_layer_1, X_activations_layer_2,
                                                            X_activations_layer_3
                                                            ],
                                                           axis=1))
        df_test_activations = pd.DataFrame(np.concatenate([X_test_activations_layer_1, X_test_activations_layer_2,
                                                           X_test_activations_layer_3
                                                           ],
                                                          axis=1))
    df_train_activations.loc[:, 'reconstruction_mse'] = np.transpose(
        mean_squared_error(np.transpose(X_train_scaled),
                           np.transpose(X_train_autoencoded),
                           multioutput='raw_values'))
    df_test_activations.loc[:, 'reconstruction_mse'] = np.transpose(
        mean_squared_error(np.transpose(X_test_scaled),
                           np.transpose(X_test_autoencoded),
                           multioutput='raw_values'))

    del X_shuffled
    gc.collect()
    print('Finished')

    return df_train_activations, df_test_activations, lst_scores_logreg


def postprocess_autoencoder(X_train_activations: pd.DataFrame, X_test_activations: pd.DataFrame,
                            X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame):
    """Postprocess the Autoencoder activations. This means RankGauss transforming the reconstruction mean squared error
    and adding external source variables from original data set.

    :param X_train_activations: Training Autoencoder activations.
    :param X_test_activations: Test Autoencoder activations.
    :param X_train_scaled: Scaled training data.
    :param X_test_scaled: Scaled test data.
    """
    X_scaled = pd.concat([X_train_scaled, X_test_scaled]).reset_index(drop=True)

    df_concat = pd.concat([X_train_activations, X_test_activations])
    df_concat.loc[:, 'reconstruction_mse'] = preprocessing.rank_gauss(df_concat.loc[:, 'reconstruction_mse'].values)

    X_train_activations = df_concat.iloc[:X_train_activations.shape[0], :]
    X_test_activations = df_concat.iloc[X_train_activations.shape[0]:, :]

    df_concat = pd.concat([X_train_activations, X_test_activations])

    for col in [c for c in X_scaled.columns if c in ['EXT_SOURCE_{}'.format(i) for i in range(3)]]:
        df_concat.loc[:, col] = X_scaled.loc[:, col]

    X_train_activations = df_concat.iloc[:X_train_activations.shape[0], :]
    X_test_activations = df_concat.iloc[X_train_activations.shape[0]:, :]

    X_train_activations.columns = [str(c) for c in X_train_activations.columns]
    X_test_activations.columns = [str(c) for c in X_test_activations.columns]

    return X_train_activations, X_test_activations


@utils.ensure_directory_exists('predictions')
def run_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, params: Dict, run_classifier: Callable,
              filename_preds_valid: str, filename_preds_test: str, n_splits: int = 10, n_runs: int = 10):
    """Run a classifier.

    :param X_train: Training data.
    :param y_train: Training label.
    :param X_test: Test data.
    :param params: Dictionary containing the parameter of the used classifier.
    :param run_classifier: Callable function that runs the classifier.
    :param filename_preds_valid: Filename for the validation predictions.
    :param filename_preds_test: Fielname for the test predictions.
    :param n_splits: Number of splits for the stratified cross validation.
    :param n_runs: Number of runs with different random seed of the cross validation.
    :return: None
    """

    for random_state in range(1, 1+n_runs):
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_preds = np.zeros((X_train.shape[0],))
        sub_preds = np.zeros((X_test.shape[0],))

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

            X_tr, y_tr = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            p_valid, p_test = run_classifier(X_tr, y_tr, X_va, y_va, X_test, params)

            oof_preds[val_idx] = p_valid
            sub_preds += p_test / n_splits

        score = roc_auc_score(y_train, oof_preds)
        print('\nAUC: {}'.format(score))

        if not os.path.isfile(filename_preds_valid):
            df_preds_lgb_valid = pd.DataFrame(oof_preds, columns=['prediction_{}'.format(random_state)])
            df_preds_lgb_test = pd.DataFrame(sub_preds, columns=['prediction_{}'.format(random_state)])
        else:
            df_preds_lgb_valid = pd.read_csv(filename_preds_valid)
            df_preds_lgb_valid.loc[:, 'prediction_{}'.format(random_state)] = oof_preds
            df_preds_lgb_test = pd.read_csv(filename_preds_test)
            df_preds_lgb_test.loc[:, 'prediction_{}'.format(random_state)] = sub_preds
        df_preds_lgb_valid.to_csv(filename_preds_valid, index=False)
        df_preds_lgb_test.to_csv(filename_preds_test, index=False)


def __run_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_runs: int,
                   filename_preds_valid: str, filename_preds_test: str, **kwargs):
    """Run Lightgmb model."""

    lgb_params = {
        'learning_rate': kwargs.get('learning_rate', 0.01),
        'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
        'feature_fraction': kwargs.get('feature_fraction', 1.0),
        'lambda_l1': kwargs.get('lambda_l1', 5),
        'lambda_l2': kwargs.get('lambda_l2', 0.5),
        'min_data_in_leaf': kwargs.get('min_data_in_leaf', 512),
        'min_sum_hessian_in_leaf': kwargs.get('min_sum_hessian_in_leaf', 128),
        'min_split_gain': kwargs.get('min_split_gain', 1.0),
        'max_bin': kwargs.get('max_bin', 256),
        'num_leaves': kwargs.get('num_leaves', 7),

        'bagging_freq': kwargs.get('bagging_freq', 1),
        'objective': 'binary',
        'metric': 'AUC',
        'boosting': kwargs.get('boosting', 'gbdt')
    }

    def __run(X_train, y_train, X_valid, y_valid, X_test):
        clf = lightgbm.train(lgb_params, train_set=lightgbm.Dataset(X_train, label=y_train),
                             valid_names=['train', 'valid'],
                             valid_sets=[lightgbm.Dataset(X_train, label=y_train),
                                         lightgbm.Dataset(X_valid, label=y_valid)],
                             num_boost_round=100000, verbose_eval=2500, early_stopping_rounds=200)

        return clf.predict(X_valid), clf.predict(X_test)

    run_model(X_train=X_train, y_train=y_train, X_test=X_test, params=lgb_params, run_classifier=__run, n_runs=n_runs,
              filename_preds_valid=filename_preds_valid, filename_preds_test=filename_preds_test)


def run_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_runs: int = 10, **kwargs):
    """Run LightGBM model on original data."""
    __run_lightgbm(X_train=X_train, y_train=y_train, X_test=X_test, n_runs=n_runs,
                   filename_preds_valid=config.filename_preds_lgb_valid,
                   filename_preds_test=config.filename_preds_lgb_test, **kwargs)


def run_dae_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_runs: int = 10, **kwargs):
    """Run LightGBM model on Autoencoder activations."""
    __run_lightgbm(X_train=X_train, y_train=y_train, X_test=X_test, n_runs=n_runs,
                   filename_preds_valid=config.filename_preds_dae_lgb_valid,
                   filename_preds_test=config.filename_preds_dae_lgb_test, **kwargs)


def run_nn(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_runs: int = 10, **kwargs):
    """Run neural net."""

    dae_nn_params = {
        'units_1': kwargs.get('units_1', 500),
        'units_2': kwargs.get('units_2', 500),
        'units_3': kwargs.get('units_3', None),
        'layer_0_dropout': kwargs.get('layer_0_dropout', 0.0),
        'layer_1_dropout': kwargs.get('layer_1_dropout', 0.5),
        'layer_2_dropout': kwargs.get('layer_2_dropout', 0.5),
        'layer_3_dropout': kwargs.get('layer_3_dropout', 0.5),
        'lambda_l2': kwargs.get('lambda_l2', 0.0005),
        'epochs': kwargs.get('epochs', 500),
        'batch_size': kwargs.get('batch_size', 128),
        'learning_rate': kwargs.get('learning_rate', 0.02),
        'opt': kwargs.get('opt', 'ftrl'),
        'batch_size_multiplier': kwargs.get('batch_size_multiplier', 4),
        'batch_sizes': kwargs.get('batch_sizes', None),
        'early_stopping': kwargs.get('early_stopping', True),
        'patience': kwargs.get('patience', 5),
        'verbose_eval': kwargs.get('verbose_eval', 1)
    }

    def __run(X_train, y_train, df_test, df_valid=None, y_valid=None, units_1=500, units_2=500, units_3=None,
              layer_0_dropout=0.0, layer_1_dropout=0.5, layer_2_dropout=0.5, layer_3_dropout=0.5, lambda_l2=0.0005,
              epochs=500, batch_size=128, learning_rate=0.02, opt='ftrl', batch_size_multiplier=4, batch_sizes=None,
              early_stopping=True, patience=5, verbose_eval=1):
        assert not early_stopping or (df_valid is not None and y_valid is not None), 'No validation sets specified.'
        assert not early_stopping or patience > 0, 'Patience needs to be greater than or equal to 1 for early stopping.'

        if batch_sizes is None:
            dict_batch_size = {0: batch_size}

        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        b_training_mode = tf.placeholder(dtype=tf.bool)
        X = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1]])

        dropout_0 = tf.layers.dropout(inputs=X, rate=layer_0_dropout, training=b_training_mode)

        # Layer 1
        layer_1 = tf.layers.Dense(units=units_1, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_l2))
        activations_1 = tf.nn.leaky_relu(layer_1(dropout_0))
        dropout_1 = tf.layers.dropout(inputs=activations_1, rate=layer_1_dropout, training=b_training_mode)

        # Layer 2
        layer_2 = tf.layers.Dense(units=units_2, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_l2))
        activations_2 = tf.nn.leaky_relu(layer_2(dropout_1))
        dropout_2 = tf.layers.dropout(inputs=activations_2, rate=layer_2_dropout, training=b_training_mode)

        # Layer 3
        if units_3 is not None:
            layer_3 = tf.layers.Dense(units=units_3, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_l2))
            activations_3 = tf.nn.leaky_relu(layer_3(dropout_2))
            dropout_3 = tf.layers.dropout(inputs=activations_3, rate=layer_3_dropout, training=b_training_mode)

        # Layer final
        final_layer = tf.layers.Dense(units=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_l2))
        if units_3 is None:
            y_pred = tf.sigmoid(final_layer(dropout_2))
        else:
            y_pred = tf.sigmoid(final_layer(dropout_3))

        # Loss / Optimizer
        loss = tf.losses.log_loss(labels=y_true, predictions=y_pred) + tf.losses.get_regularization_loss()

        if opt == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif opt == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif opt == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

        if verbose_eval is not None:
            print('Starting training')
        best_auc = 0
        loss_values_train = []
        loss_values_valid = []
        auc_values_train = []
        auc_values_valid = []

        for epoch in range(epochs):
            loss_values_batch_train = []
            X_train_shuffled = X_train.sample(frac=1)

            if batch_sizes is not None:
                batch_size = batch_sizes[epoch]
            else:
                dict_batch_size[epoch] = batch_size

            for batch in range(X_train_shuffled.shape[0] // batch_size):
                X_batch = X_train.iloc[batch * batch_size:(batch + 1) * batch_size, :]
                y_batch = y_train.iloc[batch * batch_size:(batch + 1) * batch_size]
                _, loss_value_train = session.run((train, loss), feed_dict={X: X_batch,
                                                                            y_true: y_batch.values.reshape(
                                                                                [y_batch.shape[0], 1]),
                                                                            b_training_mode: True})
                loss_values_batch_train.append(loss_value_train)

            loss_values_train.append(np.mean(loss_values_batch_train))

            if early_stopping:
                preds_valid, loss_value_valid = session.run((y_pred, loss), feed_dict={X: df_valid,
                                                                                       y_true: y_valid.values.reshape(
                                                                                           [y_valid.shape[0], 1]),
                                                                                       b_training_mode: False})
                loss_values_valid.append(loss_value_valid)
                auc_values_valid.append(roc_auc_score(y_valid, preds_valid))

                if (epoch == 0) | (auc_values_valid[-1] > best_auc):
                    best_auc = auc_values_valid[-1]

                    predictions_valid = preds_valid
                    predictions_test = session.run(y_pred, feed_dict={X: df_test, b_training_mode: False})

                elif batch_sizes is None and (epoch - np.argmax(auc_values_valid)) >= patience // 2:
                    if verbose_eval is not None and (epoch % verbose_eval != 0):
                        print('Epoch {} \tAUC (train): {:0.4f}\tAUC (valid): {:0.4f}\tbatch_size: {}, opt:{}'.format(
                            epoch, auc_values_train[-1], auc_values_valid[-1], batch_size, opt))
                    preds_train = session.run(y_pred, feed_dict={X: X_train, b_training_mode: False})
                    auc_values_train.append(roc_auc_score(y_train, preds_train))

                    batch_size = min(batch_size * batch_size_multiplier, X_train.shape[0])

                if (epoch - np.argmax(auc_values_valid)) >= patience:
                    preds_train = session.run(y_pred, feed_dict={X: X_train, b_training_mode: False})
                    auc_values_train.append(roc_auc_score(y_train, preds_train))

                    if verbose_eval is not None:
                        print('Epoch {} \tAUC (train): {:0.4f}\tAUC (valid): {:0.4f}\tbatch_size: {}, opt:{}'.format(
                            epoch, auc_values_train[-1], auc_values_valid[-1], batch_size, opt))
                    break

            if verbose_eval is not None and ((epoch % verbose_eval == 0) or (epoch == epochs - 1)):
                preds_train = session.run(y_pred, feed_dict={X: X_train, b_training_mode: False})
                auc_values_train.append(roc_auc_score(y_train, preds_train))

                if verbose_eval is not None:
                    if early_stopping:
                        print('Epoch {} \tAUC (train): {:0.4f}\tAUC (valid): {:0.4f}\tbatch_size: {}, opt:{}'.format(
                            epoch, auc_values_train[-1], auc_values_valid[-1], batch_size, opt))
                    else:
                        print('Epoch {} \tLogloss (train): {:0.4f}\tAUC (train): {:0.4f}\tbatch_size: {}, opt:{}'.format(
                            epoch, loss_values_train[-1], auc_values_train[-1], batch_size, opt))

        if early_stopping:
            best_iteration = np.argmax(auc_values_valid)
            if verbose_eval is not None:
                print('\nBest AUC (valid): {:0.4f} on epoch {}'.format(np.max(auc_values_valid), best_iteration))
            return predictions_valid, predictions_test, best_iteration, dict_batch_size
        else:
            predictions_test = session.run(y_pred, feed_dict={X: df_test, b_training_mode: False})
            session.close()
            return predictions_test

    run_model(X_train=X_train, y_train=y_train, X_test=X_test, params=dae_nn_params, run_classifier=__run,
              n_runs=n_runs, filename_preds_valid=config.filename_preds_dae_dl_valid,
              filename_preds_test=config.filename_preds_dae_dl_test)
