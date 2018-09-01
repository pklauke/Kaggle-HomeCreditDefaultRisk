# !/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import pickle

from scipy.special import erfinv
import pandas as pd
import numpy as np


def rank_gauss(x):
    """RankGauss transformation of a feature vector."""
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    efi_x /= (efi_x.max() - efi_x.min()) / 2.0
    return efi_x


def preprocessing_gbdt_models(data: pd.DataFrame, test: pd.DataFrame):
    """Preprocess training and test data for gradient boosted decision tree models."""
    for col in data.columns:
        srs_missing = pd.isnull(data.loc[:, col])
        data.loc[srs_missing, col] = -1000000

        srs_missing = pd.isnull(test.loc[:, col])
        test.loc[srs_missing, col] = -1000000

    return data, test


def preprocessing_linear_models(data: pd.DataFrame, test: pd.DataFrame):
    """Preprocess training and test data for linear models."""
    len_train = data.shape[0]

    df_concat = pd.concat([data, test])
    lst_no_flag_missing = pickle.load(open('data/lst_no_flag_missing.pkl', 'rb'))

    for col in df_concat.columns:
        srs_missing = df_concat[col].isnull()
        if np.sum(srs_missing) > 0:
            assert 'flag' not in col, 'Missing values in flag column ' + col
            if not any([c in col for c in
                        ['avg_buro_bal_', 'avg_buro_', 'avg_prev_', 'avg_cc_bal_', 'avg_inst_'] + lst_no_flag_missing]):
                df_concat.loc[:, 'flag_missing_' + col] = srs_missing.apply(np.int8)
            df_concat.loc[srs_missing, col] = df_concat.loc[~srs_missing, col].mean()

    lst_flag_cols = [c for c in df_concat.columns if 'flag_' in c]
    df_concat = df_concat.loc[:, lst_flag_cols + [c for c in df_concat.columns
                                                  if c != 'target' and c not in lst_flag_cols]]

    for col in df_concat.iloc[:, len(lst_flag_cols):].columns:
        df_concat.loc[:, col] = rank_gauss(df_concat.loc[:, col].values)

    data = df_concat.iloc[:len_train, :]
    test = df_concat.iloc[len_train:, :]

    return data, test
