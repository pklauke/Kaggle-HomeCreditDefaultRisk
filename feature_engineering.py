# !/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import downcast_dtypes
import config
from utils import ensure_directory_exists


@ensure_directory_exists(directory='data')
def run_feature_engineering():
    """Engineer features and save them."""
    gc.enable()

    buro_bal = pd.read_csv(config.path + 'bureau_balance.csv.zip')
    print('Buro bal shape : ', buro_bal.shape)

    print('transform to dummies')
    buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS',
                                                                                                             axis=1)

    print('Counting buros')
    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

    print('averaging buro bal')
    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

    avg_buro_bal.loc[:, 'flag_missing'] = 0
    avg_buro_bal.columns = ['avg_buro_bal_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()

    print('Read Bureau')
    buro = pd.read_csv(config.path + 'bureau.csv.zip')

    print('Go to dummies')
    buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
    buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
    buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

    buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
    # buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

    del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
    gc.collect()

    print('Merge with buro avg')
    buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU',
                                suffixes=('', '_bur_bal'))

    print('Counting buro per SK_ID_CURR')
    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

    print('Averaging bureau')
    avg_buro = buro_full.groupby('SK_ID_CURR').mean()

    avg_buro.loc[:, 'flag_missing'] = 0
    avg_buro.columns = ['avg_buro_' + f_ for f_ in avg_buro.columns]

    del buro, buro_full
    gc.collect()

    print('Read prev')
    prev = pd.read_csv(config.path + 'previous_application.csv.zip')

    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]

    print('Go to dummies')
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

    prev = pd.concat([prev, prev_dum], axis=1)

    del prev_dum
    gc.collect()

    print('Counting number of Prevs')
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

    print('Averaging prev')
    avg_prev = prev.groupby('SK_ID_CURR').mean()

    avg_prev.loc[:, 'flag_missing'] = 0
    avg_prev.columns = ['avg_prev_' + f_ for f_ in avg_prev.columns]

    del prev
    gc.collect()

    print('Reading POS_CASH')
    pos = pd.read_csv(config.path + 'POS_CASH_balance.csv.zip')

    print('Go to dummies')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

    print('Compute nb of prevs per curr')
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Go to averages')
    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    print('Reading CC balance')
    cc_bal = pd.read_csv(config.path + 'credit_card_balance.csv.zip')

    print('Go to dummies')
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()

    avg_cc_bal.loc[:, 'flag_missing'] = 0
    avg_cc_bal.columns = ['avg_cc_bal_' + f_ for f_ in avg_cc_bal.columns]

    del cc_bal, nb_prevs
    gc.collect()

    print('Reading Installments')
    inst = pd.read_csv(config.path + 'installments_payments.csv.zip')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_inst = inst.groupby('SK_ID_CURR').mean()

    avg_inst.loc[:, 'flag_missing'] = 0
    avg_inst.columns = ['avg_inst_' + f_ for f_ in avg_inst.columns]

    print('Read data and test')
    data = pd.read_csv(config.path + 'application_train.csv.zip')
    test = pd.read_csv(config.path + 'application_test.csv.zip')
    print('Shapes : ', data.shape, test.shape)

    y = data['TARGET']
    del data['TARGET']

    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]

    len_train = data.shape[0]
    df_concat = pd.concat([data, test])

    for f_ in categorical_feats:
        df_concat = pd.concat([df_concat.drop(columns=f_), pd.get_dummies(df_concat[f_], prefix='flag_ohe_' + f_)],
                              axis=1)

    data = df_concat.iloc[:len_train, :]
    test = df_concat.iloc[len_train:, :]

    del df_concat
    gc.collect()

    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    gc.collect()

    print('Shapes : ', data.shape, test.shape)

    for col in [c for c in data.columns if '_flag_missing' in c]:
        data.loc[:, col] = data.loc[:, col].fillna(1)
        test.loc[:, col] = test.loc[:, col].fillna(1)

    df_bureau = pd.read_csv(config.path + 'bureau.csv.zip')

    data_tmp = data.copy()
    test_tmp = test.copy()

    df_tmp = df_bureau.loc[:, ['SK_ID_CURR', 'AMT_ANNUITY']].groupby('SK_ID_CURR', as_index=False).AMT_ANNUITY.sum() \
        .rename(columns={'AMT_ANNUITY': 'bureau_sum_annuity'})
    data = data_tmp.merge(df_tmp, on='SK_ID_CURR', how='left')
    test = test_tmp.merge(df_tmp, on='SK_ID_CURR', how='left')
    data.loc[:, 'bureau_sum_annuity'] = data.loc[:, 'bureau_sum_annuity'].fillna(0)
    test.loc[:, 'bureau_sum_annuity'] = test.loc[:, 'bureau_sum_annuity'].fillna(0)
    data.loc[:, 'sum_annuity_div_income'] = data.bureau_sum_annuity / data.AMT_INCOME_TOTAL
    test.loc[:, 'sum_annuity_div_income'] = test.bureau_sum_annuity / test.AMT_INCOME_TOTAL

    df_tmp = pd.get_dummies(df_bureau.loc[:, ['SK_ID_CURR', 'CREDIT_ACTIVE']] \
                            .rename(columns={'CREDIT_ACTIVE': 'count_credit_active'}), columns=['count_credit_active'])\
        .groupby('SK_ID_CURR', as_index=False).sum()
    df_tmp.loc[:, 'credit_active_frac_active'] = df_tmp.count_credit_active_Active / df_tmp.loc[:,
                                                                                     ['count_credit_active_Bad debt',
                                                                                      'count_credit_active_Closed',
                                                                                      'count_credit_active_Sold']].sum(
        axis=1)
    data = data_tmp.merge(df_tmp, on='SK_ID_CURR', how='left')
    test = test_tmp.merge(df_tmp, on='SK_ID_CURR', how='left')

    df_app_train = pd.read_csv(config.path + 'application_train.csv.zip')
    df_app_test = pd.read_csv(config.path + 'application_test.csv.zip')

    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    for col_to_encode in [c for c in df_app_train.columns if df_app_train[c].dtype == 'object']:

        valid_cats = df_app_train.loc[:, col_to_encode].value_counts()[
            df_app_train.loc[:, col_to_encode].value_counts() > 500].index

        for train_idx, valid_idx in skf.split(df_app_train, df_app_train.TARGET):
            X_tr = df_app_train.loc[train_idx, [col_to_encode, 'TARGET']].loc[
                   (df_app_train.loc[:, col_to_encode].isin(valid_cats)), :]
            X_va = df_app_train.loc[valid_idx, [col_to_encode, 'TARGET']]

            df_target_gby_col = X_tr.groupby(col_to_encode, as_index=False).mean().rename(
                columns={'TARGET': 'target_mean'})
            data.loc[valid_idx, 'target_mean_' + col_to_encode] = df_app_train.loc[valid_idx, [col_to_encode]].merge(
                df_target_gby_col, on=col_to_encode, how='left').target_mean
            test.loc[:, 'target_mean_' + col_to_encode] = df_app_test.loc[:, [col_to_encode]].merge(df_target_gby_col,
                                                                                                    on=col_to_encode,
                                                                                                    how='left').target_mean

        del X_tr, X_va

    data['ANNUITY LENGTH'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    test['ANNUITY LENGTH'] = test['AMT_CREDIT'] / test['AMT_ANNUITY']

    data['INCOME_PER_FAM'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    test['INCOME_PER_FAM'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS']

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']

    test['DAYS_EMPLOYED_PERC'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']
    test['ANNUITY_INCOME_PERC'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']

    def add_new_features(df, dataset='train'):
        inc_by_org = \
            pd.read_csv(config.path + 'application_{}.csv.zip'.format(dataset),
                        usecols=['ORGANIZATION_TYPE', 'AMT_INCOME_TOTAL'])[
                ['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
        df['NEW_INC_BY_ORG'] = \
            pd.read_csv(config.path + 'application_{}.csv.zip'.format(dataset), usecols=['ORGANIZATION_TYPE'])[
                'ORGANIZATION_TYPE'].map(inc_by_org)
        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    add_new_features(data, 'train')
    add_new_features(test, 'test')

    def one_hot_encoder(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    def bureau_and_balance(num_rows=None, nan_as_category=True):
        bureau = pd.read_csv(config.path + 'bureau.csv.zip', nrows=num_rows)
        bb = pd.read_csv(config.path + 'bureau_balance.csv.zip', nrows=num_rows)
        bb, bb_cat = one_hot_encoder(bb, nan_as_category)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(columns='SK_ID_BUREAU', inplace=True)
        del bb, bb_agg
        gc.collect()

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
            'AMT_ANNUITY': ['max', 'mean'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left')  # , on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left')  # , on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        return bureau_agg

    tmp_bnb = bureau_and_balance().reset_index()
    data = data.merge(tmp_bnb, on='SK_ID_CURR', how='left')
    test = test.merge(tmp_bnb, on='SK_ID_CURR', how='left')
    gc.collect()

    def previous_applications(num_rows=None, nan_as_category=True):
        prev = pd.read_csv(config.path + 'previous_application.csv.zip', nrows=num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max'],
            'AMT_APPLICATION': ['min', 'max'],
            'AMT_CREDIT': ['min', 'max'],
            'APP_CREDIT_PERC': ['min', 'max', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max'],
            'AMT_GOODS_PRICE': ['min', 'max'],
            'HOUR_APPR_PROCESS_START': ['min', 'max'],
            'RATE_DOWN_PAYMENT': ['min', 'max'],
            'DAYS_DECISION': ['min', 'max'],
            'CNT_PAYMENT': ['sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = []  # ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations})  # , **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left')
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left')
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg

    tmp_pa = previous_applications().reset_index()
    data = data.merge(tmp_pa, on='SK_ID_CURR', how='left')
    test = test.merge(tmp_pa, on='SK_ID_CURR', how='left')
    print('data shape:', data.shape)
    gc.collect()

    # Preprocess installments_payments.csv
    def installments_payments(num_rows=None, nan_as_category=True):
        ins = pd.read_csv(config.path + 'installments_payments.csv.zip', nrows=num_rows)
        ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'sum'],
            'DBD': ['max', 'sum'],
            'PAYMENT_PERC': ['max', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'sum']
        }
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        gc.collect()
        return ins_agg

    tmp_ip = installments_payments().reset_index()
    data = data.merge(tmp_ip, on='SK_ID_CURR', how='left')
    test = test.merge(tmp_ip, on='SK_ID_CURR', how='left')
    print('data shape:', data.shape)
    gc.collect()

    # Preprocess credit_card_balance.csv
    def credit_card_balance(num_rows=None, nan_as_category=True):
        cc = pd.read_csv(config.path + 'credit_card_balance.csv.zip', nrows=num_rows)
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg

    tmp_ccb = credit_card_balance().reset_index()
    data = data.merge(tmp_ccb, on='SK_ID_CURR', how='left')
    test = test.merge(tmp_ccb, on='SK_ID_CURR', how='left')
    print('data shape:', data.shape)
    gc.collect()

    lst_no_flag_missing = []
    for col in [c for dataset in [tmp_ccb, tmp_ip, tmp_pa, tmp_bnb] for c in dataset]:
        lst_no_flag_missing.append(col)

    del data_tmp, test_tmp, tmp_ccb, tmp_ip, tmp_pa, tmp_bnb
    gc.collect()

    data.loc[:, 'EXT_SOURCE_VAR'] = data.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].var(axis=1)
    test.loc[:, 'EXT_SOURCE_VAR'] = test.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].var(axis=1)

    lst = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
           'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'OWN_CAR_AGE',
           'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
           'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            data.loc[:, lst[i] + '_DIV_' + lst[j]] = data.loc[:, lst[i]] / data.loc[:, lst[j]]
            test.loc[:, lst[i] + '_DIV_' + lst[j]] = test.loc[:, lst[i]] / test.loc[:, lst[j]]

    feats = get_features_to_keep(data)
    data = data.loc[:, feats]
    test = test.loc[:, feats]

    data = downcast_dtypes(data)
    test = downcast_dtypes(test)

    data.to_feather('data/data.feather')
    test.to_feather('data/test.feather')
    pickle.dump(lst_no_flag_missing, open('data/lst_no_flag_missing.pkl', 'wb'))


def get_features_to_keep(df: pd.DataFrame):
    to_drop = ['DAYS_REGISTRATION', 'CNT_FAM_MEMBERS',
               'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
               'avg_prev_NAME_GOODS_CATEGORY_Weapon', 'avg_buro_avg_buro_bal_MONTHS_BALANCE']
    to_drop += [f for f in df.columns if 'FLAG_DOCUMENT' in f]
    to_drop += ['avg_prev_NAME_SELLER_INDUSTRY_MLM partners', 'avg_cc_bal_cc_bal_status__Approved',
                'flag_ohe_ORGANIZATION_TYPE_Industry: type 7', 'Approved',
                'flag_ohe_ORGANIZATION_TYPE_Trade: type 2',
                'flag_ohe_NAME_EDUCATION_TYPE_Incomplete higher', 'flag_ohe_FLAG_OWN_REALTY_Y', 'Canceled',
                'avg_prev_NAME_GOODS_CATEGORY_House Construction', 'avg_prev_AMT_GOODS_PRICE',
                'flag_ohe_WEEKDAY_APPR_PROCESS_START_THURSDAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                'avg_buro_ty__Real estate loan', 'avg_cc_bal_cc_bal_status__Completed',
                'flag_ohe_NAME_HOUSING_TYPE_Co-op apartment', 'flag_ohe_ORGANIZATION_TYPE_Medicine',
                'avg_prev_NAME_GOODS_CATEGORY_Medicine', 'flag_ohe_OCCUPATION_TYPE_Realty agents',
                'flag_ohe_NAME_HOUSING_TYPE_With parents', 'flag_ohe_NAME_FAMILY_STATUS_Civil marriage',
                'flag_ohe_OCCUPATION_TYPE_Private service staff', 'flag_ohe_ORGANIZATION_TYPE_University',
                'avg_prev_NAME_GOODS_CATEGORY_Other', 'flag_ohe_ORGANIZATION_TYPE_Restaurant',
                'flag_ohe_NAME_INCOME_TYPE_Commercial associate',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Money for a third person',
                'NONLIVINGAPARTMENTS_AVG', 'avg_buro_avg_buro_bal_buro_bal_status_5',
                'flag_ohe_ORGANIZATION_TYPE_Industry: type 9', 'flag_ohe_ORGANIZATION_TYPE_Telecom',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Gasification / water supply']
    to_drop += ['flag_ohe_EMERGENCYSTATE_MODE_No', 'flag_ohe_NAME_INCOME_TYPE_Businessman',
                'AMT_REQ_CREDIT_BUREAU_DAY', 'flag_ohe_ORGANIZATION_TYPE_Trade: type 1',
                'flag_ohe_ORGANIZATION_TYPE_Transport: type 1',
                'flag_ohe_HOUSETYPE_MODE_terraced house', 'avg_prev_CODE_REJECT_REASON_VERIF',
                'flag_ohe_OCCUPATION_TYPE_Secretaries',
                'flag_ohe_ORGANIZATION_TYPE_Realtor', 'avg_buro_avg_buro_bal_buro_bal_status_3',
                'avg_prev_PRODUCT_COMBINATION_POS others without interest',
                'flag_ohe_WALLSMATERIAL_MODE_Mixed', 'flag_ohe_ORGANIZATION_TYPE_Industry: type 2',
                'avg_prev_NAME_TYPE_SUITE_Group of people']
    to_drop += ['avg_prev_NAME_GOODS_CATEGORY_Animals', 'avg_prev_NAME_SELLER_INDUSTRY_Tourism',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Education', 'flag_ohe_ORGANIZATION_TYPE_Postal',
                'flag_ohe_FLAG_OWN_CAR_N', 'avg_prev_CODE_REJECT_REASON_XNA',
                'flag_ohe_NAME_CONTRACT_TYPE_Revolving loans', 'avg_prev_NAME_CASH_LOAN_PURPOSE_Hobby',
                'avg_prev_CODE_REJECT_REASON_SC', 'avg_prev_NAME_GOODS_CATEGORY_Additional Service',
                'avg_prev_NAME_GOODS_CATEGORY_Homewares', 'avg_prev_FLAG_LAST_APPL_PER_CONTRACT_Y',
                'XNA', 'flag_ohe_ORGANIZATION_TYPE_Transport: type 3', 'avg_prev_CODE_REJECT_REASON_SYSTEM',
                'avg_prev_RATE_INTEREST_PRIMARY', 'avg_prev_NAME_GOODS_CATEGORY_Tourism',
                'flag_ohe_HOUSETYPE_MODE_block of flats',
                'avg_cc_bal_cc_bal_status__Demand',
                'avg_prev_NAME_PAYMENT_TYPE_Cashless from the account of the employer',
                'avg_prev_NAME_TYPE_SUITE_Other_A', 'avg_buro_ty__Another type of loan',
                'avg_prev_NAME_GOODS_CATEGORY_Medical Supplies',
                'avg_prev_NAME_GOODS_CATEGORY_Insurance', 'avg_buro_ty__Loan for business development',
                'flag_ohe_CODE_GENDER_XNA',
                'flag_ohe_NAME_TYPE_SUITE_Children', 'flag_ohe_ORGANIZATION_TYPE_Industry: type 13',
                'flag_ohe_ORGANIZATION_TYPE_Agriculture',
                'avg_buro_ty__Loan for working capital replenishment', 'flag_ohe_ORGANIZATION_TYPE_Trade: type 6',
                'flag_ohe_NAME_TYPE_SUITE_Other_B', 'flag_ohe_ORGANIZATION_TYPE_Business Entity Type 1',
                'flag_ohe_ORGANIZATION_TYPE_Trade: type 5', 'REG_REGION_NOT_LIVE_REGION',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Everyday expenses',
                'flag_ohe_NAME_INCOME_TYPE_Unemployed', 'avg_prev_NAME_CASH_LOAN_PURPOSE_Payments on other loans',
                'flag_ohe_OCCUPATION_TYPE_HR staff', 'avg_prev_NFLAG_LAST_APPL_IN_DAY',
                'flag_ohe_ORGANIZATION_TYPE_Services',
                'flag_ohe_ORGANIZATION_TYPE_Insurance', 'flag_ohe_NAME_INCOME_TYPE_Student',
                'flag_ohe_WALLSMATERIAL_MODE_Monolithic',
                'avg_buro_cu__currency 4', 'flag_ohe_EMERGENCYSTATE_MODE_Yes',
                'avg_prev_NAME_GOODS_CATEGORY_Gardening',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Furniture', 'flag_ohe_ORGANIZATION_TYPE_Transport: type 2',
                'flag_ohe_NAME_FAMILY_STATUS_Unknown', 'avg_prev_NAME_GOODS_CATEGORY_Fitness',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday',
                'flag_ohe_FONDKAPREMONT_MODE_not specified', 'flag_ohe_OCCUPATION_TYPE_Waiters/barmen staff',
                'avg_prev_NAME_CLIENT_TYPE_XNA', 'flag_ohe_ORGANIZATION_TYPE_Industry: type 6',
                'avg_buro_ty__Unknown type of loan',
                'avg_buro_ca__Bad debt', 'flag_ohe_ORGANIZATION_TYPE_Advertising',
                'flag_ohe_NAME_EDUCATION_TYPE_Academic degree',
                'flag_ohe_ORGANIZATION_TYPE_Bank', 'flag_ohe_ORGANIZATION_TYPE_Mobile',
                'avg_prev_NAME_GOODS_CATEGORY_Office Appliances',
                'flag_ohe_ORGANIZATION_TYPE_Security Ministries', 'flag_ohe_ORGANIZATION_TYPE_Legal Services',
                'Amortized debt',
                'flag_ohe_ORGANIZATION_TYPE_Security', 'avg_buro_ty__Interbank credit',
                'flag_ohe_ORGANIZATION_TYPE_XNA',
                'avg_prev_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land',
                'flag_ohe_WALLSMATERIAL_MODE_Others',
                'flag_ohe_ORGANIZATION_TYPE_Industry: type 10', 'avg_prev_NAME_CASH_LOAN_PURPOSE_Buying a garage',
                'flag_ohe_OCCUPATION_TYPE_Low-skill Laborers', 'flag_ohe_ORGANIZATION_TYPE_Industry: type 12',
                'avg_prev_RATE_INTEREST_PRIVILEGED']
    to_drop += ['BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MAX', 'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_VAR',
                'BURO_CREDIT_DAY_OVERDUE_MAX', 'BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_DAYS_CREDIT_ENDDATE_MEAN',
                'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'BURO_CNT_CREDIT_PROLONG_SUM', 'BURO_AMT_CREDIT_SUM_MAX',
                'BURO_AMT_CREDIT_SUM_MEAN', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',
                'BURO_AMT_CREDIT_SUM_LIMIT_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MEAN', 'BURO_AMT_ANNUITY_MEAN',
                'BURO_MONTHS_BALANCE_MIN_MIN', 'BURO_MONTHS_BALANCE_SIZE_MEAN', 'BURO_CREDIT_ACTIVE_Active_MEAN',
                'BURO_CREDIT_ACTIVE_Bad debt_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
                'BURO_CREDIT_ACTIVE_Sold_MEAN',
                'BURO_CREDIT_ACTIVE_nan_MEAN', 'BURO_CREDIT_CURRENCY_currency 1_MEAN',
                'BURO_CREDIT_CURRENCY_currency 2_MEAN',
                'BURO_CREDIT_CURRENCY_currency 3_MEAN', 'BURO_CREDIT_CURRENCY_currency 4_MEAN',
                'BURO_CREDIT_CURRENCY_nan_MEAN',
                'BURO_CREDIT_TYPE_Another type of loan_MEAN', 'BURO_CREDIT_TYPE_Car loan_MEAN',
                'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
                'BURO_CREDIT_TYPE_Consumer credit_MEAN', 'BURO_CREDIT_TYPE_Credit card_MEAN',
                'BURO_CREDIT_TYPE_Interbank credit_MEAN',
                'BURO_CREDIT_TYPE_Loan for business development_MEAN',
                'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
                'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
                'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
                'BURO_CREDIT_TYPE_Microloan_MEAN', 'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
                'BURO_CREDIT_TYPE_Mortgage_MEAN', 'BURO_CREDIT_TYPE_Real estate loan_MEAN',
                'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
                'BURO_CREDIT_TYPE_nan_MEAN', 'BURO_STATUS_0_MEAN_MEAN', 'BURO_STATUS_1_MEAN_MEAN',
                'BURO_STATUS_2_MEAN_MEAN', 'BURO_STATUS_3_MEAN_MEAN', 'BURO_STATUS_4_MEAN_MEAN',
                'BURO_STATUS_5_MEAN_MEAN', 'BURO_STATUS_C_MEAN_MEAN', 'BURO_STATUS_X_MEAN_MEAN',
                'BURO_STATUS_nan_MEAN_MEAN', 'ACT_CREDIT_DAY_OVERDUE_MAX', 'ACT_CREDIT_DAY_OVERDUE_MEAN',
                'ACT_CNT_CREDIT_PROLONG_SUM', 'CLS_CREDIT_DAY_OVERDUE_MAX', 'CLS_CREDIT_DAY_OVERDUE_MEAN',
                'CLS_CNT_CREDIT_PROLONG_SUM', 'CLS_AMT_CREDIT_SUM_DEBT_MEAN', 'CLS_AMT_CREDIT_SUM_OVERDUE_MEAN',
                'CLS_AMT_CREDIT_SUM_LIMIT_SUM']
    to_drop += ['APR_AMT_ANNUITY_MAX', 'APR_APP_CREDIT_PERC_MAX', 'REF_AMT_APPLICATION_MIN']
    to_drop += ['INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE', 'INSTAL_DPD_MAX', 'INSTAL_COUNT']
    to_drop += ['CC_MONTHS_BALANCE_MIN', 'CC_MONTHS_BALANCE_MAX', 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN',
                'CC_AMT_DRAWINGS_CURRENT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
                'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM',
                'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_AMT_INST_MIN_REGULARITY_MIN',
                'CC_AMT_RECEIVABLE_PRINCIPAL_VAR',
                'CC_AMT_RECIVABLE_MIN', 'CC_AMT_RECIVABLE_MAX', 'CC_CNT_DRAWINGS_ATM_CURRENT_MIN',
                'CC_CNT_DRAWINGS_CURRENT_MIN', 'CC_CNT_DRAWINGS_CURRENT_VAR', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
                'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX', 'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM',
                'CC_CNT_DRAWINGS_POS_CURRENT_MIN',
                'CC_CNT_INSTALMENT_MATURE_CUM_MIN', 'CC_CNT_INSTALMENT_MATURE_CUM_MAX', 'CC_SK_DPD_MIN',
                'CC_SK_DPD_MAX', 'CC_SK_DPD_VAR', 'CC_SK_DPD_DEF_MIN', 'CC_SK_DPD_DEF_MAX',
                'CC_NAME_CONTRACT_STATUS_Active_MIN', 'CC_NAME_CONTRACT_STATUS_Active_MAX',
                'CC_NAME_CONTRACT_STATUS_Approved_MIN',
                'CC_NAME_CONTRACT_STATUS_Approved_MAX', 'CC_NAME_CONTRACT_STATUS_Approved_SUM',
                'CC_NAME_CONTRACT_STATUS_Approved_VAR',
                'CC_NAME_CONTRACT_STATUS_Completed_MIN', 'CC_NAME_CONTRACT_STATUS_Completed_MAX',
                'CC_NAME_CONTRACT_STATUS_Demand_MIN',
                'CC_NAME_CONTRACT_STATUS_Demand_MAX', 'CC_NAME_CONTRACT_STATUS_Demand_SUM',
                'CC_NAME_CONTRACT_STATUS_Demand_VAR',
                'CC_NAME_CONTRACT_STATUS_Refused_MIN', 'CC_NAME_CONTRACT_STATUS_Refused_MAX',
                'CC_NAME_CONTRACT_STATUS_Refused_SUM',
                'CC_NAME_CONTRACT_STATUS_Refused_VAR', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN',
                'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
                'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM', 'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
                'CC_NAME_CONTRACT_STATUS_Signed_MIN',
                'CC_NAME_CONTRACT_STATUS_Signed_MAX', 'CC_NAME_CONTRACT_STATUS_Signed_SUM',
                'CC_NAME_CONTRACT_STATUS_nan_MIN',
                'CC_NAME_CONTRACT_STATUS_nan_MAX', 'CC_NAME_CONTRACT_STATUS_nan_SUM',
                'CC_NAME_CONTRACT_STATUS_nan_VAR',
                'CC_COUNT']
    to_drop += ['CNT_CHILDREN_DIV_AMT_INCOME_TOTAL;', 'CNT_CHILDREN_DIV_DAYS_BIRTH;',
                'CNT_CHILDREN_DIV_EXT_SOURCE_2;',
                'AMT_INCOME_TOTAL_DIV_OWN_CAR_AGE;', 'AMT_CREDIT_DIV_AMT_ANNUITY;',
                'AMT_CREDIT_DIV_AMT_GOODS_PRICE;']
    feats = [f for f in df.columns if f not in ['SK_ID_CURR']
             and f not in to_drop]

    return feats
