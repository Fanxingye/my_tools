import pandas as pd
from label_encoder import LabelEncoder
from gbdt_feature import LightGBMFeatureTransformer
from groupby_feature import get_category_columns, groupby_generate_feature
from sklearn.feature_selection import SelectFromModel


def get_baseline_total_data(df):
    return pd.get_dummies(df).fillna(0)


def get_groupby_total_data(
        df,
        target_name,
        threshold=0.9,
        k=5,
        methods=['min', 'max', 'sum', 'mean', 'std', 'count'],
        reserve=True):
    generated_feature = groupby_generate_feature(df, target_name, threshold, k,
                                                 methods, reserve)
    return generated_feature


def generate_cross_feature(df: pd.DataFrame, crossed_cols, keep_all=True):
    df_cc = df.copy()
    crossed_colnames = []
    for cols in crossed_cols:
        for c in cols:
            df_cc[c] = df_cc[c].astype('str')
        colname = '_'.join(cols)
        df_cc[colname] = df_cc[list(cols)].apply(lambda x: '_'.join(x), axis=1)

        crossed_colnames.append(colname)
    if keep_all:
        return df_cc
    else:
        return df_cc[crossed_colnames]


def get_cross_columns(category_cols):
    crossed_cols = []
    for i in range(0, len(category_cols) - 1):
        for j in range(i + 1, len(category_cols)):
            crossed_cols.append((category_cols[i], category_cols[j]))
    return crossed_cols


def get_GBDT_total_data(df, target_name, task='classification'):
    cat_col_names = get_category_columns(df, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data = label_encoder.fit_transform(df)
    clf = LightGBMFeatureTransformer(task=task,
                                     categorical_feature=cat_col_names,
                                     params={
                                         'n_estimators': 100,
                                         'max_depth': 3
                                     })
    X = total_data.drop(target_name, axis=1)
    y = total_data[target_name]
    clf.fit(X, y)
    X_enc = clf.dense_transform(X, keep_original=False)
    total_data = pd.concat([X_enc, y], axis=1)
    return total_data


def get_groupby_GBDT_total_data(groupby_df,
                                target_name,
                                task='classification',
                                keep_original=False):
    """Get all data after doing groupby operator and put in GBDT"""
    cat_col_names = get_category_columns(groupby_df, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data = label_encoder.fit_transform(groupby_df)
    clf = LightGBMFeatureTransformer(task=task,
                                     categorical_feature=cat_col_names,
                                     params={
                                         'n_estimators': 100,
                                         'max_depth': 3
                                     })
    X = total_data.drop(target_name, axis=1)
    y = total_data[target_name]
    clf.fit(X, y)
    X_enc = clf.dense_transform(X, keep_original=keep_original)
    total_data = pd.concat([X_enc, y], axis=1)
    return total_data


def select_feature(df, target_name, estimator):
    X = df.drop(target_name, axis=1)
    y = df[target_name]
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    support = selector.get_support()
    col_names = X.columns[support]
    X = selector.transform(X)
    X = pd.DataFrame(X, columns=col_names)
    total_data = X
    total_data[target_name] = y
    return total_data
