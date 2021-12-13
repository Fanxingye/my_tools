from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import pandas as pd


def drop_data(df_all, target_name):
    df_all = df_all.drop_duplicates()
    row_treshold = int(df_all.shape[1] * 0.9)
    col_treshold = int(df_all.shape[0] * 0.6)
    df_all = df_all.dropna(axis=0, thresh=row_treshold)
    df_all = df_all.dropna(axis=1, thresh=col_treshold)
    df_all = df_all.reset_index(drop=True)
    for col in df_all.columns:
        if 'id' in col.lower() and df_all[col].nunique() == len(
                df_all) and col != target_name:
            df_all = df_all.drop(col, axis=1)
    return df_all


def scaler(df_x, columns=None, method='standard'):
    """
    Scaler numerical features.
    method: 'standard'、'minmax'、'maxabs' or 'robust'
    """

    if columns is None:
        num_col_names = df_x.select_dtypes(include=['float', 'int']).columns
    else:
        num_col_names = columns
    if method == 'standard':
        sc = StandardScaler()
    elif method == 'minmax':
        sc = MinMaxScaler()
    elif method == 'maxabs':
        sc = MaxAbsScaler()
    elif method == 'robust':
        sc = RobustScaler()
    else:
        raise Exception(
            "method not in ['standard', 'minmax', 'maxabs', 'robust']")
    df_x[num_col_names] = sc.fit_transform(df_x[num_col_names])
    return df_x


def imputing_numerical_missing_features(df_x, columns=None, method='zero'):
    """
    Impute numerical features.
    method: 'zero'、'mean'、'sum'、'mode'、'bfill' or 'ffill'
    """

    if columns is None:
        num_col_names = df_x.select_dtypes(include=['float', 'int']).columns
    else:
        num_col_names = columns
    sub_df = df_x[num_col_names]
    if method == 'zero':
        sub_df = sub_df.fillna(0)
    elif method == 'mean':
        sub_df = sub_df.fillna(sub_df.mean())
    elif method == 'sum':
        sub_df = sub_df.fillna(sub_df.sum())
    elif method == 'mode':
        sub_df = sub_df.fillna(sub_df.mode()[0])
    elif method == 'bfill':
        sub_df = sub_df.fillna(method='bfill', axis=1)
    elif method == 'ffill':
        sub_df = sub_df.fillna(method='ffill', axis=1)
    else:
        raise Exception(
            "method not in ['zero', 'mean', 'sum', 'mode', 'bfill', 'ffill']")
    df_x[num_col_names] = sub_df
    return df_x


def imputing_category_missing_features(df_x, columns=None, method='None'):
    """
    Impute category features.
    method: 'None'、'bfill' or 'ffill'
    """
    if columns is None:
        cat_col_names = df_x.select_dtypes(include='object').columns
    else:
        cat_col_names = columns
    sub_df = df_x[cat_col_names]
    if method == 'None':
        sub_df = sub_df.fillna("None")
    elif method == 'bfill':
        sub_df = sub_df.fillna(method='bfill', axis=1)
    elif method == 'ffill':
        sub_df = sub_df.fillna(method='ffill', axis=1)
    else:
        raise Exception("method not in ['None', 'bfill', 'ffill']")
    df_x[cat_col_names] = sub_df
    return df_x


def detect_outliers(df_all, column):
    Q1 = df_all[column].quantile(0.25)
    Q3 = df_all[column].quantile(0.75)
    IQR = 3 * (Q3 - Q1)
    val_low = Q1 - IQR
    val_up = Q3 + IQR
    return val_low, val_up


def drop_outliers(df_all, label, columns=None):
    if columns is None:
        num_col_names = df_all.select_dtypes(include=['float', 'int']).columns
        if label in num_col_names:
            num_col_names.remove(label)
    else:
        num_col_names = columns
    if type(num_col_names) == str:
        val_low, val_up = detect_outliers(df_all, column=num_col_names)
        index = df_all[(df_all[num_col_names] < val_low) |
                       (df_all[num_col_names] > val_up)].index
        df_all = df_all.drop(index).reset_index(inplace=False).drop('index',
                                                                    axis=1)
    elif type(num_col_names) == list:
        for col in num_col_names:
            val_low, val_up = detect_outliers(df_all, column=col)
            index = df_all[(df_all[col] < val_low) |
                           (df_all[col] > val_up)].index
            df_all = df_all.drop(index).reset_index(inplace=False).drop(
                'index', axis=1)
    else:
        raise Exception("The type of columns must be str or list.")
    return df_all


def upsample(data, label, max_num):
    """ upsample few shot data"""
    vc = data[label].value_counts()
    less_than_ten = vc[vc < max_num].index.tolist()
    data_app = pd.DataFrame()
    for l in less_than_ten:
        num = vc[l]
        n_resample = max_num - num
        for times in range(n_resample):
            data_re = data[data[label] == l].sample(n=1)
            data_app = pd.concat([data_app, data_re], axis=0)
    data_more = pd.concat([data, data_app], axis=0).reset_index(
        inplace=False).drop('index', axis=1)

    return data_more
