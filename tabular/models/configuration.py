config = {
    "randomforest_classification": {
        'n_estimators': 200,
        'random_state': 42,
    },
    "randomforest_regression": {
        'n_estimators': 200,
        'random_state': 42,
    },
    "adaboost_classification":{
        'n_estimators': 200,
        'learning_rate': 0.1,
        'random_state': 42
    },
    "adaboost_regression":{
        'n_estimators': 200,
        'learning_rate': 0.1,
        'random_state': 42
    },
    "gradientboost_classification":{
        'n_estimators': 200,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'random_state': 42,
        'verbose': -1
    },
    "gradientboost_regression":{
        'n_estimators': 200,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'random_state': 42,
        'verbose': -1
    },
    "svm_classification":{
        'C': 10000.0,
        'kernel': 'rbf',
        'gamma': 0.1,
        'tol': 1e-3,
        'probability': True,
        'shrinking': True,
        'class_weight': 'balanced',
        'random_state': 42,
    },
    "svm_regression":{
        'C': 10000.0,
        'kernel': 'rbf',
        'gamma': 0.1,
        'tol': 1e-3,
        'probability': True,
        'shrinking': True
    },
    "lr_classification":{
        'C': 10000.0,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'random_state': 42,
        'verbose': -1
    },
    "ridge_regression":{
        'alpha': 0.1,
        'normalize': False,
        'solver': 'lbfgs',
        'random_state': 42
    },
    "xgboost_classification":{
        'objective': 'multi:softmax',
        'eval_metric':  'logloss',
        'num_class': 4,
        'n_estimators': 30,
        'max_depth': 4,
        'subsample': 0.8,
        'gamma': 0.01,
        'eta': 0.01
    },
    "xgboost_regression":{
        'objective': 'reg:squarederror',
        'eval_metric':  'rmse',
        'n_estimators': 30,
        'learning_rate': 0.3,
        'max_depth': 4,
        'subsample': 0.8,
        'gamma': 0.01,
        'eta': 0.01
    },
    "lightgbm_classification":{
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'verbose': -1,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'learning_rate': 0.01,
        'n_jobs': -1,
        'device': 'cpu',
        'num_boost_round': 30,
        'early_stopping_round': 10,
        'verbosity': 1
    },
    "lightgbm_regression":{
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'multi_logloss',
        'verbose': -1,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'learning_rate': 0.01,
        'n_jobs': -1,
        'device': 'cpu',
        'num_boost_round': 30,
        'early_stopping_round': 10,
        'verbosity': 1
    }
}