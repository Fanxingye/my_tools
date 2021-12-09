import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from configuration import config


class SklearnEstimator(object):
    def __init__(self, name: str, task: str, params=None):
        self.name = name
        self.task = task
        self.params = params if params is not None else config[self.name + "_" + self.task]
        self.classification_dict = {
            "randomforest": RandomForestClassifier(),
            "adaboost": AdaBoostClassifier(),
            "gradientboost": GradientBoostingClassifier(),
            "svm": SVC(),
            "lr": LogisticRegression()
        }
        self.regression_dict = {
            "randomforest": RandomForestRegressor(),
            "adaboost": AdaBoostRegressor(),
            "gradientboost": GradientBoostingRegressor(),
            "svm": SVR(),
            "ridge": Ridge()
        }
        self.learner = self.get_estimator()

    def get_estimator(self):

        if self.task == "classification":
            estimator = self.classification_dict.get(self.name)
        elif self.task == "regression":
            estimator = self.regression_dict.get(self.name)
        else:
            raise Exception("task must be 'classification' or 'regression'")
        estimator = estimator.set_params(**self.params)
        return estimator

    def fit(self, X_data, Y_data, n_fold=1):
        X_data = pd.DataFrame(X_data)
        Y_data = pd.DataFrame(Y_data)

        sk = StratifiedShuffleSplit(
            n_splits=n_fold, test_size=0.2, train_size=0.8, random_state=2021)
        scores_train = []
        scores_val = []
        k = 0
        for train_ind, val_ind in sk.split(X_data, Y_data):
            k = k + 1
            train_x = X_data.iloc[train_ind].values
            train_y = Y_data.iloc[train_ind].values
            val_x = X_data.iloc[val_ind].values
            val_y = Y_data.iloc[val_ind].values

            self.learner.fit(train_x, train_y)
            pred_train = self.learner.predict(train_x)
            pred_val = self.learner.predict(val_x)

            score_train = accuracy_score(train_y, pred_train)
            score_val = accuracy_score(val_y, pred_val)
            scores_train.append(score_train)
            scores_val.append(score_val)
            print(f'fold: {k}, Accuracy of training: {score_train}')
            print(f'fold: {k}, Accuracy of validation: {score_val}')
        print(f'Mean Accuracy of training: {np.mean(scores_train)}')
        print(f'Mean Accuracy of validation: {np.mean(scores_val)}')
        return self

    def predict(self, X_test):
        return self.learner.predict(X_test)

    def predict_proba(self, X_test):
        if self.task == "classification":
            return self.learner.predict_proba(X_test)
        else:
            raise Exception("Regressor has no 'predict_proba' methods!")


class XLGBEstimator(object):
    def __init__(self, name: str, task: str, params=None):
        self.name = name
        self.task = task
        self.params = params if params is not None else config[self.name + "_" + self.task]

    def fit(self, X_data, Y_data, n_fold=1):
        X_data = pd.DataFrame(X_data)
        Y_data = pd.DataFrame(Y_data)
        print(f'num_class: {Y_data.value_counts()}')

        sk = StratifiedShuffleSplit(
            n_splits=n_fold, test_size=0.2, train_size=0.8, random_state=2021)
        scores_train = []
        scores_val = []

        k = 0
        for train_ind, val_ind in sk.split(X_data, Y_data):
            k = k + 1
            train_x = X_data.iloc[train_ind].values
            train_y = Y_data.iloc[train_ind].values
            val_x = X_data.iloc[val_ind].values
            val_y = Y_data.iloc[val_ind].values
            if self.name == 'xgboost':
                dtrain = xgb.DMatrix(train_x, label=train_y)
                dval = xgb.DMatrix(val_x, val_y)
                self.bst = xgb.train(self.params, dtrain)
                pred_train = self.bst.predict(dtrain)
                pred_val = self.bst.predict(dval)
            elif self.name == "lightgbm":
                dtrain = lgb.Dataset(train_x, train_y)
                dval = lgb.Dataset(val_x, val_y)
                self.bst = lgb.train(self.params,
                                     dtrain,
                                     valid_sets=dval)
                pred_train = self.bst.predict(train_x)
                pred_train = np.argmax(pred_train, axis=1)
                pred_val = self.bst.predict(val_x)
                pred_val = np.argmax(pred_val, axis=1)
            else:
                raise Exception("name must be 'xgboost' or 'lightgbm'!")

            score_train = accuracy_score(train_y, pred_train)
            score_val = accuracy_score(val_y, pred_val)
            scores_train.append(score_train)
            scores_val.append(score_val)
            print(f'fold: {k}, Accuracy of training: {score_train}')
            print(f'fold: {k}, Accuracy of validation: {score_val}')
        print(f'Mean Accuracy of training: {np.mean(scores_train)}')
        print(f'Mean Accuracy of validation: {np.mean(scores_val)}')
        return self

    def predict(self, X_test):
        if self.name == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            pred = self.bst.predict(dtest)
        elif self.name == "lightgbm":
            dtest = X_test
            pred = self.bst.predict(dtest)
            pred = np.argmax(pred, axis=1)
        else:
            raise Exception("name must be 'xgboost' or 'lightgbm'!")
        return pred
