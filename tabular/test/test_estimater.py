import sys
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from models.estimator import SklearnEstimator, XLGBEstimator
from models.configuration import config

if __name__ == "__main__":
    iris = load_iris(as_frame=True)
    data_x, data_y = iris['data'], iris['target']
    name = "lightgbm"
    task = "classification"
    params = config.get(name + '_' + task)
    estimator = XLGBEstimator(name, params)
    estimator.fit(data_x, data_y)
    pred = estimator.predict(data_x)
    acc = accuracy_score(data_y, pred)
    print(acc)

