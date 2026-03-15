from sktime.split.fh import ForecastingHorizonSplitter
from sktime.utils.plotting import plot_series

from .config import params
from .dataset import load_dataset
from .forecasters import create_forecaster


def train():
    X, y = load_dataset([2], None, None)

    fh = list(range(1, 25))
    cv = ForecastingHorizonSplitter(fh=fh)

    for train_idx, test_idx in cv.split(y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    forecaster = create_forecaster(
        params["general"]["model"], params["models"]["cnn3d"]
    )

    forecaster.fit(y=y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(X=X_test, fh=fh)

    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
