from sktime.split.fh import ForecastingHorizonSplitter
from sktime.utils.plotting import plot_series

from .config import params
from .dataset import get_customer_ids, load_dataset
from .forecasters import create_forecaster


def train():
    customer_ids = get_customer_ids(
        [
            "BT 400 V",
        ]
    )
    X, y = load_dataset([3], None, None)

    fh = list(range(1, 25 * 4))
    cv = ForecastingHorizonSplitter(fh=fh)

    for train_idx, test_idx in cv.split(y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    forecaster = create_forecaster(
        params["general"]["model"], params["models"]["cnn3d"]
    )

    forecaster.fit(y=y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(X=X_test)

    for customer_id in y_test.index.get_level_values(0).unique():
        fig, ax = plot_series(
            # y_train.xs(customer_id),
            y_test.xs(customer_id),
            y_pred.xs(customer_id),
            labels=["y_test", "y_pred"],
        )
        fig.savefig(f"tmp/prediction_{customer_id}.png")
