from rich.console import Console
from sktime.split.temporal_train_test_split import temporal_train_test_split
from sktime.utils.plotting import plot_series

from .config import params
from .dataset import load_dataset
from .forecasters import create_forecaster


def train():
    console = Console()

    with console.status("Loading dataset..."):
        customer_ids = params["general"]["customer_ids"]
        X, y = load_dataset(customer_ids=customer_ids)
    console.print("Dataset loaded!")
    console.print(f"X shape: {X.shape}, y shape: {y.shape}")

    with console.status("Splitting dataset..."):
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y, X, test_size=params["general"]["test_size"]
        )
    console.print("Dataset split!")

    console.print(
        f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}"
        f", X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
    )

    fh = list(range(1, 25 * 4))

    forecaster = create_forecaster(
        params["general"]["model"], params["models"][params["general"]["model"]]
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
