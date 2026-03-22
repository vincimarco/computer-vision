import pandas as pd
import plotly.graph_objects as go
import yaml
from loguru import logger
from rich.console import Console
from sktime.split.expandingwindow import ExpandingWindowSplitter

from computer_vision.utils import create_fh

from .config import TEST_OUTPUT_DIR, TRAIN_OUTPUT_DIR, params
from .dataset import load_dataset
from .forecasters import load_forecaster
from .metrics import metrics


def test():
    console = Console()
    logger.info("Starting evaluation process")

    with console.status("Loading dataset..."):
        customer_ids = params["general"]["customer_ids"]
        logger.info(f"Loading dataset for customer_ids: {customer_ids}")
        X, y = load_dataset(customer_ids=customer_ids)
    logger.info(f"Dataset loaded successfully - X shape: {X.shape}, y shape: {y.shape}")

    fh = create_fh(params["general"]["fh"])
    logger.debug(f"Forecast horizon: {fh}")

    forecaster_path = TRAIN_OUTPUT_DIR / f"{params['general']['model']}.zip"
    forecaster = load_forecaster(forecaster_path)
    logger.info(f"Forecaster loaded: {params['general']['model']}")

    cutoff_str: str = params["general"]["cutoff"]
    cutoff = pd.to_datetime(cutoff_str, format="ISO8601")
    logger.debug(f"Cutoff datetime: {cutoff}")

    initial_window = cutoff - pd.to_datetime("2019-01-01T00:00:00-03:00")
    initial_window = initial_window // pd.Timedelta(
        "15min"
    )  # convert to number of days
    logger.debug(f"Initial window (in 15min intervals): {initial_window}")

    step = params["general"]["step"]
    logger.debug(f"Step length: {step}")

    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=initial_window,
        step_length=step,
    )
    logger.info("ExpandingWindowSplitter created")

    metrics_per_step: list[dict[str, float]] = []
    first_y_train = None
    ys_test = []
    ys_pred = []
    for i, (train_idx, test_idx) in enumerate(cv.split(y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        forecaster.update(y_train, X_train, update_params=False)
        y_pred = forecaster.predict(fh=fh, X=X_test)

        if first_y_train is None:
            first_y_train = y_train
        ys_test.append(y_test)
        ys_pred.append(y_pred)

        metric_values = {
            metric_name: metric_func(y_true=y_test, y_pred=y_pred, y_train=y_train)
            for metric_name, metric_func in metrics.items()
        }
        metrics_per_step.append(metric_values)

    ys_test: pd.DataFrame = pd.concat(ys_test)
    ys_test = ys_test.sort_index(level=0)
    ys_pred: pd.DataFrame = pd.concat(ys_pred)
    ys_pred = ys_pred.sort_index(level=0)

    plot_preds(customer_ids, first_y_train, ys_test, ys_pred)

    metrics_df = pd.DataFrame(metrics_per_step)
    logger.debug(metrics_df)
    avg_metrics = metrics_df.mean().to_frame().T
    logger.debug(avg_metrics)
    yaml.dump(
        avg_metrics.to_dict(orient="list"),
        (TEST_OUTPUT_DIR / "metrics.yaml").open("w"),
    )


def plot_preds(
    customer_ids: list[int],
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
):
    for id in customer_ids:
        logger.info(f"Processing visualizations for customer {id}")
        title = f"Customer {id} - Forecasts"
        fig = go.Figure()
        fig.update_layout(
            {"title": title, "xaxis_title": "Time", "yaxis_title": "Consumption (Wh)"}
        )

        y_train_id = y_train.xs(id)
        logger.debug(
            f"Customer {id} training data - Type: {type(y_train_id)}, Shape: {y_train_id.shape}"
        )
        logger.debug(
            f"Customer {id} training data columns: {y_train_id.columns.tolist() if hasattr(y_train_id, 'columns') else 'N/A'}"
        )
        logger.debug(f"Customer {id} training data head:\n{y_train_id.head()}")

        fig.add_scatter(
            x=y_train_id.index,
            y=y_train_id.loc[:, "value"],
            mode="lines",
            name="y_train",
            line=dict(color="#2c7fb8"),
        )

        y_test_id = y_test.xs(id)
        y_pred_id = y_pred.xs(id)

        logger.debug(f"Customer {id} - Merged y_test shape: {y_test_id.shape}")
        logger.debug(f"Customer {id} - Merged y_pred shape: {y_pred_id.shape}")

        fig.add_scatter(
            x=y_test_id.index,
            y=y_test_id.iloc[:, 0],
            mode="lines",
            name="y_test",
            line=dict(color="#f58231"),
        )

        fig.add_scatter(
            x=y_pred_id.index,
            y=y_pred_id.iloc[:, 0],
            mode="lines",
            name="y_pred",
            line=dict(color="#7fcdbb"),
        )

        output_dir = TEST_OUTPUT_DIR / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"prediction_{id}.html"
        fig.write_html(output_file)
        logger.info(f"Prediction plot saved for customer {id} to {output_file}")

    logger.info("Evaluation process completed successfully")
