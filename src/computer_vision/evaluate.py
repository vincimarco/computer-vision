import pandas as pd
import plotly.graph_objects as go
import yaml
from loguru import logger
from rich.console import Console
from sktime.forecasting.model_evaluation import evaluate
from sktime.split.expandingwindow import ExpandingWindowSplitter

from computer_vision.utils import create_fh

from .config import EVAL_DIR, params
from .dataset import load_dataset
from .forecasters import create_forecaster
from .metrics import metrics


def eval():
    console = Console()
    logger.info("Starting evaluation process")

    with console.status("Loading dataset..."):
        customer_ids = params["general"]["customer_ids"]
        logger.info(f"Loading dataset for customer_ids: {customer_ids}")
        X, y = load_dataset(customer_ids=customer_ids)
    logger.info(f"Dataset loaded successfully - X shape: {X.shape}, y shape: {y.shape}")

    fh = create_fh(params["general"]["fh"])
    logger.debug(f"Forecast horizon: {fh}")

    forecaster = create_forecaster(
        params["general"]["model"], params["models"][params["general"]["model"]]
    )
    logger.info(f"Forecaster created: {params['general']['model']}")

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

    logger.info("Starting model evaluation...")
    results = evaluate(
        forecaster=forecaster,
        cv=cv,
        y=y,
        X=X,
        strategy="no-update_params",
        scoring=list(metrics.values()),
        return_data=True,
        error_score="raise",
    )
    logger.info("Model evaluation completed")

    metric_columns = [
        column for column in results.columns if column.startswith("test_")
    ]
    logger.debug(f"Metric columns: {metric_columns}")

    metrics_results = results[metric_columns]
    avg_metrics_results = metrics_results.mean().to_frame(name="mean").T
    logger.info(f"Average metrics: {avg_metrics_results.to_dict(orient='records')[0]}")

    yaml.dump(
        avg_metrics_results.to_dict(orient="records")[0],
        (EVAL_DIR / "metrics.yaml").open("w"),
    )
    logger.info(f"Metrics saved to {EVAL_DIR / 'metrics.yaml'}")

    console.print("Evaluation results:")
    console.print(metrics_results)

    OUTPUT_DIR = EVAL_DIR / "out"
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Output directory created: {OUTPUT_DIR}")

    y_train = results["y_train"][0]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    logger.debug("Extracted y_train, y_test, y_pred from results")

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

        y_test_id = y_test.apply(lambda x: x.xs(id))
        y_pred_id = y_pred.apply(lambda x: x.xs(id))

        # Merge all folds into single Series/DataFrame
        y_test_merged = pd.concat(y_test_id.tolist())
        y_pred_merged = pd.concat(y_pred_id.tolist())

        logger.debug(f"Customer {id} - Merged y_test shape: {y_test_merged.shape}")
        logger.debug(f"Customer {id} - Merged y_pred shape: {y_pred_merged.shape}")

        # Convert to DataFrame if Series
        if isinstance(y_test_merged, pd.Series):
            y_test_merged = y_test_merged.to_frame(name="value")
        if isinstance(y_pred_merged, pd.Series):
            y_pred_merged = y_pred_merged.to_frame(name="value")

        fig.add_scatter(
            x=y_test_merged.index,
            y=y_test_merged.iloc[:, 0],
            mode="lines",
            name="y_test",
            line=dict(color="#f58231"),
        )

        fig.add_scatter(
            x=y_pred_merged.index,
            y=y_pred_merged.iloc[:, 0],
            mode="lines",
            name="y_pred",
            line=dict(color="#7fcdbb"),
        )

        output_file = OUTPUT_DIR / f"prediction_{id}.html"
        fig.write_html(output_file)
        logger.info(f"Prediction plot saved for customer {id} to {output_file}")

    logger.info("Evaluation process completed successfully")
