import pandas as pd
import yaml
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

    with console.status("Loading dataset..."):
        customer_ids = params["general"]["customer_ids"]
        X, y = load_dataset(customer_ids=customer_ids)
    console.print("Dataset loaded!")
    console.print(f"X shape: {X.shape}, y shape: {y.shape}")

    fh = create_fh(params["general"]["fh"])

    forecaster = create_forecaster(
        params["general"]["model"], params["models"][params["general"]["model"]]
    )

    cutoff_str: str = params["general"]["cutoff"]
    cutoff = pd.to_datetime(cutoff_str, format="ISO8601")
    initial_window = cutoff - pd.to_datetime("2019-01-01T00:00:00-03:00")
    initial_window = initial_window // pd.Timedelta(
        "15min"
    )  # convert to number of days
    step = params["general"]["step"]

    console.print(f"Using cutoff: {cutoff} and fh: {fh}")
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=initial_window,
        step_length=step,
    )
    # cv = ExpandingCutoffSplitter(
    #     cutoff=cutoff,
    #     fh=fh,
    #     step_length=step,
    # )

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

    metric_columns = [
        column for column in results.columns if column.startswith("test_")
    ]
    metrics_results = results[metric_columns]
    avg_metrics_results = metrics_results.mean().to_frame(name="mean").T
    yaml.dump(
        avg_metrics_results.to_dict(orient="records")[0],
        (EVAL_DIR / "metrics.yaml").open("w"),
    )

    console.print("Evaluation results:")
    console.print(metrics_results)

    # for customer_id in y_test.index.get_level_values(0).unique():
    #     fig, ax = plot_series(
    #         # y_train.xs(customer_id),
    #         y_test.xs(customer_id),
    #         y_pred.xs(customer_id),
    #         labels=["y_test", "y_pred"],
    #     )
    #     fig.savefig(EVAL_DIR / f"prediction_{customer_id}.png")
