import pandas as pd
from loguru import logger
from rich.console import Console
from sktime.split.expandingwindow import ExpandingWindowSplitter

from computer_vision.utils import create_fh

from .config import TRAIN_OUTPUT_DIR, params
from .dataset import load_dataset
from .forecasters import load_forecaster


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

    for train_idx, test_idx in cv.split(y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        forecaster.update(y_train, X_train, update_params=False)
        y_pred = forecaster.predict(fh=fh, X=X_test)
        print(y_pred)
