import pandas as pd
from loguru import logger
from rich.console import Console

from computer_vision.utils import create_fh

from .config import TRAIN_OUTPUT_DIR, params
from .dataset import load_dataset
from .forecasters import create_forecaster


def train():
    console = Console()
    logger.info("Starting evaluation process")

    cutoff_str: str = params["general"]["cutoff"]
    cutoff = pd.to_datetime(cutoff_str, format="ISO8601", utc=True).tz_convert(
        "America/Montevideo"
    )
    cutoff = cutoff - pd.Timedelta("15min")
    logger.debug(f"Cutoff datetime: {cutoff}")

    fh = create_fh(params["general"]["fh"])
    logger.debug(f"Forecast horizon: {fh}")

    with console.status("Loading dataset..."):
        customer_ids = params["general"]["customer_ids"]
        logger.info(f"Loading dataset for customer_ids: {customer_ids}")
        X, y = load_dataset(customer_ids=customer_ids, end_date=cutoff.to_pydatetime())
    logger.info(f"Dataset loaded successfully - X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Last index: {y.xs(params['general']['customer_ids'][0]).index.max()}")

    forecaster = create_forecaster(
        params["general"]["model"], params["models"][params["general"]["model"]]
    )
    logger.info(f"Forecaster created: {params['general']['model']}")

    forecaster.fit(y=y, X=X, fh=fh)

    output_dir = TRAIN_OUTPUT_DIR / params["general"]["model"]
    forecaster.save(output_dir)
    logger.info(f"Model saved in {output_dir}.zip")
