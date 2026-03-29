import os
import pathlib

from loguru import logger
from sktime.forecasting.compose import ForecastingPipeline


class Config:
    def __init__(self):
        self.model_name = os.getenv("FORECASTER_MODEL_NAME", "model")


def main() -> None:
    config = Config()

    forecaster = load_forecaster(model_name=config.model_name)
    print(forecaster.cutoff)


def load_forecaster(model_name: str | None = None) -> ForecastingPipeline:
    logger.info("Loading forecaster...")
    forecaster: ForecastingPipeline = ForecastingPipeline.load_from_path(
        pathlib.Path(f"/forecaster/{model_name}.zip")
    )
    logger.info("Forecaster loaded successfully!")
    logger.debug(f"Forecaster details: {forecaster}")
    return forecaster
