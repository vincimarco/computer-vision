import pathlib

from loguru import logger
from sktime.forecasting.compose import ForecastingPipeline


def forecast():
    print("Forecasting...")

    forecaster = load_forecaster(model_name="cnn3d")
    print(forecaster.cutoff)


def load_forecaster(model_name: str | None = None) -> ForecastingPipeline:
    logger.info("Loading forecaster...")
    forecaster: ForecastingPipeline = ForecastingPipeline.load_from_path(
        pathlib.Path(f"/forecaster/{model_name}.zip")
    )
    logger.info("Forecaster loaded successfully!")
    logger.debug(f"Forecaster details: {forecaster}")
    return forecaster
