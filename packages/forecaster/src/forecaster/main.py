import os

from loguru import logger
from sktime.forecasting.compose import ForecastingPipeline


class Config:
    def __init__(self):
        self.model_file = os.environ["FORECASTER_MODEL_FILE"]

        self.mqtt_host = os.environ["METER_MQTT_HOST"]
        self.mqtt_port = int(os.environ["METER_MQTT_PORT"])
        self.mqtt_username = os.environ["METER_MQTT_USERNAME"]
        self.mqtt_password = os.environ["METER_MQTT_PASSWORD"]

        self.ca_cert = os.environ.get("METER_CA_CERT")
        self.certfile = os.environ.get("METER_CERTFILE")
        self.keyfile = os.environ.get("METER_KEYFILE")


def main() -> None:
    config = Config()

    forecaster = load_forecaster(model_file=config.model_file)
    print(forecaster.cutoff\)


def load_forecaster(model_file: str | None = None) -> ForecastingPipeline:
    logger.info("Loading forecaster...")
    forecaster: ForecastingPipeline = ForecastingPipeline.load_from_path(model_file)
    logger.info("Forecaster loaded successfully!")
    logger.debug(f"Forecaster details: {forecaster}")
    return forecaster
