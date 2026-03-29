import json
import os
import pathlib
import ssl

import paho.mqtt.client as paho
import polars as pl
from loguru import logger
from paho.mqtt.enums import MQTTProtocolVersion

# ---------------------------------------------------------------------------- #
#                                   CALLBACKS                                  #
# ---------------------------------------------------------------------------- #


def on_connect(client, userdata, flags, rc):
    logger.info("Connected with result code " + str(rc))


# ---------------------------------------------------------------------------- #
#                                 CONFIGURATION                                #
# ---------------------------------------------------------------------------- #


class Config:
    def __init__(self):
        self.meter_id = os.environ["METER_ID"]

        self.mqtt_host = os.environ["METER_MQTT_HOST"]
        self.mqtt_port = int(os.environ["METER_MQTT_PORT"])
        self.mqtt_topic = os.environ["METER_MQTT_TOPIC"]
        self.mqtt_username = os.environ["METER_MQTT_USERNAME"]
        self.mqtt_password = os.environ["METER_MQTT_PASSWORD"]

        self.ca_cert = os.environ.get("METER_CA_CERT")
        self.certfile = os.environ.get("METER_CERTFILE")
        self.keyfile = os.environ.get("METER_KEYFILE")

    def tls_credentials_available(self) -> bool:
        return all([self.ca_cert, self.certfile, self.keyfile])


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #


def main() -> None:
    config = Config()

    meter_data = load_data(config)

    logger.info("Setting up MQTT client...")
    client = paho.Client(protocol=MQTTProtocolVersion.MQTTv5)
    client.on_connect = on_connect

    if config.tls_credentials_available():
        logger.info("Setting TLS...")
        logger.debug(
            {
                "CA_CERT": config.ca_cert,
                "CERTFILE": config.certfile,
                "KEYFILE": config.keyfile,
            }
        )
        client.tls_set(
            config.ca_cert,
            config.certfile,
            config.keyfile,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )
    else:
        logger.warning(
            "TLS credentials not found in environment variables. Connecting without TLS."
        )
        client.username_pw_set(config.mqtt_username, config.mqtt_password)

    logger.info("Connecting to broker...")
    client.connect(config.mqtt_host, config.mqtt_port, 60)

    client.loop_start()

    for row in meter_data.iter_rows(named=True):
        client.publish(
            config.mqtt_topic,
            json.dumps(row, default=str),
        )

    client.loop_stop()
    client.disconnect()


def load_data(config: Config) -> pl.DataFrame:
    logger.info("Reading data...")
    meter_file = pathlib.Path(f"/meter/data/{config.meter_id}.parquet")
    meter_data = pl.scan_parquet(meter_file)
    meter_data = meter_data.with_columns(pl.col("datetime"))
    meter_data = meter_data.collect(engine="streaming")
    logger.info("Data read!")
    logger.debug(meter_data)
    return meter_data


if __name__ == "__main__":
    main()
