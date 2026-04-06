import os
import pathlib

import pandas as pd
import psycopg
from loguru import logger
from sktime.forecasting.compose import ForecastingPipeline


def forecast():
    logger.info("Starting forecast pipeline...")
    forecaster = load_forecaster(model_name="cnn3d")
    logger.debug(f"Forecaster cutoff: {forecaster.cutoff}")

    latest_panel = load_latest_panel_data(forecaster.cutoff)
    if not latest_panel.empty:
        logger.info("Updating forecaster with latest measurements from DB...")
        forecaster.update(latest_panel)

    predictions = predict_forecasts(forecaster)
    save_forecasts(predictions)
    logger.info("Forecasting completed.")


def load_forecaster(model_name: str | None = None) -> ForecastingPipeline:
    logger.info("Loading forecaster...")
    forecaster: ForecastingPipeline = ForecastingPipeline.load_from_path(
        pathlib.Path(f"/forecaster/{model_name}.zip")
    )
    logger.info("Forecaster loaded successfully!")
    logger.debug(f"Forecaster details: {forecaster}")
    return forecaster


def load_latest_panel_data(cutoff):
    cutoff_ts = pd.Timestamp(cutoff)
    query = """
        SELECT misuratore_id, timestamp, consumo
        FROM misura
        WHERE timestamp > %s
        ORDER BY misuratore_id, timestamp
    """
    with psycopg.connect(**db_settings()) as conn:
        df = pd.read_sql(query, conn, params=(cutoff_ts,), parse_dates=["timestamp"])

    if df.empty:
        logger.info("No new measurements found after cutoff %s", cutoff_ts)
        return pd.Series(dtype="float64")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    series = (
        df.set_index(["misuratore_id", "timestamp"])["consumo"]
        .astype(float)
        .sort_index()
    )
    series.index = series.index.set_names(["misuratore_id", "timestamp"])
    return series


def predict_forecasts(forecaster):
    fh = getattr(forecaster, "fh", None)
    if fh is None:
        fh = getattr(forecaster, "_fh", None)
    if fh is None:
        fh = 1

    logger.info("Predicting with forecasting horizon %s", fh)
    predictions = forecaster.predict(fh=fh)

    if isinstance(predictions, pd.DataFrame) and predictions.shape[1] == 1:
        predictions = predictions.iloc[:, 0]

    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions)

    if isinstance(predictions.index, pd.MultiIndex):
        predictions.index = predictions.index.set_names(["misuratore_id", "timestamp"])
    else:
        raise ValueError("Expected forecasting output with a MultiIndex")

    predictions.name = "consumo"
    return predictions


def save_forecasts(predictions: pd.Series):
    if predictions.empty:
        logger.warning("No forecast values to save.")
        return

    forecast_df = predictions.reset_index()
    forecast_df.columns = ["misuratore_id", "timestamp", "consumo"]

    insert_sql = """
        INSERT INTO misura_forecast (misuratore_id, timestamp, consumo)
        VALUES (%s, %s, %s)
        ON CONFLICT (misuratore_id, timestamp)
        DO UPDATE SET consumo = EXCLUDED.consumo
    """

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS misura_forecast (
            misuratore_id BIGINT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            consumo DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (misuratore_id, timestamp)
        )
    """

    rows = [
        (int(row.misuratore_id), row.timestamp.to_pydatetime(), float(row.consumo))
        for row in forecast_df.itertuples(index=False)
    ]

    with psycopg.connect(**db_settings()) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.executemany(insert_sql, rows)
        conn.commit()

    logger.info("Saved %d forecast rows to misura_forecast", len(rows))


def db_settings():
    settings = {
        "host": os.environ["FORECASTER_DB_HOST"],
        "port": int(os.environ.get("FORECASTER_DB_PORT", 5432)),
        "dbname": os.environ["FORECASTER_DB_DATABASE"],
        "user": os.environ["FORECASTER_DB_USER"],
        "password": os.environ["FORECASTER_DB_PASSWORD"],
    }

    sslrootcert = os.environ.get("FORECASTER_DB_SSLROOTCERT")
    sslcert = os.environ.get("FORECASTER_DB_SSLCERT")
    sslkey = os.environ.get("FORECASTER_DB_SSLKEY")
    sslmode = os.environ.get("FORECASTER_DB_SSLMODE", "verify-full")

    if sslrootcert:
        settings["sslrootcert"] = sslrootcert
    if sslcert:
        settings["sslcert"] = sslcert
    if sslkey:
        settings["sslkey"] = sslkey
    if sslrootcert or sslcert or sslkey:
        settings["sslmode"] = sslmode

    return settings
