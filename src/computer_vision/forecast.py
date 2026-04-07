import os
import pathlib

import pandas as pd
import psycopg
from apscheduler.schedulers.blocking import BlockingScheduler
from get_docker_secret import get_docker_secret
from loguru import logger
from sktime.forecasting.compose import ForecastingPipeline


def forecast():
    db_settings = get_db_settings()
    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_forecast_pipeline,
        "cron",
        kwargs={"db_settings": db_settings},
        minute=0,
        second=0,
        misfire_grace_time=None,
    )
    logger.info("Starting forecast scheduler...")
    scheduler.start()


def run_forecast_pipeline(db_settings: dict | None = None):
    logger.info("Starting forecast pipeline...")
    forecaster = load_forecaster(model_name="cnn3d")
    logger.debug(f"Forecaster cutoff: {forecaster.cutoff}")

    latest_panel = load_latest_panel_data(forecaster.cutoff, db_settings)
    if not latest_panel.empty:
        logger.info("Updating forecaster with latest measurements from DB...")
        y = latest_panel
        X = pd.DataFrame(index=y.index)
        forecaster.update(y, X, update_params=False)
        logger.debug(
            f"Forecaster updated with latest panel data. New cutoff: {forecaster.cutoff}"
        )

    new_X = create_future_panel_data(X)
    predictions = predict_forecasts(forecaster, X=new_X)
    save_forecasts(predictions)
    logger.info("Forecasting completed.")


def load_forecaster(model_name: str | None = None) -> ForecastingPipeline:
    logger.info("Loading forecaster...")
    forecaster_path = os.environ.get("FORECASTER_MODEL_PATH")
    if forecaster_path:
        logger.info(f"Loading forecaster from environment path: {forecaster_path}")
        forecaster = ForecastingPipeline.load_from_path(pathlib.Path(forecaster_path))
        logger.info("Forecaster loaded successfully from environment path!")
        logger.debug(f"Forecaster details: {forecaster}")
        return forecaster
    else:
        logger.info(
            "No environment path set for forecaster. Loading from default path..."
        )
        forecaster: ForecastingPipeline = ForecastingPipeline.load_from_path(
            pathlib.Path(f"/forecaster/{model_name}.zip")
        )
    logger.info("Forecaster loaded successfully!")
    logger.debug(f"Forecaster details: {forecaster}")
    return forecaster


def load_latest_panel_data(cutoff, db_settings: dict) -> pd.DataFrame:
    query = """
        SELECT misuratore_id, timestamp, consumo
        FROM misura
        WHERE timestamp > %s
        ORDER BY misuratore_id, timestamp
    """
    with psycopg.connect(**(db_settings)) as conn:
        df = pd.read_sql(
            query,
            conn,
            params=(cutoff.strftime("%Y-%m-%dT%H:%M:%S")[0],),
            parse_dates=["timestamp"],
        )

    if df.empty:
        logger.info("No new measurements found after cutoff %s", cutoff)
        return pd.Series(dtype="float64")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index(["misuratore_id", "timestamp"]).sort_index()
    logger.info("Loaded %d new measurements from DB", len(df))
    logger.debug(f"Latest panel data sample:\n{df}")
    logger.debug(f"Panel data for each meter:\n{df.groupby(level=0).size()}")
    for misuratore_id, group in df.groupby(level=0):
        logger.debug(
            f"Meter {misuratore_id} - First timestamp: {group.index.get_level_values(1).min()}, "
            f"Last timestamp: {group.index.get_level_values(1).max()}, "
            f"Number of records: {len(group)}"
        )
    df.index = df.index.set_names(["id", "datetime"])
    df = df.rename(columns={"consumo": "value"})
    return df


def create_future_panel_data(X):
    freq = pd.Timedelta("15min")
    meters = (7001, 15805, 18052, 50176, 115138)

    ts_indexes = []
    for meter in meters:
        df = X.xs(meter)
        last_ts = df.index.max()
        last_ts = last_ts + freq
        index = pd.date_range(
            start=last_ts, end=last_ts + pd.Timedelta(hours=23, minutes=45), freq=freq
        )
        ts_indexes.append(index)

    index = pd.MultiIndex.from_tuples(
        [(meter, ts) for meter, ts_list in zip(meters, ts_indexes) for ts in ts_list],
        names=["id", "datetime"],
    )
    new_X = pd.DataFrame(index=index)
    return new_X


def predict_forecasts(forecaster, X):
    logger.debug(f"Predicting forecasts for index:\n{X.index}")
    for misuratore_id, group in X.groupby(level=0):
        logger.debug(
            f"Meter {misuratore_id} - First timestamp: {group.index.get_level_values(1).min()}, "
            f"Last timestamp: {group.index.get_level_values(1).max()}, "
            f"Number of records: {len(group)}"
        )

    predictions = forecaster.predict(X=X)

    return predictions


def save_forecasts(predictions: pd.DataFrame, db_settings: dict):
    if predictions.empty:
        logger.warning("No forecast values to save.")
        return

    model_id = os.environ.get("FORECASTER_MODEL_ID", 1)
    forecast_time = pd.Timestamp.utcnow()

    forecast_df = predictions.reset_index()
    forecast_df = forecast_df.rename(
        columns={"id": "misuratore_id", "datetime": "timestamp", "value": "consumo"}
    )
    forecast_df.columns = ["misuratore_id", "timestamp", "consumo"]
    forecast_df["misuratore_id"] = forecast_df["misuratore_id"].astype(int)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["consumo"] = forecast_df["consumo"].astype(float)

    insert_previsione_sql = """
        INSERT INTO previsione (modello_id, misuratore_id, timestamp)
        VALUES (%s, %s, %s)
        RETURNING id
    """

    insert_step_sql = """
        INSERT INTO step (id_previsione, timestamp, valore)
        VALUES (%s, %s, %s)
    """

    with psycopg.connect(**db_settings) as conn:
        with conn.cursor() as cur:
            for misuratore_id, group in forecast_df.groupby("misuratore_id"):
                cur.execute(
                    insert_previsione_sql,
                    (model_id, misuratore_id, forecast_time.to_pydatetime()),
                )
                previsione_id = cur.fetchone()[0]

                step_rows = [
                    (previsione_id, row.timestamp.to_pydatetime(), float(row.consumo))
                    for row in group.itertuples(index=False)
                ]
                cur.executemany(insert_step_sql, step_rows)

        conn.commit()

    logger.info(
        "Saved %d forecast steps for %d previsione records",
        len(forecast_df),
        forecast_df["misuratore_id"].nunique(),
    )


def get_db_settings():
    settings = {
        "host": os.environ["FORECASTER_DB_HOST"],
        "port": int(os.environ.get("FORECASTER_DB_PORT", 5432)),
        "dbname": get_docker_secret("FORECASTER_DB_NAME"),
        "user": get_docker_secret("FORECASTER_DB_USER"),
        "password": get_docker_secret("FORECASTER_DB_PASSWORD"),
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
