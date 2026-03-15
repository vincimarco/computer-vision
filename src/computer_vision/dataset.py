from datetime import datetime
from typing import Literal

import pandas as pd
import polars as pl

from .config import FINAL_DATA_DIR


def load_dataset(
    customer_ids: list[int],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = load_customer_time_series(customer_ids, start_date, end_date)
    y = y.groupby("id").apply(lambda g: g.loc[g.name].asfreq("15min"))

    X = pd.DataFrame(index=y.index)
    X = X.groupby("id").apply(lambda g: g.loc[g.name].asfreq("15min"))

    return X, y


def load_customer_time_series(
    customer_ids: list[int],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers_*.parquet")

    lf = lf.filter(pl.col("id").is_in(customer_ids))

    if start_date is not None:
        lf = lf.filter(pl.col("datetime") >= start_date)
    if end_date is not None:
        lf = lf.filter(pl.col("datetime") <= end_date)

    df = lf.collect(engine="streaming").to_pandas()
    df = df.set_index(["id", "datetime"])
    return df


def load_customers(customer_ids: list[int]) -> pd.DataFrame:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers.parquet")

    lf = lf.filter(pl.col("customer_id").is_in(customer_ids))

    df = lf.collect(engine="streaming").to_pandas()
    df = df.set_index("customer_id")

    return df


# ---------------------------------------------------------------------------- #
#                                  STATISTICS                                  #
# ---------------------------------------------------------------------------- #


def get_customer_ids(
    by_tension_class: Literal[
        "BT 230 V",
        "BT 400 V",
        "MT 6.4 KV",
        "MT 15 KV",
        "MT 22 KV",
        "MT 31.5 KV",
        "MT 63 KV",
        "*",
        None,
    ] = "*",
) -> list[int]:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers.parquet")

    if by_tension_class is None:
        lf = lf.filter(pl.any_horizontal(pl.col("tension").is_null()))

    if by_tension_class != "*" and by_tension_class is not None:
        lf = lf.filter(pl.col("tension") == by_tension_class)

    lf = lf.select("customer_id").unique().sort("customer_id")

    return lf.collect().to_series().to_list()


def count_unique_in_column(
    column_name: Literal["customer_id", "tension", "power"],
) -> int:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers.parquet")
    lf = lf.select(column_name).unique().count()
    return lf.collect().item()


def count_customers_per_class(column_name: Literal["tension", "power"]) -> pl.DataFrame:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers.parquet")
    lf = lf.group_by(column_name).len().sort(column_name)
    return lf.collect()
