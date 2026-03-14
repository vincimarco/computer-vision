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

    X = load_customers(customer_ids)

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


def get_customer_ids() -> list[int]:
    lf = pl.scan_parquet(FINAL_DATA_DIR / "customers.parquet")
    lf = lf.select("customer_id").unique()
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
