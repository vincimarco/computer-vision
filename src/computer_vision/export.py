import pandas as pd

from .config import EXPORT_DATA_DIR, params
from .dataset import load_dataset


def export():
    start_date = params["general"]["cutoff"]
    start_date = pd.to_datetime(start_date, format="ISO8601", utc=True).tz_convert(
        "America/Montevideo"
    )
    end_date = params["test"]["end_date"]
    end_date = pd.to_datetime(end_date, format="ISO8601", utc=True).tz_convert(
        "America/Montevideo"
    )

    X, y = load_dataset(
        params["general"]["customer_ids"],
        start_date=start_date,
        end_date=end_date,
    )

    for id in params["general"]["customer_ids"]:
        group = y.xs(id)
        group = group.dropna()
        group = group.sort_index()
        group.to_parquet(f"{EXPORT_DATA_DIR}/{id}.parquet")
    # Export logic here
    pass
