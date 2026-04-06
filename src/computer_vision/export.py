import datetime

import pytz

from .config import EXPORT_DATA_DIR, params
from .dataset import load_dataset


def export():
    X, y = load_dataset(
        params["general"]["customer_ids"],
    )

    for id in params["general"]["customer_ids"]:
        group = y.xs(id)
        group = group[
            group.index
            >= datetime.datetime(2020, 9, 1, tzinfo=pytz.timezone("America/Montevideo"))
        ]
        group = group.dropna()
        group = group.sort_index()
        group.to_parquet(f"{EXPORT_DATA_DIR}/{id}.parquet")
    # Export logic here
    pass
