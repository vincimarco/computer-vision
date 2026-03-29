import datetime

from .config import EXPORT_DATA_DIR, params
from .dataset import load_dataset


def export():
    X, y = load_dataset(params["general"]["customer_ids"])

    for id in params["general"]["customer_ids"]:
        group = y.xs(id)
        group.index = group.index + datetime.timedelta(669)
        group = group.sort_index()
        group.to_parquet(f"{EXPORT_DATA_DIR}/{id}.parquet")
    # Export logic here
    pass
