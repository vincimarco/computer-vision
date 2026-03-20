import plotly.express as px
import tqdm

from .config import IMGS_DIR
from .dataset import get_customer_ids, load_dataset


def plot():
    BATCH_SIZE = 1000

    imgs_dir = IMGS_DIR / "time_series"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    ids = get_customer_ids(
        [
            "BT 400 V",
            "MT 6.4 KV",
            "MT 15 KV",
            "MT 22 KV",
            "MT 31.5 KV",
            "MT 63 KV",
        ]
    )
    for i in tqdm.tqdm(
        range(0, len(ids), BATCH_SIZE),
        desc="Plotting customers...",
        total=len(ids) // BATCH_SIZE,
    ):
        batch = ids[i : i + BATCH_SIZE]
        X, y = load_dataset(batch)
        y = y.rename(columns={"value": "Consumption (Wh)"})
        for id in tqdm.tqdm(batch, desc="Plotting batch...", leave=False):
            fig = px.line(y.xs(id), title=f"Customer {id}", y="Consumption (Wh)")
            fig.write_html(imgs_dir / f"customer_{id}.html")
