import plotly.express as px
import tqdm

from .config import IMGS_DIR
from .dataset import get_customer_ids, load_dataset


def plot():
    imgs_dir = IMGS_DIR / "time_series"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    tension_classes = [
        "BT 400 V",
        "MT 6.4 KV",
        "MT 15 KV",
        "MT 22 KV",
        "MT 31.5 KV",
        "MT 63 KV",
    ]

    for tension in (
        pbar := tqdm.tqdm(tension_classes, desc="Plotting tension classes...")
    ):
        pbar.set_description(f"Plotting tension class {tension}...")
        ids = get_customer_ids([tension])  # pyright: ignore[reportArgumentType]
        X, y = load_dataset(ids)
        y = y.rename(columns={"value": "Consumption (Wh)"})
        for id in tqdm.tqdm(ids, desc="Plotting batch...", leave=False):
            dir = imgs_dir / tension
            dir.mkdir(parents=True, exist_ok=True)
            fig = px.line(y.xs(id), title=f"Customer {id}", y="Consumption (Wh)")
            fig.write_html(dir / f"customer_{id}.html")
