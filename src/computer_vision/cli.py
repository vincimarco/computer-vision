import pathlib

import cyclopts
import polars as pl

app = cyclopts.App()

preprocessing_app = app.command(cyclopts.App(name="preprocessing"))


@preprocessing_app.command()
def interim():
    RAW_DATA_DIR = pathlib.Path("data/1.raw/ECDUY")
    INTERIM_DATA_DIR = pathlib.Path("data/2.interim/ECDUY")
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DATA_DIR.glob("*.csv.tar.gz"))
    files.sort()
    for file in files:
        print(f"Processing {file}...")
        lf = pl.scan_csv(
            file, new_columns=["datetime", "id", "value"], ignore_errors=True
        )
        lf = lf.drop_nulls("datetime")
        lf = lf.with_columns(pl.from_epoch(pl.col("datetime")))
        lf.sink_parquet(INTERIM_DATA_DIR / f"{file.stem}.parquet")
